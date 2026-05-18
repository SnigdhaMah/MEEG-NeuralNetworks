"""
dataset/meeg_dataset.py

---------------------------------------------------------------------------------------
MEEG dataset loader with preprocessing matching the AT-DGNN paper:
  Xiao et al., "MEEG and AT-DGNN", BIBM 2024.
  Dataset: https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR

Raw format:
  subject_1.dat … subject_32.dat   (pickle, encoding='latin1')

  Each .dat:
    "data"   : np.ndarray  (20, 32, 59900)
                 20 trials × 32 channels × 59900 timepoints @ 1000 Hz
    "labels" : np.ndarray  (20, 2)
                 col 0 = valence  {0=low, 1=high}
                 col 1 = arousal  {0=low, 1=high}

Preprocessing:
  1. Downsample  1000 Hz → 200 Hz
  2. Bandpass filter  1–50 Hz  (Butterworth order 4)

Label encoding (4-class VA quadrant):
  0 HVHA  high-valence high-arousal   (excited / happy)
  1 HVLA  high-valence low-arousal    (calm / relaxed)
  2 LVHA  low-valence  high-arousal   (angry / fearful)
  3 LVLA  low-valence  low-arousal    (sad / bored)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly, butter, filtfilt, welch
from math import gcd

# Constants from paper
RAW_SRATE = 1000 # original recording rate
TARGET_SRATE = 200  # downsampled to this
BANDPASS = (1, 50) # bandpass range

# frequency bands(from paper)
FREQ_BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  14),
    "beta":  (14, 31),
    "gamma": (31, 50),
}
N_BANDS = len(FREQ_BANDS)   # 5

# 32-ch 10-20 layout (same as DEAP)
CH_NAMES = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7",
    "CP5","CP1","P3","P7","PO3","O1","Oz","Pz",
    "Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz",
    "C4","T8","CP6","CP2","P4","P8","PO4","O2",
]
N_CHANNELS = 32
N_SUBJECTS = 32
N_TRIALS   = 20

# 4-class label map  (valence_bit, arousal_bit) → class index
_VA_TO_CLASS = {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}
CLASS_NAMES  = {0: "HVHA", 1: "HVLA", 2: "LVHA", 3: "LVLA"}


##################### Preprocessing ##########################
def downsample(data: np.ndarray, orig: int, target: int) -> np.ndarray:
    g = gcd(orig, target)
    up, down = target // g, orig // g
    return resample_poly(data, up, down, axis=-1).astype(np.float32)


def bandpass(data: np.ndarray, sfreq: float,
              lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq  = sfreq / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def preprocess_trial(trial: np.ndarray) -> np.ndarray:
    """
    trial: (32, 59900)  @ 1000 Hz
    returns : (32, 11980)  @ 200 Hz, bandpass 1-50 Hz
    """
    downsampled = downsample(trial, RAW_SRATE, TARGET_SRATE)
    filtered = bandpass(downsampled, TARGET_SRATE, BANDPASS[0], BANDPASS[1])
    return filtered


################## Feature extraction (same as MEEG paper) #####################
def compute_psd(segment: np.ndarray, sfreq: float = TARGET_SRATE) -> np.ndarray:
    """
    Log-mean PSD per frequency band.
    segment : (32, n_times)  @ sfreq Hz
    returns : (32, 5)
    """
    nperseg = min(int(sfreq * 2), segment.shape[-1])
    freqs, psd = welch(segment, fs=sfreq, nperseg=nperseg)
    out = []
    for lo, hi in FREQ_BANDS.values():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        val = (np.log1p(psd[:, idx].mean(axis=-1))
               if len(idx) > 0 else np.zeros(segment.shape[0]))
        out.append(val.astype(np.float32))
    return np.stack(out, axis=-1)


def compute_de(segment: np.ndarray, sfreq: float = TARGET_SRATE) -> np.ndarray:
    """
    Differential Entropy per band: 0.5 * log(2πe·σ²).
    segment : (32, n_times)  @ sfreq Hz
    returns : (32, 5)
    """
    nyq = sfreq / 2.0
    out = []
    for lo, hi in FREQ_BANDS.values():
        b, a = butter(4, [lo / nyq, min(hi / nyq, 0.999)], btype="band")
        filt = filtfilt(b, a, segment, axis=-1)
        de   = 0.5 * np.log(2 * np.pi * np.e * filt.var(axis=-1) + 1e-8)
        out.append(de.astype(np.float32))
    return np.stack(out, axis=-1)


####################### Raw data loader ###########################

def load_meeg_raw(root: str, window_sec: float = 4.0, overlap_sec: float = 2.0, apply_preproc: bool  = True,) -> list:
    """
    Load all MEEG .dat files, preprocess, and segment into windows.

    [Parameters]
    root : local directory containing subject_*.dat files.
    window_sec : sliding window length in seconds (post-200 Hz).
    overlap_sec : overlap between consecutive windows.
    apply_preproc : if True (default), run downsample + bandpass per trial. Set False if you load already-processed files.

    [Returns]
    List of dicts, one per window:
      {
        "subject_id" : str           | e.g. "subject_1"
        "trial_idx"  : int           | 0-based within subject  (0-19)
        "segment"    : np.ndarray    | (32, win_samples) @ 200 Hz, float32
        "label"      : int           | 0-3  (HVHA / HVLA / LVHA / LVLA)
        "valence"    : int           | 0 or 1
        "arousal"    : int           | 0 or 1
      }
    """
    sfreq        = float(TARGET_SRATE)
    win_samples  = int(window_sec  * sfreq)
    step_samples = int((window_sec - overlap_sec) * sfreq)
    if step_samples <= 0:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be less than window_sec ({window_sec})")
    dat_files = sorted(
        (f for f in os.listdir(root) if f.endswith(".dat")),
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    all_samples = []
    for fname in dat_files:
        subject_id = fname.replace(".dat", "")
        with open(os.path.join(root, fname), "rb") as fh:
            raw = pickle.load(fh, encoding="latin1")
        data   = raw["data"].astype(np.float32)   # (20, 32, 59900)
        labels = raw["labels"]                    # (20, 2)
        assert data.shape[1] == N_CHANNELS, (
            f"{fname}: expected {N_CHANNELS} channels, got {data.shape[1]}")
        for trial_idx in range(data.shape[0]):
            trial = data[trial_idx]                 # (32, 59900)
            if apply_preproc:
                trial = preprocess_trial(trial)     # (32, 11980) @ 200 Hz
            val_bit = int(labels[trial_idx, 0])
            aro_bit = int(labels[trial_idx, 1])
            label   = _VA_TO_CLASS[(val_bit, aro_bit)]
            n_times = trial.shape[-1]
            start = 0
            while start + win_samples <= n_times:
                all_samples.append({
                    "subject_id": subject_id,
                    "trial_idx":  trial_idx,
                    "segment":    trial[:, start : start + win_samples].copy(),
                    "label":      label,
                    "valence":    val_bit,
                    "arousal":    aro_bit,
                })
                start += step_samples
    _report(all_samples, len(dat_files), window_sec, overlap_sec)
    return all_samples


def _report(samples, n_files, window_sec, overlap_sec):
    from collections import Counter
    n_win = int(window_sec  * TARGET_SRATE)
    n_stp = int((window_sec - overlap_sec) * TARGET_SRATE)
    print(f"Loaded {n_files} subjects → {len(samples)} windows  "
          f"({window_sec}s / {overlap_sec}s overlap / "
          f"{n_win} samples @ {TARGET_SRATE} Hz)")
    counts = Counter(s["label"] for s in samples)
    total  = len(samples)
    dist   = "  ".join(
        f"{CLASS_NAMES[k]}={v}({v/total*100:.0f}%)"
        for k, v in sorted(counts.items()))
    print(f"Label dist: {dist}\n")


###################### Dataset class ##########################

class MEEGDataset(Dataset):
    """
    [Parameters]
    samples   : output of load_meeg_raw()
    feature   : "psd" | "de" | "raw"
                psd / de  →  Tensor (32, 5)
                raw       →  Tensor (32, win_samples)
    transform : optional callable applied to the feature tensor
    task      : "emotion"  →  4-class label (0-3)
                "valence"  →  binary label  (0/1)
                "arousal"  →  binary label  (0/1)
    """

    def __init__(self, samples: list, feature: str = "psd",
                 transform=None, task: str = "emotion"):
        assert feature in ("psd", "de", "raw"), \
            f"feature must be psd/de/raw; got {feature!r}"
        assert task in ("emotion", "valence", "arousal"), \
            f"task must be emotion/valence/arousal; got {task!r}"
        self.samples   = samples
        self.feature   = feature
        self.transform = transform
        self.task      = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        seg = s["segment"]    # (32, win_samples)
        if self.feature == "psd":
            feat = compute_psd(seg)
        elif self.feature == "de":
            feat = compute_de(seg)
        else:
            feat = seg
        feat  = torch.from_numpy(feat)   # (32, n_feat)
        label = (s["label"]   if self.task == "emotion" else
                 s["valence"] if self.task == "valence"  else
                 s["arousal"])
        if self.transform is not None:
            feat = self.transform(feat)
        return {
            "feature":    feat,
            "label":      torch.tensor(label, dtype=torch.long),
            "subject_id": s["subject_id"],
            "trial_idx":  torch.tensor(s["trial_idx"], dtype=torch.long),
        }

    ############## GNN branch helper ###############################
    @staticmethod
    def electrode_adjacency(threshold: float = None) -> torch.Tensor:
        """
        (32, 32) adjacency matrix from 2-D projected 10-20 positions.
        threshold=None → fully connected (minus self-loops).
        Pass e.g. threshold=0.5 for distance-pruned sparse graph.
        """
        POS = {
            "Fp1":(-0.18, 0.95),"AF3":(-0.35, 0.85),"F3": (-0.55, 0.60),
            "F7": (-0.80, 0.55),"FC5":(-0.80, 0.25),"FC1":(-0.35, 0.25),
            "C3": (-0.72, 0.00),"T7": (-1.00, 0.00),"CP5":(-0.80,-0.25),
            "CP1":(-0.35,-0.25),"P3": (-0.55,-0.60),"P7": (-0.80,-0.55),
            "PO3":(-0.35,-0.85),"O1": (-0.18,-0.95),"Oz": ( 0.00,-1.00),
            "Pz": ( 0.00,-0.50),"Fp2":( 0.18, 0.95),"AF4":( 0.35, 0.85),
            "Fz": ( 0.00, 0.50),"F4": ( 0.55, 0.60),"F8": ( 0.80, 0.55),
            "FC6":( 0.80, 0.25),"FC2":( 0.35, 0.25),"Cz": ( 0.00, 0.00),
            "C4": ( 0.72, 0.00),"T8": ( 1.00, 0.00),"CP6":( 0.80,-0.25),
            "CP2":( 0.35,-0.25),"P4": ( 0.55,-0.60),"P8": ( 0.80,-0.55),
            "PO4":( 0.35,-0.85),"O2": ( 0.18,-0.95),
        }
        coords = np.array([POS[ch] for ch in CH_NAMES])
        diff   = coords[:, None] - coords[None, :]
        dist   = np.sqrt((diff ** 2).sum(-1))
        adj    = (1.0 - np.eye(N_CHANNELS) if threshold is None
                  else (dist < threshold).astype(float))
        np.fill_diagonal(adj, 0.0)
        return torch.tensor(adj, dtype=torch.float32)