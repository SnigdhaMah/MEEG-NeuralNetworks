# MEEG Dataset — Usage Guide

## Overview

| Property | Value |
|---|---|
| Subjects | 32 |
| Trials per subject | 20 (1-min music clips) |
| Channels | 32 (10-20 system, same layout as DEAP) |
| Raw sampling rate | 1000 Hz |
| Preprocessed rate | 200 Hz |
| Labels | Valence × Arousal (binary each → 4 classes) |
| Source paper | Xiao et al., *MEEG and AT-DGNN*, BIBM 2024 |

---

## Setup

### 1. Install dependencies

```bash
pip install scipy numpy torch
```

### 2. Get the data

Download all 32 `.dat` files from Google Drive and place them in a local folder.

1. Open the [MEEG Google Drive folder](https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR)
2. Download all 32 `subject_*.dat` files
3. Place them under `data/raw/MEEG/` (or any path you prefer):

```
data/raw/MEEG/
  subject_1.dat
  subject_2.dat
  ...
  subject_32.dat
```

Alternatively, use `gdown` in a terminal to download the whole folder at once:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Tabw5sjpFiwy88yP-C-LnunNFrrre9AR -O data/raw/MEEG
```

---

## Loading the data

### `load_meeg_raw(root, window_sec, overlap_sec, apply_preproc)`

Loads all subject files, applies preprocessing, and segments each trial into sliding windows. Returns a flat list of window dicts.

```python
from data.meeg_dataset import load_meeg_raw

samples = load_meeg_raw(
    root          = "data/raw/MEEG",
    window_sec    = 4.0,    # window length in seconds
    overlap_sec   = 2.0,    # overlap between consecutive windows
    apply_preproc = True,   # set False if files are already preprocessed
)
```

**Preprocessing** (`apply_preproc=True`, matches paper Section III-A):

1. Downsample 1000 Hz → 200 Hz (polyphase resampling)
2. Bandpass 1–50 Hz (4th-order Butterworth, zero-phase)

After preprocessing, each trial is `(32, 11980)` — 32 channels × ≈59.9 s @ 200 Hz.

**Sliding window** — with the defaults (4 s, 2 s overlap, 200 Hz):

| Parameter | Value |
|---|---|
| Window size | 800 samples (4 s × 200 Hz) |
| Step size | 400 samples (2 s step) |
| Windows per trial | ≈ 29 |
| Total windows (32 subjects × 20 trials) | ≈ 18,560 |

**Each window dict** contains:

```python
{
    "subject_id" : "subject_1",   # str
    "trial_idx"  : 3,             # int, 0-based within subject (0–19)
    "segment"    : np.ndarray,    # (32, 800) float32 — channels × samples
    "label"      : 0,             # int, 4-class (see label table below)
    "valence"    : 1,             # int, 0 = low / 1 = high
    "arousal"    : 1,             # int, 0 = low / 1 = high
}
```

**4-class label encoding**

| Class | Name | Valence | Arousal | Example emotion |
|---|---|---|---|---|
| 0 | HVHA | high | high | Excited, happy |
| 1 | HVLA | high | low  | Calm, relaxed  |
| 2 | LVHA | low  | high | Angry, fearful |
| 3 | LVLA | low  | low  | Sad, bored     |

---

## Creating a PyTorch Dataset

Wrap the sample list in `MEEGDataset` to get a standard PyTorch `Dataset`.
Two parameters control what the model receives:

**`feature`** — what goes into `batch["feature"]`

| Value | Output shape | Notes |
|---|---|---|
| `"psd"` | `(32, 5)` | Log-mean PSD per band — recommended for GNN / compact models |
| `"de"`  | `(32, 5)` | Differential Entropy per band — alternative spectral feature |
| `"raw"` | `(32, 800)` | Raw time-series — required for temporal CNN / Transformer |

**`task`** — what goes into `batch["label"]`

| Value | Label values | Use when |
|---|---|---|
| `"valence"` | 0 / 1 (binary) | replicating paper valence results |
| `"arousal"` | 0 / 1 (binary) | replicating paper arousal results |
| `"emotion"` | 0–3 (4-class)  | joint VA quadrant classification |

```python
from data.meeg_dataset import MEEGDataset

train_ds = MEEGDataset(train_samples, feature="psd", task="valence")
val_ds   = MEEGDataset(val_samples,   feature="psd", task="valence")
test_ds  = MEEGDataset(test_samples,  feature="psd", task="valence")
```

**Batch format** — every `DataLoader` batch is a dict:

```python
{
    "feature"    : Tensor,  # (B, 32, n_feat)  — model input
    "label"      : Tensor,  # (B,)  int64
    "subject_id" : list,    # (B,)  strings
    "trial_idx"  : Tensor,  # (B,)  int64
}
```

---

## GNN adjacency matrix

The GNN branch needs an electrode adjacency matrix derived from 10-20 positions:

```python
adj = MEEGDataset.electrode_adjacency()               # (32, 32) fully connected
adj = MEEGDataset.electrode_adjacency(threshold=0.5)  # (32, 32) distance-pruned
```

---

## Evaluation protocol

### Configurations run by all three models

Every model (GNN, Transformer, CNN) is evaluated under the same three feature / task configurations so results are directly comparable:

| Config | `feature` | `task` | Label type |
|---|---|---|---|
| A | `"psd"` | `"valence"` | binary (0 / 1) |
| B | `"psd"` | `"arousal"` | binary (0 / 1) |
| C | `"raw"` | `"emotion"` | 4-class (0–3) |

Configs A and B allow direct comparison against the AT-DGNN paper numbers (valence 86.01 %, arousal 83.74 %).  
Config C tests joint VA quadrant classification using the raw time-series, which is the most demanding setting.

---

### Experiment 1 — same-subject, different trials (within-subject baseline)

**Goal:** establish how well each model learns when trained and tested on the *same* subjects. Because MEEG has only one recording session per subject, "different sessions" is approximated by a trial-level split: for each subject, 80 % of their 20 trials are first kept for the train/validation pool and 20 % are held out for testing. All windows from the same trial stay in the same partition to prevent leakage from overlapping windows.

This is repeated for three subject-count settings to produce a learning curve:

| Run | Subjects used | Train trials/subject | Val trials/subject | Test trials/subject |
|---|---|---|---|---|
| 1 | 32 (all) | 14 | 2 | 4 |
| 2 | 20 | 14 | 2 | 4 |
| 3 | 10 | 14 | 2 | 4 |

For runs 2–3, subjects are selected deterministically in the example below. The test set always contains the *same subjects* that were trained on, but different trials.

```python
from data.meeg_dataset import load_meeg_raw, MEEGDataset
from data.splits import get_cross_trial_splits

all_samples = load_meeg_raw("data/raw/MEEG")

for n_subjects in [32, 20, 10]:
    # Restrict to the first n_subjects by numeric subject id
    subset_ids = sorted(
        {s["subject_id"] for s in all_samples},
        key=lambda sid: int(sid.split("_")[-1]),
    )[:n_subjects]
    samples    = [s for s in all_samples if s["subject_id"] in subset_ids]

    # Trial-level split: same subjects in train/val/test, different trials.
    # With 20 trials/subject, test_ratio=0.2 and val_ratio=0.1 produce
    # 14 train trials, 2 validation trials, and 4 test trials per subject.
    train_s, val_s, test_s = get_cross_trial_splits(samples, test_ratio=0.2,
                                                     val_ratio=0.1, seed=42)

    for config, feature, task in [("A", "psd", "valence"),
                                   ("B", "psd", "arousal"),
                                   ("C", "raw", "emotion")]:
        train_ds = MEEGDataset(train_s, feature=feature, task=task)
        val_ds   = MEEGDataset(val_s,   feature=feature, task=task)
        test_ds  = MEEGDataset(test_s,  feature=feature, task=task)
        # ... train model, record accuracy for (n_subjects, config) ...
```

`get_cross_trial_splits` is implemented directly in `data/splits.py`. It
assigns whole trials to train/val/test and keeps all overlapping windows from
the same `(subject_id, trial_idx)` in one partition to prevent leakage.

Two split styles are supported:

```python
# Ratio-based split. If shuffle_trials=False, this uses early trials for train,
# then validation, then late trials for test.
train_s, val_s, test_s = get_cross_trial_splits(
    samples,
    test_ratio=0.2,
    val_ratio=0.1,
    shuffle_trials=False,
)

# Exact trial-count split. With 20 trials/subject, this means:
# trial 0-13 -> train, trial 14-15 -> val, trial 16-19 -> test
# when shuffle_trials=False.
train_s, val_s, test_s = get_cross_trial_splits(
    samples,
    train_trials=14,
    val_trials=2,
    test_trials=4,
    shuffle_trials=False,
)
```

Set `shuffle_trials=True` to randomly assign trials within each subject before
splitting. Set `shuffle_trials=False` when you want a fixed split by ascending
`trial_idx`, such as "first N trials for train, last M trials for test".

---

### Experiment 2 — cross-subject generalisation

**Goal:** measure how well models trained on a set of subjects generalise to *held-out subjects they have never seen*. The same three subject-count settings are used, but now the test set contains different subjects from the training set.

| Run | Subjects used | Train subjects | Val subjects | Test subjects |
|---|---|---|---|---|
| 1 | 32 | 23 | 2 | 7 |
| 2 | 25 | 18 | 2 | 5 |
| 3 | 15 | 11 | 1 | 3 |

Subject assignment is fixed across all models and configs (same seed) so that results are comparable.

```python
from data.meeg_dataset import load_meeg_raw, MEEGDataset
from data.splits import get_cross_subject_splits

all_samples = load_meeg_raw("data/raw/MEEG")

for n_total, n_test in [(32, 7), (25, 5), (15, 3)]:
    subset_ids = sorted(
        {s["subject_id"] for s in all_samples},
        key=lambda sid: int(sid.split("_")[-1]),
    )[:n_total]
    samples = [s for s in all_samples if s["subject_id"] in subset_ids]

    train_s, val_s, test_s = get_cross_subject_splits(
        samples,
        test_ratio = n_test / n_total,
        val_ratio  = 0.1,
        seed       = 42,
    )

    for config, feature, task in [("A", "psd", "valence"),
                                   ("B", "psd", "arousal"),
                                   ("C", "raw", "emotion")]:
        train_ds = MEEGDataset(train_s, feature=feature, task=task)
        val_ds   = MEEGDataset(val_s,   feature=feature, task=task)
        test_ds  = MEEGDataset(test_s,  feature=feature, task=task)
        # ... train model, record accuracy for this subject-count setting ...
```

---

### Expected result structure

Running both experiments produces a table of the form:

| Model | Config | Exp 1 (32 subj) | Exp 1 (25) | Exp 1 (15) | Exp 1 (10) | Exp 2 (25→7) | Exp 2 (20→5) | Exp 2 (12→3) | Exp 2 (8→2) |
|---|---|---|---|---|---|---|---|---|---|
| GNN | A (valence) | — | — | — | — | — | — | — | — |
| GNN | B (arousal) | — | — | — | — | — | — | — | — |
| GNN | C (emotion) | — | — | — | — | — | — | — | — |
| Transformer | A | — | … | | | | | | |
| CNN | A | — | … | | | | | | |

The gap between Exp 1 and Exp 2 at each subject count quantifies how much performance degrades when moving from within-subject to cross-subject generalisation — this is the main finding the experiments are designed to show.

---

### AT-DGNN baseline training settings (from paper)

Use these as a reference when comparing against the published results:

| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 1e-3 (stage 1), 1e-4 (stage 2) |
| Loss | Cross-entropy |
| Stage 1 epochs | 200 (with early stopping) |
| Stage 2 epochs | 20 (fine-tune on all folds, stop at 100 % train acc) |
| Evaluation | Trial-wise 10-fold CV (outer) + 4-fold CV (inner) |
| Window size | 100 samples (= `fs / 2` at 200 Hz) |
| Temporal kernel sizes | 100, 50, 25 samples |
| DGNN layers | 3 |
| Temporal learner layers | 3 |

Paper targets (AT-DGNN-Gen): **valence 86.01 %**, **arousal 83.74 %**
