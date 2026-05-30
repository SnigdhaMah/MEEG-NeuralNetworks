"""
data/splits.py

---------------------------------------------------------------------------------------
Split strategies for MEEG experiments.

Three strategies matching common EEG evaluation protocols:

1. trial_kfold  (paper standard)
   Trial-wise K-fold cross-validation.
   All windows from trial i go entirely to one fold.
   Prevents leakage from overlapping sliding windows.
   → Use this to reproduce / compare against AT-DGNN results.

2. cross_trial
   Same-subject, different-trial train/val/test split.
   Each subject can appear in every partition, but each trial belongs to
   exactly one partition.
   → Use this as the within-subject baseline in data/dataset.md.

3. cross_subject
   Hold out a subset of subjects entirely.
   No subject appears in both train and test.
   → Use this to test generalisation to new subjects.

Usage
-----
from data.meeg_dataset import load_meeg_raw
from data.splits import (
    get_trial_kfold_splits,
    get_cross_trial_splits,
    get_cross_subject_splits,
)

samples = load_meeg_raw("MEEG/")

# ── Reproduce AT-DGNN evaluation ─────────────────────────────────────────────
# Returns a generator of (train, val, test) triples, one per fold.
for fold, (train_s, val_s, test_s) in enumerate(get_trial_kfold_splits(samples, k=10)):
    ...   # train/evaluate model on this fold

# ── Same-subject, different-trial split ───────────────────────────────────────
train_s, val_s, test_s = get_cross_trial_splits(samples, seed=42)

# ── Cross-subject split ───────────────────────────────────────────────────────
train_s, val_s, test_s = get_cross_subject_splits(samples, seed=42)
"""

import random
from collections import defaultdict
from typing import List, Tuple, Iterator

Sample = dict


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Trial-wise K-fold  (paper standard)
# ─────────────────────────────────────────────────────────────────────────────

def get_trial_kfold_splits(
    samples    : List[Sample],
    k          : int  = 10,
    seed       : int  = 42,
    val_folds  : int  = 1,
) -> Iterator[Tuple[List[Sample], List[Sample], List[Sample]]]:
    """
    Trial-wise K-fold cross-validation, matching the AT-DGNN paper.

    All windows belonging to the same (subject, trial_idx) are kept
    together — they never split across train/test — to prevent leakage
    from overlapping windows.

    Parameters
    ----------
    samples   : flat list from load_meeg_raw()
    k         : number of outer folds (paper uses 10)
    seed      : random seed for fold assignment
    val_folds : number of folds to use as validation within train set
                (paper uses 4-fold inner CV; set val_folds=1 for a
                simple train/val/test split per outer fold)

    Yields
    ------
    (train_samples, val_samples, test_samples) for each outer fold.
    All windows from a trial are in exactly one partition.
    """
    rng = random.Random(seed)

    # Collect unique (subject, trial) keys, then group their windows
    by_trial = defaultdict(list)
    for s in samples:
        by_trial[(s["subject_id"], s["trial_idx"])].append(s)

    trial_keys = sorted(by_trial.keys())
    rng.shuffle(trial_keys)

    # Assign each trial to a fold (round-robin after shuffle)
    fold_of = {key: i % k for i, key in enumerate(trial_keys)}

    for test_fold in range(k):
        test_keys  = [key for key, f in fold_of.items() if f == test_fold]
        other_keys = [key for key, f in fold_of.items() if f != test_fold]

        # val: take the next val_folds folds (wrapping) from the remaining
        val_folds_ids = [(test_fold + 1 + i) % k for i in range(val_folds)]
        val_keys   = [key for key in other_keys
                      if fold_of[key] in val_folds_ids]
        train_keys = [key for key in other_keys
                      if fold_of[key] not in val_folds_ids]

        train = [s for key in train_keys for s in by_trial[key]]
        val   = [s for key in val_keys   for s in by_trial[key]]
        test  = [s for key in test_keys  for s in by_trial[key]]

        _report_fold(test_fold, k, train, val, test)
        yield train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Cross-trial holdout  (within-subject baseline)
# ─────────────────────────────────────────────────────────────────────────────

def get_cross_trial_splits(
    samples    : List[Sample],
    test_ratio : float = 0.2,
    val_ratio  : float = 0.1,
    train_trials: int = None,
    val_trials  : int = None,
    test_trials : int = None,
    shuffle_trials: bool = True,
    seed       : int   = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Split each subject's trials into train/val/test partitions.

    This is the within-subject baseline described in data/dataset.md:
    every subject can appear in all three partitions, but all overlapping
    windows from the same (subject, trial_idx) stay together.

    Parameters
    ----------
    samples        : flat list from load_meeg_raw()
    test_ratio     : fraction of each subject's trials to reserve for test.
                     Ignored if test_trials is set.
    val_ratio      : fraction of the remaining trials to reserve for validation.
                     Ignored if val_trials is set.
    train_trials   : optional exact number of train trials per subject.
                     If set, val_trials and test_trials should also be set.
    val_trials     : optional exact number of validation trials per subject.
    test_trials    : optional exact number of test trials per subject.
    shuffle_trials : if True, shuffle each subject's trials before splitting.
                     If False, split by ascending trial_idx, so the first
                     trials go to train, then validation, then test.
    seed           : RNG seed used when shuffle_trials=True

    Returns
    -------
    (train_samples, val_samples, test_samples)
    All windows from a trial are in exactly one partition.
    """
    rng = random.Random(seed)
    by_trial = defaultdict(list)
    trials_by_subject = defaultdict(list)

    for s in samples:
        key = (s["subject_id"], s["trial_idx"])
        if key not in by_trial:
            trials_by_subject[s["subject_id"]].append(key)
        by_trial[key].append(s)

    train_keys, val_keys, test_keys = [], [], []
    for subject_id in sorted(trials_by_subject):
        keys = sorted(trials_by_subject[subject_id])
        if shuffle_trials:
            rng.shuffle(keys)

        n_trials = len(keys)
        if any(v is not None for v in (train_trials, val_trials, test_trials)):
            if None in (train_trials, val_trials, test_trials):
                raise ValueError(
                    "train_trials, val_trials, and test_trials must be set together."
                )
            if train_trials < 0 or val_trials < 0 or test_trials < 0:
                raise ValueError("trial counts must be non-negative.")
            requested = train_trials + val_trials + test_trials
            if requested > n_trials:
                raise ValueError(
                    f"{subject_id}: requested {requested} trials but only "
                    f"{n_trials} are available."
                )

            train_keys.extend(keys[:train_trials])
            val_start = train_trials
            val_end = val_start + val_trials
            val_keys.extend(keys[val_start:val_end])
            test_keys.extend(keys[val_end:val_end + test_trials])
            continue

        n_test = max(1, round(n_trials * test_ratio))
        remaining = keys[:n_trials - n_test]
        test = keys[n_trials - n_test:]
        n_val = max(1, round(len(remaining) * val_ratio)) if remaining else 0

        train_keys.extend(remaining[:len(remaining) - n_val])
        val_keys.extend(remaining[len(remaining) - n_val:])
        test_keys.extend(test)

    train = [s for key in train_keys for s in by_trial[key]]
    val = [s for key in val_keys for s in by_trial[key]]
    test = [s for key in test_keys for s in by_trial[key]]

    print(f"\n── Cross-trial split ──")
    print(f"  train: {len(train_keys)} trials → {len(train)} windows")
    print(f"  val:   {len(val_keys)} trials → {len(val)} windows")
    print(f"  test:  {len(test_keys)} trials → {len(test)} windows")
    _report_label_dist("train", train)
    _report_label_dist("test", test)
    print()
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Cross-subject
# ─────────────────────────────────────────────────────────────────────────────

def get_cross_subject_splits(
    samples         : List[Sample],
    test_ratio      : float          = 0.2,
    val_ratio       : float          = 0.1,
    seed            : int            = 42,
    test_subjects   : List[str]      = None,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Hold out entire subjects for test (and val).

    All windows from subject X go entirely into one partition,
    so there is no cross-subject leakage.

    Parameters
    ----------
    samples       : flat list from load_meeg_raw()
    test_ratio    : fraction of subjects for test  (ignored if test_subjects given)
    val_ratio     : fraction of remaining subjects for val
    seed          : RNG seed
    test_subjects : explicit list of subject_id strings to use as test set

    Returns
    -------
    (train_samples, val_samples, test_samples)
    """
    rng      = random.Random(seed)
    all_subj = sorted({s["subject_id"] for s in samples})
    n        = len(all_subj)

    if test_subjects is not None:
        test_set  = set(test_subjects)
        remaining = [s for s in all_subj if s not in test_set]
    else:
        shuffled  = all_subj[:]
        rng.shuffle(shuffled)
        n_test   = max(1, round(n * test_ratio))
        test_set  = set(shuffled[:n_test])
        remaining = shuffled[n_test:]

    rng.shuffle(remaining)
    n_val     = max(1, round(len(remaining) * val_ratio))
    val_set   = set(remaining[:n_val])
    train_set = set(remaining[n_val:])

    train = [s for s in samples if s["subject_id"] in train_set]
    val   = [s for s in samples if s["subject_id"] in val_set]
    test  = [s for s in samples if s["subject_id"] in test_set]

    print(f"\n── Cross-subject split ──")
    print(f"  train: {len(train_set)} subjects → {len(train)} windows")
    print(f"  val:   {len(val_set)} subjects → {len(val)} windows")
    print(f"  test:  {len(test_set)} subjects → {len(test)} windows")
    _report_label_dist("train", train)
    _report_label_dist("test",  test)
    print()
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _report_fold(fold_idx, k, train, val, test):
    if fold_idx == 0 or fold_idx == k - 1:   # only print first and last fold
        print(f"  Fold {fold_idx+1}/{k}: "
              f"train={len(train)}  val={len(val)}  test={len(test)} windows")


def _report_label_dist(name, split):
    from collections import Counter
    if not split:
        return
    class_names = {0: "HVHA", 1: "HVLA", 2: "LVHA", 3: "LVLA"}
    counts = Counter(s["label"] for s in split)
    total  = len(split)
    dist   = "  ".join(
        f"{class_names.get(k, k)}={v}({v/total*100:.0f}%)"
        for k, v in sorted(counts.items()))
    print(f"  {name} labels: {dist}")
