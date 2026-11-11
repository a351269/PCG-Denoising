import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold, train_test_split

from pcg_denoising.utils.utils import get_pathology

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Fold:
    index: int
    train: list
    val: list
    test: list

    def paths(self, split, root):
        """Absolute paths of one split ("train", "val" or "test")."""
        return [Path(root) / p for p in getattr(self, split)]

    def pathology_counts(self):
        """{"train": {"AS": 12, ...}, "val": {...}, "test": {...}}"""
        return {s: dict(sorted(Counter(get_pathology(Path(p).name) for p in getattr(self, s)).items()))
                for s in SPLITS}


def list_target_files(root):
    """Audio files under `root`, as sorted paths relative to it."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _require_two_each(labels, what):
    small = sorted(k for k, c in Counter(labels).items() if c < 2)
    if small:
        raise ValueError(f"Cannot stratify {what}: pathology {small} has fewer than 2 recordings")


def make_folds(files, strategy, seed, val_fraction, k=None, test_size=None):
    """files: relative paths of the recordings. Returns a list of Fold."""
    if len(files) == 0:
        raise ValueError("No files to split")

    labels = [get_pathology(Path(f).name) for f in files]
    unknown = [f for f, p in zip(files, labels) if p is None]
    if unknown:
        raise ValueError(f"{len(unknown)} files have no pathology in their name, e.g. {unknown[0]}")

    files = np.array(files)
    pathology = np.array(labels)
    ids = np.arange(len(files))

    if strategy == "holdout":
        if test_size is None:
            raise ValueError("holdout needs test_size")
        _require_two_each(labels, "the holdout split")
        pairs = [train_test_split(ids, test_size=test_size, stratify=pathology, random_state=seed)]
    elif strategy in ("kfold", "stratified_kfold"):
        if not k or k < 2:
            raise ValueError(f"{strategy} needs k >= 2")
        if strategy == "kfold":
            splits = KFold(k, shuffle=True, random_state=seed).split(ids)
        else:
            splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, pathology)
        pairs = [(ids[a], ids[b]) for a, b in splits]
    elif strategy == "leave_pathology_out":
        pairs = [(ids[a], ids[b]) for a, b in LeaveOneGroupOut().split(ids, groups=pathology)]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    folds = []
    for i, (train, test) in enumerate(pairs):
        if val_fraction:
            _require_two_each(pathology[train].tolist(), f"the validation split of fold {i}")
            train, val = train_test_split(train, test_size=val_fraction,
                                          stratify=pathology[train], random_state=seed)
        else:
            val = np.array([], dtype=int)

        fold = Fold(i, files[np.sort(train)].tolist(), files[np.sort(val)].tolist(),
                    files[np.sort(test)].tolist())
        _check_fold(fold, files)
        folds.append(fold)
    return folds


def _check_fold(fold, files):
    sets = [set(fold.train), set(fold.val), set(fold.test)]
    if len(set().union(*sets)) != sum(len(s) for s in sets):
        raise AssertionError(f"fold {fold.index}: a recording is in more than one split")
    if set().union(*sets) != set(files.tolist()):
        raise AssertionError(f"fold {fold.index}: some recordings are in no split")


def create_folds(raw_target_path, folds_save_path, strategy="stratified_kfold", k=10,
                 val_fraction=0.1, test_size=None, seed=0):
    """val_fraction is the share of each fold's TRAINING recordings used for validation.
    For k-fold the test share is 1/k; for holdout it is test_size."""
    files = list_target_files(raw_target_path)
    folds = make_folds(files, strategy, seed, val_fraction, k, test_size)

    save_path = Path(folds_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(strategy=strategy, k=k, val_fraction=val_fraction, test_size=test_size, seed=seed)
    save_path.write_text(json.dumps({"settings": settings, "folds": [asdict(f) for f in folds]}, indent=2))
    return folds


def load_folds(folds_path, raw_target_path=None):
    """Load saved folds. With raw_target_path, also check that every recording still exists."""
    data = json.loads(Path(folds_path).read_text())
    folds = [Fold(**d) for d in data["folds"]]

    if raw_target_path is not None:
        root = Path(raw_target_path)
        missing = [p for f in folds for s in SPLITS for p in getattr(f, s) if not (root / p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} recordings from {folds_path} are missing under {root}, e.g. {missing[0]}")
    return folds


def print_fold_summary(folds):
    """Recordings per pathology in every split of every fold."""
    counts = [(f.index, f.pathology_counts()) for f in folds]
    pathologies = sorted({p for _, c in counts for split in c.values() for p in split})

    header = f"{'Fold':>4}  {'Split':<5}  {'Total':>5}  " + "  ".join(f"{p:>5}" for p in pathologies)
    print(header)
    print("-" * len(header))
    for index, c in counts:
        for split in SPLITS:
            row = c[split]
            print(f"{index:>4}  {split:<5}  {sum(row.values()):>5}  "
                  + "  ".join(f"{row.get(p, 0):>5}" for p in pathologies))