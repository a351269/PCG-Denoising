import numpy as np
import pandas as pd
from librosa import load
from pathlib import Path

from pcg_denoising.utils.utils import get_pathology
from pcg_denoising.utils.noise import SISDR

COLUMNS = ["fold", "noise_type", "dB", "kind", "pathology", "file",
           "sisdr", "sisdr_noisy", "sisdr_improvement", "run_time"]
CONDITION = ["fold", "noise_type", "dB"]


def sisdr_scores(references, estimates, sr):
    """SI-SDR of every estimate against its reference. Both lists in the same order,
    one float per pair, returned in that order."""
    if len(references) != len(estimates):
        raise ValueError(f"{len(references)} references but {len(estimates)} estimates")

    scores = []
    for ref, est in zip(references, estimates):
        a, _ = load(ref, sr=sr)
        b, _ = load(est, sr=sr)
        if len(a) != len(b):
            raise ValueError(f"Length differs for {Path(ref).name} ({len(a)}) and {Path(est).name} ({len(b)})")
        scores.append(float(SISDR(a, b)))
    return np.array(scores)


def score_condition(fold, noise_type, dB, files, targets, noisy, predictions, kinds, sr, run_time):
    """One row per recording for one fold and one evaluation condition.
    files: relative paths (fold.test); targets / noisy / predictions / kinds: same order as files.
    run_time: average seconds per audio of this run."""
    if not (len(files) == len(targets) == len(noisy) == len(predictions) == len(kinds)):
        raise ValueError("files, targets, noisy, predictions and kinds must have the same length")

    sisdr = sisdr_scores(targets, predictions, sr)
    sisdr_noisy = sisdr_scores(targets, noisy, sr)
    return pd.DataFrame({
        "fold": fold, "noise_type": noise_type, "dB": dB, "kind": list(kinds),
        "pathology": [get_pathology(Path(f).name) for f in files],
        "file": [Path(f).as_posix() for f in files],
        "sisdr": sisdr, "sisdr_noisy": sisdr_noisy, "sisdr_improvement": sisdr - sisdr_noisy,
        "run_time": run_time,
    }, columns=COLUMNS)


def load_results(csv_path):
    return pd.read_csv(csv_path)


def completed(csv_path):
    """{(fold, noise_type, dB)} already in the csv, to skip them when resuming."""
    if not Path(csv_path).exists():
        return set()
    df = load_results(csv_path)
    return set(map(tuple, df[CONDITION].drop_duplicates().itertuples(index=False, name=None)))


def append_results(csv_path, new):
    """Add rows to the csv. Rows of a (fold, noise_type, dB) that is already there are replaced,
    so running a condition again does not duplicate it."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = load_results(path)
        keys = set(map(tuple, new[CONDITION].drop_duplicates().itertuples(index=False, name=None)))
        keep = [tuple(r) not in keys for r in old[CONDITION].itertuples(index=False, name=None)]
        new = pd.concat([old[keep], new], ignore_index=True)

    tmp = path.with_name(path.stem + ".tmp.csv")
    new.to_csv(tmp, index=False)
    tmp.replace(path)