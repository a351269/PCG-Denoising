"""
Helpers for visualizations.
"""
import numpy as np
import pandas as pd
from librosa import load
from pathlib import Path
from dataclasses import dataclass
from scipy.signal import welch
import matplotlib.pyplot as plt

import pcg_denoising.splits as SP
import pcg_denoising.utils.time_frequency as TF
from pcg_denoising.config import FoldDirs, find_root
from pcg_denoising.utils.utils import get_pathology

EPS = 1e-10
KINDS = ("AWGN", "APGN", "PhysioNet", "Speech")
COLORS = {"target": "#20B800", "noisy": "#D81B60", "prediction": "#1F00AA"}
PATHOLOGY_NAMES = {"AS": "aortic stenosis", "MS": "mitral stenosis", "MVP": "mitral valve prolapse",
                   "MR": "mitral regurgitation", "N": "normal"}
PANEL_TITLES = {"STFT": "STFT {} spectrogram", "WSST": "WSST {} scalogram", "CWT": "CWT {} scalogram",
                "S": r"$\mathit{{S}}$-transform {} scalogram"}
plt.rcParams["svg.fonttype"] = "none"          # keep text as text in the SVG


# ---- finding the files of one recording ------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """One test recording under one evaluation condition."""
    run_dir: Path
    fold_idx: int
    signal: str                     # relative path, e.g. "AS/Card_Clean_AS_0001.wav"
    pathology: str | None           # "AS", "MS", "MVP", "MR" or "N"
    noise_type: str
    noise_level: float
    target: Path
    noisy: Path
    prediction: Path
    kind: str | None                # noise actually applied to this recording (differs from noise_type for "Sum")
    sisdr: float | None             # SI-SDR of the prediction, dB
    sisdr_noisy: float | None       # SI-SDR of the noisy input, dB

    @property
    def pathology_name(self):
        return PATHOLOGY_NAMES.get(self.pathology, str(self.pathology))


def pick_signal(cfg, fold_idx, pathology, n=0):
    """The n-th test recording (sorted order) of `pathology` in fold `fold_idx`, as a relative path."""
    folds = SP.load_folds(Path(cfg.folds_json))
    fold = next((f for f in folds if f.index == fold_idx), None)
    if fold is None:
        raise ValueError(f"{cfg.experiment.experiment_name} has no fold {fold_idx} (folds: {[f.index for f in folds]})")
    files = [f for f in fold.test if get_pathology(Path(f).name) == pathology]
    if not files:
        have = sorted({get_pathology(Path(f).name) for f in fold.test})
        raise ValueError(f"Fold {fold_idx} has no {pathology} test recordings (it has: {have})")
    if not 0 <= n < len(files):
        raise ValueError(f"n={n} is out of range: fold {fold_idx} has {len(files)} {pathology} test recordings")
    return files[n]


def locate(cfg, signal, noise_type, noise_level, fold_idx=None):
    """Resolve the target, noisy and predicted audio of one TEST recording.

    run:        folder name under cfg.directory.runs (e.g. "STFT_test")
    signal:     path of the recording relative to the raw target folder (see pick_signal)
    fold_idx:   None -> the fold whose test set holds `signal` (k-fold: exactly one)
    kind / sisdr / sisdr_noisy come from results.csv (None if the run has no csv)."""
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No run folder {run_dir}")

    signal = Path(signal).as_posix()
    folds = SP.load_folds(cfg.folds_json)
    if fold_idx is None:
        owners = [f.index for f in folds if signal in f.test]
        if not owners:
            raise ValueError(f"{signal} is not in the test set of any fold of {cfg.experiment.experiment_name}")
        fold_idx = owners[0]
    else:
        fold = next((f for f in folds if f.index == fold_idx), None)
        if fold is None:
            raise ValueError(f"{cfg.experiment.experiment_name} has no fold {fold_idx} (folds: {[f.index for f in folds]})")
        if signal not in fold.test:
            raise ValueError(f"{signal} is not in the test set of fold {fold_idx}")

    d = FoldDirs(run_dir / f"fold_{fold_idx}")
    paths = {"target": Path(cfg.directory.pcg_raw) / signal,
             "noisy": d.test_noisy(noise_type, noise_level) / signal,
             "prediction": d.test_prediction(noise_type, noise_level) / signal}
    for label, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{label} audio not found: {p} "
                                    f"(was this condition evaluated, and the files not released?)")

    kind = sisdr = sisdr_noisy = None
    csv = cfg.csv_results
    if csv.exists():
        df = pd.read_csv(csv)
        rows = df[(df["fold"] == fold_idx) & (df["noise_type"] == noise_type)
                  & np.isclose(df["dB"], noise_level) & (df["file"] == signal)]
        if rows.empty:
            raise ValueError(f"results.csv has no row for {signal}, fold {fold_idx}, {noise_type} {noise_level} dB")
        row = rows.iloc[0]
        kind, sisdr, sisdr_noisy = str(row["kind"]), float(row["sisdr"]), float(row["sisdr_noisy"])

    return Case(run_dir, fold_idx, signal, get_pathology(Path(signal).name), noise_type, noise_level,
                paths["target"], paths["noisy"], paths["prediction"], kind, sisdr, sisdr_noisy)


def load_signal(path, sr):
    x, _ = load(path, sr=sr)
    return x


# ---- time-frequency images ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TFImage:
    values: np.ndarray              # (freq, time): magnitude in dB, or linear magnitude (see scale)
    t: np.ndarray                   # seconds, one value per column
    f: np.ndarray                   # Hz, ascending, one value per row
    ylabel: str
    scale: str = "db"               # "db" or "linear"
    label: str = "Magnitude [dB]"   # colorbar label


def _image(S, duration, f, scale="db"):
    mag = np.abs(np.asarray(S))
    if scale == "db":
        values, label = 20 * np.log10(mag + EPS), "Magnitude [dB]"
    elif scale == "linear":
        values, label = mag, "Magnitude"
    else:
        raise ValueError(f"scale must be 'db' or 'linear', got {scale!r}")
    return TFImage(values, np.linspace(0, duration, mag.shape[1]), np.asarray(f), "Frequency [Hz]", scale, label)


def tf_images(x, sr, cfg, scale="db"):
    """STFT, WSST, CWT and Stockwell magnitudes of one signal (scale="db" or "linear"), as
    {name: TFImage}, using the presets in cfg.transform (they only need to look good, not match
    the transform the run used).

    - STFT, S: linear frequency axis, 0 to sr/2 in Hz.
    - WSST, CWT: the frequencies come from the transform (cycles/sample * sr = Hz). They are
      descending and log-spaced, so the rows are flipped to ascending. For the CWT they are the
      frequencies of the synchrosqueezed output, a close approximation of the rows' frequencies,
      which is fine for a picture.
    - S: the signal is cut into consecutive windows of s_time_window samples, each transformed and
      joined along time; the last samples that do not fill a window are left out."""
    tf, duration = cfg.transform, len(x) / sr
    images = {}

    s = tf.stft
    S = TF.make_stft(sr, s.nperseg, s.noverlap, s.fft_size)(x)
    images["STFT"] = _image(S, duration, np.linspace(0, sr / 2, S.shape[0]), scale)

    for name, transform, w in (("WSST", TF.WSST, tf.wsst), ("CWT", TF.CWT, tf.cwt)):
        S, f = transform(x, w.wavelet, w.nv, w.n, w.padtype, w.scaletype, w.preset, w.downsample)
        f = np.asarray(f) * sr
        if f[0] > f[-1]:
            S, f = np.asarray(S)[::-1], f[::-1]
        images[name] = _image(S, duration, f, scale)

    window = tf.stockwell.s_time_window
    n = len(x) // window
    if n < 1:
        raise ValueError(f"Signal has {len(x)} samples, shorter than the Stockwell window ({window})")
    transform = TF.make_stockwell(sr)
    S = transform(x)
    images["S"] = _image(S, duration, np.linspace(0, sr / 2, S.shape[0]), scale)
    # S = np.hstack([transform(x[i * window:(i + 1) * window]) for i in range(n)])
    # images["S"] = _image(S, n * window / sr, np.linspace(0, sr / 2, S.shape[0]), scale)

    return images


# ---- plots -------------------------------------------------------------------------------------------------------

def color_limits(images, range_db=80):
    """Color limits {name: (vmin, vmax)} from the images of the TARGET signal. Pass the same limits
    for the noisy and restored signals so the three figures use one scale.
    dB images show the top `range_db` dB; linear images go from 0 to the maximum."""
    return {name: (im.values.max() - range_db if im.scale == "db" else 0, im.values.max())
            for name, im in images.items()}


def plot_signal(x, sr, images, limits, title, pathology, ylim=None, color=COLORS["target"],
                label="Target", target=None, target_front=False):
    """The signal in time on top, its four time-frequency images below (2x2), the colorbar on the right.
    pathology: full name (case.pathology_name), used in the panel titles.
    label: legend text of the signal. target: the clean signal, drawn over the signal
    (target_front=True) or under it (False)."""
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04])

    ax = fig.add_subplot(gs[0, :])
    t = np.arange(len(x)) / sr
    ax.plot(t, x, color=color, linewidth=1.0, zorder=2, label=label)
    if target is not None:
        ax.plot(t, target, color=COLORS["target"], linewidth=1.0, linestyle='--', zorder=3 if target_front else 1, label="Target")
    ax.set_xlim(0, len(x) / sr)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("Amplitude")
    ax.grid()
    ax.legend(loc="upper right")

    axes = [fig.add_subplot(gs[r, c]) for r, c in ((1, 0), (1, 1), (2, 0), (2, 1))]
    for i, (ax, (name, im)) in enumerate(zip(axes, images.items())):
        shown = ax.pcolormesh(im.t, im.f, im.values, cmap="hot", shading="auto", rasterized=True, vmin=limits[name][0], vmax=limits[name][1])
        ax.set_title(PANEL_TITLES[name].format(pathology))
        ax.set_ylabel(im.ylabel)
        if i >= 2:
            ax.set_xlabel("Time [s]")
    cax = fig.add_subplot(gs[1:, 2])
    fig.colorbar(shown, cax=cax, label=next(iter(images.values())).label)
    fig.suptitle(title)
    return fig


def average_psd(files, sr, nperseg=256):
    """Mean power spectral density of several recordings: (frequencies [Hz], power [dB]).
    The power is averaged first and converted to dB after."""
    powers = []
    for f in files:
        x = load_signal(f, sr)
        if len(x) < nperseg:
            raise ValueError(f"{Path(f).name} has {len(x)} samples, fewer than nperseg={nperseg}")
        freqs, p = welch(x, fs=sr, nperseg=nperseg)
        powers.append(p)
    return freqs, 10 * np.log10(np.mean(powers, axis=0) + 1e-12)


def plot_psd(cfg, noise_level, noise_type=None, fold_idx=None, nperseg=256):
    """Average PSD of the target, noisy and predicted recordings, one panel per noise kind.

    Every evaluated recording at `noise_level` counts, whatever way the run evaluated it (each noise
    type on its own, or "Sum"): a panel holds the recordings that received that kind of noise.
    noise_type: None -> all evaluation conditions; "Sum", "AWGN", ... -> only that one.
    fold_idx:   None -> all folds."""
    run_dir = Path(cfg.run_dir)
    df = pd.read_csv(cfg.csv_results)
    df = df[np.isclose(df["dB"], noise_level)]
    if noise_type is not None:
        df = df[df["noise_type"] == noise_type]
    if fold_idx is not None:
        df = df[df["fold"] == fold_idx]
    sr = cfg.audio.sr
    line_colors = {"Target": COLORS["target"], "Noisy": COLORS["noisy"], "Prediction": COLORS["prediction"]}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for ax, kind in zip(axes.flat, KINDS):
        title = f"Average PSD at {noise_level:g} dB of {kind} noise"
        items = list(df[df["kind"] == kind].itertuples())
        if not items:
            ax.set_title(f"{title} (no recordings)")
            continue
        dirs = {r.fold: FoldDirs(run_dir / f"fold_{r.fold}") for r in items}      # the files sit in the fold's folder
        sets = {"Target": [Path(cfg.directory.pcg_raw) / r.file for r in items],
                "Noisy": [dirs[r.fold].test_noisy(r.noise_type, r.dB) / r.file for r in items],
                "Prediction": [dirs[r.fold].test_prediction(r.noise_type, r.dB) / r.file for r in items]}
        for label, files in sets.items():
            freqs, db = average_psd(files, sr, nperseg)
            ax.plot(freqs, db, color=line_colors[label], label=label)
        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_title(title)
        ax.grid()
        ax.legend()

    for ax in axes[-1]:
        ax.set_xlabel("Frequency [Hz]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Power [dB]")
    return fig


def save_svg(fig, name, folder=None):
    """Save the figure as images/<name>.svg at the project root (or in `folder`)."""
    folder = Path(folder) if folder else find_root(Path.cwd()) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path