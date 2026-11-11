import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from librosa import load


DATASET_NAMES = {"target": "mag_target", "noisy": "mag_noisy"}


def get_compression():
    """Lossless compression settings for HDF5 datasets."""
    try:
        import hdf5plugin
        return hdf5plugin.Blosc(cname="zstd", clevel=3, shuffle=hdf5plugin.Blosc.SHUFFLE)
    except ImportError:
        return dict(compression="lzf", shuffle=True)


def make_dataset(f, name, n_freq, frames, compression):
    return f.create_dataset(name, shape=(0, n_freq, frames), maxshape=(None, n_freq, frames), 
                            dtype="float32", chunks=(16, n_freq, frames), **compression)


def window_starts(t, time_window, overlap=0):
    """Creates a list of sample starting points across the time axis on the time-frequency representation"""
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in [0, 1)")
    
    step = time_window - int(np.ceil(time_window * overlap))
    starts = list(range(0, t - time_window + 1, step))
    
    if starts[-1] != t - time_window:
        starts.append(t - time_window)
    
    return starts


def append_windows(dset, mags):
    n = dset.shape[0]
    dset.resize(n + len(mags), axis=0)
    dset[n:] = mags
    return len(mags)


def add_file_tf(dset, file_path, sr, transform, freq_window, time_window, overlap=0, per_window=False):
    """
    Load one audio file, apply `transform` (audio -> complex scalogram of shape
    (freq, time)), cut it into (freq_window, time_window) magnitude windows and
    append them to the HDF5 dataset.
    
    Frequency rows beyond the transform's output are zero padded. Returns how many were added.
    """
    xs, _ = load(file_path, sr=sr)

    if per_window:
        length = len(xs)
    else:
        tf = transform(xs)
        length = tf.shape[1]
        n_f = tf.shape[0]
        if n_f > freq_window:
            raise ValueError(f"Transform has {n_f} frequency rows but freq_window is {freq_window}")

    if length < time_window:
        print(f"{Path(file_path).name} is shorter than the expected window, skipped")
        return 0
    
    starts = window_starts(length, time_window, overlap)
    mags = np.zeros((len(starts), freq_window, time_window), dtype=np.float32)    
    for i, s in enumerate(starts):
        if per_window:
            mags[i] = np.abs(transform(xs[s:s + time_window]))
        else:
            mags[i, :n_f] = np.abs(tf[:, s:s + time_window])
    return append_windows(dset, mags)


def build_tf_dataset(files, name, save_path, sr, transform, freq_window, time_window, overlap=0, per_window=False):
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_file, "a") as f:
        if name in f:
            del f[name]
        dset = make_dataset(f, name, freq_window, time_window, get_compression())
        counts = [add_file_tf(dset, p, sr, transform, freq_window, time_window, overlap, per_window) for p in tqdm(files, desc=name, unit=" Audio")]

    print(f"{name}: {sum(counts)} windows from {len(files)} files -> {save_file}")
    return counts


def build_pair(save_path, target_files, noisy_files, **tf_args):
    """tf_args go to build_tf_dataset: sr, transform, freq_window, time_window, overlap, per_window."""
    if len(target_files) != len(noisy_files):
        raise ValueError(f"{len(target_files)} target files but {len(noisy_files)} noisy files")

    Path(save_path).unlink(missing_ok=True)
    counts = {split: build_tf_dataset(files, DATASET_NAMES[split], save_path, **tf_args) for split, files in (("target", target_files), ("noisy", noisy_files))}

    for t, n, tc, nc in zip(target_files, noisy_files, counts["target"], counts["noisy"]):
        if tc != nc:
            raise ValueError(f"Window count differs for {Path(t).name} ({tc}) and {Path(n).name} ({nc})")
    return sum(counts["target"])