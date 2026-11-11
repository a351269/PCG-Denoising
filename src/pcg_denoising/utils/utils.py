import re
import shutil
from pathlib import Path


PATHOLOGY = re.compile(r"(?<![A-Za-z0-9])(AS|MR|MS|MVP|N)(?![A-Za-z0-9])")


def get_pathology(name):
    m = PATHOLOGY.search(name)
    return m.group(1) if m else None


def reset_dir(path, allowed_root=None) -> Path:
    path = Path(path)

    if allowed_root is not None:
        allowed_root = Path(allowed_root).resolve()
        if path == allowed_root or not path.is_relative_to(allowed_root):
            raise ValueError(f"Refusing to delete {path}: not inside {allowed_root}")
    
    if path.exists():
        shutil.rmtree(path)
        print(f"Files in {path} deleted")
    path.mkdir(parents=True, exist_ok=True)

    return path


def transform_args(cfg):
    t, sr, name = cfg.transform.active, cfg.audio.sr, cfg.transform.name
    if name == "STFT":
        return sr, t.nperseg, t.noverlap, t.fft_size
    if name in ("WSST", "CWT"):
        return sr, t.wavelet, t.nv, t.cwt_freq_window, t.cwt_time_window, t.n, t.padtype, t.scaletype, t.preset, t.downsample
    return sr, t.s_time_window