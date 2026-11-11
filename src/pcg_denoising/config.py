import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class Experiment:
    experiment_name: str

@dataclass(frozen=True)
class Seed:
    folds_seed: int
    noise_segmentation_seed: int
    network_initialization_seed: int
    noise_training_seed: int
    noise_test_seed: int

@dataclass(frozen=True)
class Noise:
    training_noise: int
    training_noise_type: str
    evaluation_noise: list
    evaluation_noise_type: list

@dataclass(frozen=True)
class Audio:
    sr: int

@dataclass(frozen=True)
class Stft:
    nperseg: int
    noverlap: int
    fft_size: int

@dataclass(frozen=True)
class Wavelet:
    wavelet: str
    n: str
    nv: int
    padtype: str
    scaletype: str
    preset: str
    downsample: int
    cwt_freq_window: int
    cwt_time_window: int
    overlap: float

@dataclass(frozen=True)
class S:
    s_time_window: int

@dataclass(frozen=True)
class Transform:
    name: str
    stft: Stft
    wsst: Wavelet
    cwt: Wavelet
    s: S

    @property
    def active(self):
        return getattr(self, self.name.lower())

@dataclass(frozen=True)
class Training:
    folds: int
    learning_rate: float
    batch_size: int
    epochs: int
    dataset_split: float
    training_split: float
    keep_datasets: bool = False

Stage = Literal["soundwave", "train", "test"]
Source = Literal["physionet", "speech"]
Kind = Literal["target", "noisy", "prediction"]
Noise_file = Literal["PhysioNet_train.wav", "Speech_train.wav", "PhysioNet_test.wav", "Speech_test.wav"]

@dataclass(frozen=True)
class Directory:
    pcg_raw: Path
    physionet_raw: Path
    speech_raw: Path

    noise_processed: Path

    def noise_dir(self, noise: Noise_file) -> Path:
        return self.noise_processed / noise

    runs: Path


@dataclass(frozen=True)
class FoldDirs:
    root: Path

    @property
    def train_noisy(self):
        return self.root / "train_noisy"
    @property
    def val_noisy(self):
        return self.root / "val_noisy"
    @property
    def test(self):
        return self.root / "test"
    @property
    def train_h5(self):
        return self.root / "train.h5"
    @property
    def val_h5(self):
        return self.root / "val.h5"
    @property
    def model(self):
        return self.root / "model.pt"
    @property
    def model_history(self):
        return self.root / "model_history.json"

    def test_noisy(self, noise_type, dB):
        return self.test / f"{noise_type}_{dB}dB" / "noisy"
    def test_prediction(self, noise_type, dB):
        return self.test / f"{noise_type}_{dB}dB" / "prediction"

@dataclass(frozen=True)
class Config:
    experiment: Experiment
    seed: Seed
    noise: Noise
    audio: Audio
    transform: Transform
    training: Training
    directory: Directory

    @property
    def run_dir(self) -> Path:
        return self.directory.runs / f"{self.transform.name}_{self.experiment.experiment_name}"
    @property
    def folds_json(self) -> Path:
        return self.run_dir / "folds.json"
    @property
    def csv_results(self) -> Path:
        return self.run_dir / "results.csv"
    @property
    def xlsx_results(self) -> Path:
        return self.run_dir / "report.xlsx"

    def fold_dirs(self, fold_index: int) -> FoldDirs:
        return FoldDirs(self.run_dir / f"fold_{fold_index}")

def find_root(start: Path) -> Path:
    """Closest parent folder that contains pyproject.toml."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start

def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result

def read_toml(path):
       with open(path, "rb") as f:
           return tomllib.load(f)

def load_config(path="config/config.toml", override_path=None) -> Config:
    path = Path(path).resolve()
    raw = read_toml(path)

    if override_path is not None:
        override_path = Path(override_path).resolve()
        raw = merge_config(raw, read_toml(override_path))
    
    root = find_root(path.parent)
    t = raw["transform"]
    transform = Transform(t["name"], Stft(**t["stft"]), Wavelet(**t["wsst"]), Wavelet(**t["cwt"]), S(**t["s"]))
    directory = Directory(**{k: root / v for k, v in raw["directory"].items()})
    return Config(
        experiment=Experiment(**raw["experiment"]),
        seed=Seed(**raw["seed"]), 
        noise=Noise(**raw["noise"]), 
        audio=Audio(**raw["audio"]), 
        transform=transform, 
        training=Training(**raw["training"]), 
        directory=directory
        )