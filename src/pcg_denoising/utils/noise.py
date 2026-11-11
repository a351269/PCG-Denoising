import json
import random
import numpy as np
import soundfile as sf
import colorednoise as cn

from tqdm import tqdm
from pathlib import Path

from librosa.core import load
from collections import Counter, defaultdict

from pcg_denoising.splits import list_target_files
from pcg_denoising.utils.utils import get_pathology


"""
Noise generation/addition functions

"""
def addWGN(signalIN, SNR=30,seed=[]):
    """
    Esta funcion agrega ruido blanco aditivo Gaussiano 
    de manera controlada a una señal de entrada. Los parámetros
    de entrada son:
        
    signalOUT = addWGN(signal, SNR,seed)
    
    signal: señal de entrada
    SNR: relación señal a ruido en dB (opcional, default 30 dB)
    seed: semilla aleatoria (opcional, por default no se inicializa)
    """    
        
    if(bool(seed)):
        np.random.seed(seed)
    
    # calcula la potencia de la señal
    signal_Power_dB = 10 * np.log10(np.mean(signalIN ** 2))
    # calcula la potencia del ruido en función del SNR deseado
    noise_Power_dB = signal_Power_dB - SNR
    noise_Power = 10 ** (noise_Power_dB / 10)
    
    # genera el AWGN con media cero
    noise = np.random.normal(0, np.sqrt(noise_Power), len(signalIN))
    noise.resize(signalIN.shape)
    signalOUT = signalIN + noise
    
    return signalOUT


def addPGN(signalIN, SNR=30,seed=[]):
    """
    Esta funcion agrega ruido rosado aditivo Gaussiano 
    de manera controlada a una señal de entrada. Los parámetros
    de entrada son:
        
    signalOUT = addPGN(signal, SNR,seed)
    
    signal: señal de entrada
    SNR: relación señal a ruido en dB (opcional, default 30 dB)
    seed: semilla aleatoria (opcional, por default no se inicializa)
    """
    
    beta = 1  # generador de la forma 1/f^beta, beta=1 corresponde a rosado
        
    if(bool(seed)):
        np.random.seed(seed)
    
    # calcula la potencia de la señal
    signal_Power_dB = 10 * np.log10(np.mean(signalIN ** 2))
    # calcula la potencia del ruido en función del SNR deseado
    noise_Power_dB = signal_Power_dB - SNR
    noise_Power = 10 ** (noise_Power_dB / 10)
    
    # genera el APGN con media cero
    noise = cn.powerlaw_psd_gaussian(beta, len(signalIN))
    # calcula la potencia original del ruido rosa
    mynoise_Power_dB = 10 * np.log10(np.mean(noise ** 2))
    # pone el ruido rosa a cero dB
    noise = noise/np.sqrt((10**(mynoise_Power_dB/10)))
    # agrega el APGN con el SNR deseado
    noise = np.sqrt(noise_Power)*noise
    noise.resize(signalIN.shape)
    signalOUT = signalIN + noise
    
    return signalOUT


def addSignal2Signal(signalIN1, signalIN2, SSR=30):
    """
    Esta función agrega una de entrada S1 a otra señal de entrada S2
    de manera controlada a un SSR (Signal to Signal Ratio). Los parámetros
    de entrada son:
        
    signalOUT = addSignal2Signal(signalIN1, signalIN2, Fs, SNR)
    
    signalIN1: señal de entrada principal
    signalIN2: señal de entrada que va a ser montada sobre signalIN1
    SSR: relación señalIN1 a señalIN2 en dB (opcional, default 30 dB)
    """
    pad = np.size(signalIN1)-np.size(signalIN2)
    if(pad > 0):
        signalIN2 = np.pad(signalIN2,(0,pad),'constant')
        print("Warning: len(SignalIN1) > len(SignalIN2)")
    elif (pad < 0):
        signalIN2 = signalIN2[0:np.size(signalIN1)]
        print("Warning: len(SignalIN1) < len(SignalIN2)")
    
    # calcula la potencia de la señal principal
    signalIN1_Power_dB = 10 * np.log10(np.mean(signalIN1 ** 2))
    # calcula la potencia de la segunda señal
    signalIN2_Power_dB = 10 * np.log10(np.mean(signalIN2 ** 2))
    # pone la señal2 a cero dB
    signalIN2 = signalIN2/np.sqrt((10**(signalIN2_Power_dB/10)))
    # calcula la potencia de la 2a señal en función del SSR deseado
    signal_Power_dB = signalIN1_Power_dB - SSR
    signal_Power = 10 ** (signal_Power_dB / 10)
    # ajusta la señal de salida de acuerdo al SSR deseado
    signalOUT = signalIN1 + np.sqrt(signal_Power)*signalIN2
        
    return signalOUT


"""
Noise metrics
"""
def SISDR(x, xh):
    """
    Esta función calcula la Scale Invariant Signal to Distorsion Ration (SISDR)
    de acuerdo al artículo de Le Roux, Jonathan, et al. "SDR–half-baked or well 
    done?." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech
    and Signal Processing (ICASSP). IEEE, 2019.
    
    El uso de la función es el siguiente
        
    SISDR_value = SISDR(source_signal, hat_signal)
    
    source_signal: señal original vector de dimensión Nx1 que se utiliza como.
                   referencia Es la señal limpia sin ninguna distorsión.
    hat_signal: señal reconstruida de dimensión Nx1 (obtenida a través de algún 
                método de síntesis o de separación de fuentes) la cual se desea 
                estimar su relación señal a distorsión en escala de dBs.
    SISDR_value: valor de la relación señal a distorsión invariante a escala 
                 (SISDR) entre source_signal y hat_signal


   author: MAA
   Sat Oct 30 15:40:40 2021
   """    
    SISDR_value = -1000.0
    
    x_size = x.shape
    xh_size = xh.shape
    
    if(len(xh.shape) != len(x_size)):
        return SISDR_value
    elif(x_size[0] < xh_size[0]):
        xh = xh[0:x_size[0]]
        print("Warning: hat_signal adjusted to source_signal length")
    elif(x_size[0] > xh_size[0]):
        x = x[0:xh_size[0]]
        print("source_signal adjusted to hat_signal length")
        
    alfa = np.dot(x, xh) / np.sum(x**2)
    
    SISDR_value = 10*np.log10(np.sum((alfa*x)**2) / sum((alfa*x - xh)**2))
    
    return SISDR_value


def balanced_split(items, weight, test_fraction, rng):
    """Random order; each item goes to the side that ends up closest to its target share
    of the total weight (counting the item itself). Returns (train, test), in assignment order."""
    items = list(items)
    rng.shuffle(items)
 
    train, test = [], []
    w_train = w_test = 0.0
    for item in items:
        w = weight(item)
        total = w_train + w_test + w
        deficit_train = (1 - test_fraction) * total - w_train
        deficit_test = test_fraction * total - w_test
        if deficit_train >= deficit_test:
            train.append(item)
            w_train += w
        else:
            test.append(item)
            w_test += w
    return train, test
 
 
def write_concatenated(path, files, sr):
    """All files, resampled to sr, one after the other in a single float32 mono wav."""
    with sf.SoundFile(path, "w", samplerate=sr, channels=1, subtype="FLOAT") as out:
        for f in files:
            signal, _ = load(f, sr=sr)
            out.write(signal)

def split_noise(raw_physionet_path, raw_speech_path, noise_save_path, sr, eval_percentage, seed=0):
    sources = {"PhysioNet": raw_physionet_path, "Speech": raw_speech_path}
    out_dir = Path(noise_save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, root in sources.items():
        root = Path(root)
        files = list_target_files(root)
        if len(files) < 2:
            raise ValueError(f"{name}: need at least 2 recordings in {root} to split, found {len(files)}")

        seconds = {f: sf.info(root / f).duration for f in files}

        rng = random.Random(f"{seed}-{name}")
        train, test = balanced_split(files, seconds.__getitem__, eval_percentage, rng)
        if not train or not test:
            raise ValueError(f"{name}: eval_percentage={eval_percentage} leaves the "
                             f"{'train' if not train else 'test'} noise empty with {len(files)} recordings")
 
        write_concatenated(out_dir / f"{name}_train.wav", [root / f for f in train], sr)
        write_concatenated(out_dir / f"{name}_test.wav", [root / f for f in test], sr)
 
        total = sum(seconds.values())
        test_seconds = sum(seconds[f] for f in test)
        manifest[name] = dict(train=train, test=test,
                              train_seconds=round(total - test_seconds, 2),
                              test_seconds=round(test_seconds, 2),
                              test_share=round(test_seconds / total, 4))
        print(f"{name}: {len(train)} recordings for training, {len(test)} for evaluation "
              f"({100 * test_seconds / total:.1f}% of {total:.0f} s, target {100 * eval_percentage:.0f}%)")
 
    result = {"settings": dict(eval_percentage=eval_percentage, sr=sr, seed=seed), "sources": manifest}
    (out_dir / "noise_split.json").write_text(json.dumps(result, indent=2))
    return result



def random_segment(noise, size, rng):
    """A random stretch of `size` samples; the noise is repeated if it is shorter than that."""
    if len(noise) < size:
        noise = np.tile(noise, int(np.ceil(size / len(noise))))
    start = rng.integers(0, len(noise) - size + 1)
    return noise[start:start + size]

def balanced_kinds(n, types, rng):
    """n noise types, each used n // len(types) times (the remainder is spread randomly)."""
    order = list(types)
    # Which noise types get an extra contamination on uneven cases
    rng.shuffle(order)
    kinds = [order[i % len(order)] for i in range(n)]
    # Random order across files
    rng.shuffle(kinds)
    return kinds

def assign_kinds(files, types, rng, group_of=None):
    """{file: noise type}, balanced over all files, or within each group if group_of is given."""
    groups = defaultdict(list)
    for p in files:
        groups[group_of(p) if group_of else None].append(p)

    assignment = {}
    # Sorted for seed reproducibility
    for key in sorted(groups, key=str):
        group = groups[key]
        for p, kind in zip(group, balanced_kinds(len(group), types, rng)):
            assignment[p] = kind
    return assignment

def print_noise_summary(files, assignment, types, pathology_of):
    """Table of audios found per pathology and how many got each noise type."""
    if not files:
        print("No audio files found")
        return

    counts = defaultdict(Counter)
    for p in files:
        counts[str(pathology_of(p))][assignment[p]] += 1

    width = max([len("Pathology"), *map(len, counts)])
    header = f"{'Pathology':<{width}}  {'Audios':>6}  " + "  ".join(f"{t:>9}" for t in types)
    rule = "-" * len(header)

    print(header)
    print(rule)
    for key in sorted(counts):
        row = counts[key]
        print(f"{key:<{width}}  {sum(row.values()):>6}  " + "  ".join(f"{row[t]:>9}" for t in types))
    print(rule)
    total = Counter(assignment.values())
    print(f"{'Total':<{width}}  {len(files):>6}  " + "  ".join(f"{total[t]:>9}" for t in types))

def add_noise(files, target_root, noise_paths, noisy_save_path, dB, sr, noise_type,
              seed=0, balance_by_pathology=True, summary=True):
    """Add noise to every recording in `files`.
 
    files:        recordings as paths relative to target_root (e.g. fold.train + fold.val)
    noise_paths:  {"PhysioNet": ".../physionet_train.wav", "Speech": ".../speech_train.wav"}
                  (the train or the eval noise files from split_noise); only the ones used are needed
    noise_type:   "AWGN", "APGN", "PhysioNet", "Speech", or "Sum" (an even mix of all four)
 
    The noisy audio is written to noisy_save_path with the same relative paths and names as the
    targets. Returns (noisy_files, kinds): the noisy paths and the noise type each recording got,
    in the same order as `files`, so target / noisy / prediction can be paired with zip()."""

    noise_types = ("AWGN", "APGN", "PhysioNet", "Speech")

    if noise_type != "Sum" and noise_type not in noise_types:
        raise ValueError(f"Unknown noise type {noise_type!r}. Use 'Sum' or one of {noise_types}")
 
    rng = np.random.default_rng(seed)
    types = noise_types if noise_type == "Sum" else (noise_type,)
 
    recordings = {}
    for kind in types:
        if kind in ("PhysioNet", "Speech"):
            if kind not in noise_paths:
                raise ValueError(f"noise_paths has no entry for {kind!r}")
            recordings[kind], _ = load(noise_paths[kind], sr=sr)
 
    def make_noisy(signal, kind):
        if kind == "AWGN":
            return np.float32(addWGN(signal, SNR=dB, seed=int(rng.integers(2**32))))   # a new seed per file
        if kind == "APGN":
            return np.float32(addPGN(signal, SNR=dB, seed=int(rng.integers(2**32))))
        segment = random_segment(recordings[kind], len(signal), rng)
        return addSignal2Signal(signal, segment, SSR=dB)
 
    pathology_of = lambda f: get_pathology(Path(f).name)
    assignment = assign_kinds(files, types, rng, pathology_of if balance_by_pathology else None)
    if summary:
        print_noise_summary(files, assignment, types, pathology_of)
 
    out_root = Path(noisy_save_path)
    noisy_files = []
    for rel in tqdm(files, desc="Adding noise", unit=" Audio"):
        signal, _ = load(Path(target_root) / rel, sr=sr)
        out = out_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, make_noisy(signal, assignment[rel]), sr)
        noisy_files.append(out)
 
    return noisy_files, [assignment[rel] for rel in files]

