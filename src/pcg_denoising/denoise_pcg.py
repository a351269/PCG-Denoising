import time
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from librosa.core import load
from scipy.signal import stft, istft
from scipy.io.wavfile import write

from ssqueezepy import issq_cwt, Wavelet
from ssqueezepy import icwt, Wavelet
from ssqueezepy.utils import cwt_scalebounds, make_scales

import pcg_denoising.network_training as NT
import pcg_denoising.utils.time_frequency as TF
from pcg_denoising.utils.utils import reset_dir, transform_args


"""
TF Transforms
"""
def get_STFT(y, *args):
    sr, nperseg, noverlap, fft_size = args
    f, t, spec = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, nfft=fft_size)
    
    if len(t) >= 64:
        S = int(np.floor(len(t)/64))
        spec2 = np.zeros((S+1, 129, 64), dtype=complex)
        
        for j in range(S):
            spec2[j] = spec[:,(j*64):((j+1)*64)]
            
        time_axis = len(spec[0])

        spec2[S] = spec[:,(time_axis-64):time_axis]
        
        
    if len(t) < 64:
        print("Hay audios mas cortos de la duracion esperada")
    
    mag = np.zeros((S+1, 129, 64))
    phase = np.zeros((S+1, 129, 64), dtype=complex)
    for i in range(S+1):
        mag[i] = np.abs(spec2[i])
        phase[i] = np.exp(1.j*np.angle(spec2[i]))
        
    return mag, phase, S, len(t)

def get_iSTFT(mag, phase, Audio_num, Audio_length, sr, NPERSEG, H, fft):
    stft_inv = np.zeros((129, Audio_length), dtype=complex)
    
    for i in range(Audio_num):
        mult = mag[i]*phase[i]
        start = i * mult.shape[1]
        end = (i + 1) * mult.shape[1]
        stft_inv[:, start:end] = mult
            
    start = Audio_length - mult.shape[1]
    end = Audio_length
    
    
    stft_inv[:, start:end] = mag[Audio_num]*phase[Audio_num]
    
    t1, y = istft(stft_inv, fs=sr, nperseg=NPERSEG, noverlap=H, nfft=fft)
    
    return y

def get_WSST(xs, *args):
    sr, wavelet, NV, freq_window, time_window, N, padtype, scaletype, preset, downsample = args
    ssq, _ = TF.WSST(xs, wavelet, NV, N, padtype, scaletype, preset, downsample)
    t = len(ssq[0,:])
    
    if t >= time_window:
        S = int(np.floor(t/time_window))
        ssq_array = np.zeros((S+1, freq_window, time_window), dtype=complex)
        
        for j in range(S):
            ssq_array[j, 0:ssq.shape[0]] = ssq[:,(j*time_window):((j+1)*time_window)]
        
        time_axis = len(ssq[0])
        
        ssq_array[S, 0:ssq.shape[0]] = ssq[:,(time_axis-time_window):time_axis]
        
    if t < time_window:
        print("Hay audios mas cortos de la duracion esperada")
    
    mag_array = np.zeros((S+1, freq_window, time_window))
    phase_array = np.zeros((S+1, freq_window, time_window), dtype=complex)
    for i in range(S+1):
        mag_array[i] = np.abs(ssq_array[i])
        phase_array[i] = np.exp(1.j*np.angle(ssq_array[i]))
    
    return mag_array, phase_array, S

def get_iWSST(mag, phase, Audio_num, Audio_length, wavelet, freq_window, N):

    ssq_inv = np.zeros((freq_window, Audio_length), dtype=complex)
    
    for i in range(Audio_num):  
        mult = mag[i]*phase[i]
        start = i * mult.shape[1]
        end = (i + 1) * mult.shape[1]
        ssq_inv[:, start:end] = mult
            
    start = Audio_length - mult.shape[1]
    end = Audio_length
    
    ssq_inv[:, start:end] = mag[Audio_num]*phase[Audio_num]
    
    wavelet = Wavelet(wavelet, N=N, dtype='float64')
    y = issq_cwt(ssq_inv, wavelet)
    
    return y


def get_CWT(xs, *args):
    sr, wavelet, NV, freq_window, time_window, N, padtype, scaletype, preset, downsample = args
    CWT, _ = TF.CWT(xs, wavelet, NV, N, padtype, scaletype, preset, downsample)
    freq_size = len(CWT[:, 0])
    t = len(CWT[0,:])
    
    if t >= time_window:
        S = int(np.floor(t/time_window))
        cwt_array = np.zeros((S+1, freq_window, time_window), dtype=complex)
        
        for j in range(S):
            cwt_array[j, 0:CWT.shape[0]] = CWT[:,(j*time_window):((j+1)*time_window)]
        
        time_axis = len(CWT[0])
        
        cwt_array[S, 0:CWT.shape[0]] = CWT[:,(time_axis-time_window):time_axis]
        
    if t < time_window:
        print("Hay audios mas cortos de la duracion esperada")
    
    mag_array = np.zeros((S+1, freq_window, time_window))
    phase_array = np.zeros((S+1, freq_window, time_window), dtype=complex)
    for i in range(S+1):
        mag_array[i] = np.abs(cwt_array[i])
        phase_array[i] = np.exp(1.j*np.angle(cwt_array[i]))
    
    return mag_array, phase_array, S, freq_size

def get_iCWT(mag, phase, Audio_num, Audio_length, freq_size, wavelet, NV, N, padtype, scaletype, preset, downsample):
    cwt_inv = np.zeros((freq_size, Audio_length), dtype=complex)
    
    for i in range(Audio_num):  
        mult = mag[i]*phase[i]
        start = i * mult.shape[1]
        end = (i + 1) * mult.shape[1]
        cwt_inv[0:freq_size, start:end] = mult[0:freq_size]
            
    start = Audio_length - mult.shape[1]
    end = Audio_length
    
    cwt_inv[0:freq_size, start:end] = mag[Audio_num, 0:freq_size, :]*phase[Audio_num, 0:freq_size, :]
    
    wavelet = Wavelet(wavelet, N=N, dtype='float64')
    min_scale, max_scale = cwt_scalebounds(wavelet, N=N,  preset=preset)
    scales = make_scales(N, min_scale, max_scale, nv=NV, scaletype=scaletype, wavelet=wavelet, downsample=downsample)
    y = icwt(cwt_inv, wavelet=wavelet, nv=NV, scales=scales, padtype=padtype)
    
    return y


def get_S(x, *args):
    sr, timeWindow = args
    t = len(x)
    tWindow = int(timeWindow)
    fWindow = int(timeWindow/2 + 1)
    
    if t >= tWindow:
        S = int(np.floor(t/tWindow))
        xSTArray = np.zeros((S+1, fWindow, tWindow), dtype=complex)
        
        for j in range(S):
            xWindow = x[(j*tWindow):((j+1)*tWindow)]
            xST, tArray, fArray = TF.S(xWindow, sr)
            xSTArray[j] = xST
        
        xWindow = x[(t-tWindow):t]
        xST, tArray, fArray = TF.S(xWindow, sr)
        xSTArray[S] = xST
        
    if t < tWindow:
        print("Hay audios mas cortos de la duracion esperada")
    
    magArray = np.zeros((S+1, fWindow, tWindow))
    phaseArray = np.zeros((S+1, fWindow, tWindow), dtype=complex)
    for i in range(S+1):
        magArray[i] = np.abs(xSTArray[i])
        phaseArray[i] = np.exp(1.j*np.angle(xSTArray[i]))
    
    return magArray, phaseArray, S

def get_iS(mag, phase, Audio_num, Audio_length):    
    y = np.zeros((Audio_length))
    
    for i in range(Audio_num):  
        mult = mag[i]*phase[i]
        xiST = TF.iS(mult)
        
        start = i * mult.shape[1]
        end = (i + 1) * mult.shape[1]
        y[start:end] = xiST
        
    mult = mag[Audio_num]*phase[Audio_num]
    xiST = TF.iS(mult)
    
    start = Audio_length - mult.shape[1]
    end = Audio_length
    y[start:end] = xiST
    
    return y


"""
Get TF Transform
"""
def TF_processing(transform, x, *args):
    if transform == "STFT":
        spec, phase, Num_Audios, t = get_STFT(x, *args)
        x  = spec[:, 0:128, :]
        return x, phase, Num_Audios, t
    elif transform == "WSST":
        ssq, phase, Audio_num = get_WSST(x, *args)
        return ssq, phase, Audio_num
    elif transform == "CWT":
        CWT, phase, Audio_num, freq_size = get_CWT(x, *args)
        return CWT, phase, Audio_num, freq_size
    elif transform == "S":
        sMag, phase, Audio_num = get_S(x, *args)
        stock = sMag[:, :-1, :]
        return stock, phase, Audio_num


def iTF_Processing(transform, x, length, *args):
    if transform == "STFT":
        phase, Num_Audios, t, sr, nperseg, noverlap, fft_size = args
        mask_rz = np.zeros((Num_Audios+1, 129, 64))
        mask_rz[:, 0:128, :] = x
        reconstruction = get_iSTFT(mask_rz, phase, Num_Audios, t, sr, nperseg, noverlap, fft_size)
        return reconstruction[0:length]
    elif transform == "WSST":
        phase, Audio_num, sr, WAVELET, NV, freq_window, time_window, N, *_ = args
        reconstruction = get_iWSST(x, phase, Audio_num, length, WAVELET, freq_window, N)
        return reconstruction[0:length]
    elif transform == "CWT":
        phase, Audio_num, freq_size, sr, WAVELET, NV, freq_window, time_window, N, padtype, scaletype, preset, downsample = args
        reconstruction = get_iCWT(x, phase, Audio_num, length, freq_size, WAVELET, NV, N, padtype, scaletype, preset, downsample)
        return reconstruction[0:length]
    elif transform == "S":
        phase, Audio_num, sr, timeWindow = args
        tWindow = int(timeWindow)
        fWindow = int(timeWindow/2 + 1)
        output_rz = np.zeros((Audio_num+1, fWindow, tWindow))
        output_rz[:, 0:fWindow-1, :] = x
        reconstruction = get_iS(output_rz, phase, Audio_num, length)
        return reconstruction


"""
Denoise PCG Signal
"""
def denoise_pcg(transform, model, noisy_path, prediction_path, *args):
    #==================== Loading the audio file ====================
    x, _ = load(noisy_path, sr=args[0])
    original_length = len(x)

    #==================== Audio processing ====================
    x, *output = TF_processing(transform, x, *args)
    
    #==================== Network processing ====================
    mask = NT.Predict(model, x)
    mask = mask[:, :, :, 0]

    #==================== Spectrogram to audio conversion ====================
    y = iTF_Processing(transform, mask, original_length, *output, *args)

    #==================== Saving audio ====================
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    write(prediction_path, args[0], y.astype("float32"))
    return prediction_path


"""
Run Network Iteration
"""

def runNetwork(transform, model, files, test_noisy_dir, test_prediction_dir, *args):
    reset_dir(test_prediction_dir)
    
    predictions = []
    startTime = time.perf_counter()
    for file in tqdm(files, desc="Denoising", unit=" Audio"):
        predictions.append(denoise_pcg(transform, model, Path(test_noisy_dir) / file, Path(test_prediction_dir) / file, *args))

    AvgRunTime = (time.perf_counter() - startTime ) / max(len(files), 1)

    return predictions, AvgRunTime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="insert audio file to denoise")
    parser.add_argument("model", type=Path, help="trained model path")
    parser.add_argument("--output", type=Path, default="output/", help="output directory")
    parser.add_argument("--config", default=None, help="override toml that sets the transform of the model")
    args = parser.parse_args()

    for p in (args.file, args.model):
        if not p.is_file():
            parser.error(f"{p} does not exist")

    import pcg_denoising.config as config
    cfg = config.load_config(override_path=args.config)
    model = NT.load_model(args.model)

    out_file = args.output / f"{args.file.stem}_denoised.wav"
    denoise_pcg(cfg.transform.name, model, args.file, out_file, *transform_args(cfg))
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()