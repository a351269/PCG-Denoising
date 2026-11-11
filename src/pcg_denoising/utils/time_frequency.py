import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import toeplitz
from scipy.signal import stft
from ssqueezepy import ssq_cwt, Wavelet
from ssqueezepy.utils import cwt_scalebounds, make_scales

def make_stft(sr, NPERSEG, noverlap, FFT_SIZE):
    return lambda xs: stft(xs, fs=sr, nperseg=NPERSEG, noverlap=noverlap, nfft=FFT_SIZE)[2]

def make_wsst(wavelet, NV, N, padtype, scaletype, preset, downsample):
    return lambda xs: WSST(xs, wavelet, NV, N, padtype, scaletype, preset, downsample)[0]

def make_cwt(wavelet, NV, N, padtype, scaletype, preset, downsample):
    return lambda xs: CWT(xs, wavelet, NV, N, padtype, scaletype, preset, downsample)[0]

def make_stockwell(sr):
    return lambda xw: S(xw, sr)[0]


def CWT(xs, wavelet, NV, N, padtype, scaletype, preset, downsample):
    wavelet = Wavelet(wavelet, N=N, dtype='float64')
    min_scale, max_scale = cwt_scalebounds(wavelet, N=N,  preset=preset)
    scales = make_scales(N, min_scale, max_scale, nv=NV, scaletype=scaletype, wavelet=wavelet, downsample=downsample)
    
    Tx, Wx, freqs, *_ = ssq_cwt(xs, wavelet, scales=scales, padtype=padtype)
    
    return Wx, freqs


def WSST(xs, wavelet, NV, N, padtype, scaletype, preset, downsample):
    wavelet = Wavelet(wavelet, N=N, dtype='float64')
    min_scale, max_scale = cwt_scalebounds(wavelet, N=N,  preset=preset)
    scales = make_scales(N, min_scale, max_scale, nv=NV, scaletype=scaletype, wavelet=wavelet, downsample=downsample)
    
    Tx, Wx, freqs, *_ = ssq_cwt(xs, wavelet, scales=scales, padtype=padtype)
    
    return Tx, freqs


def S(x, Fs):
    N = x.size
    Nh = np.int64(np.fix(N/2))
    odvn=1

    if N%2:
        odvn=1
    else:
        odvn = 0
    
    freq=np.concatenate((np.arange(0,Nh+1), np.arange(-Nh+1-odvn, 0)))/N
    Xft = fft(x)

    # Calcula todas las Gaussianas en la frecuencia como una sola matriz
    invfreq = 1/freq[1:Nh+1]
    tmp1 = np.tile(freq,(Nh,1))
    tmp2 = np.tile(np.transpose(np.reshape(invfreq,(1,Nh))),(1,N))
    waves = 2*np.pi*np.multiply(tmp1,tmp2)
    gausswin = np.exp(-np.square(waves)/2)

    # Genera una matrix Toeplitz usando versiones desplazadas de la FFT (shifted)
    wgXft = toeplitz(np.conj(np.transpose(Xft[0:Nh+1])), Xft)

    # elimina la primera fila correspondiente a la componente de DC
    wgXft = wgXft[1:Nh+1,:]

    # Calcula la transformada de Stockwell como la ifft de las versiones desplzadas
    # de la FFT multiplicadas por la correspondiente ventana Gaussiana
    xST = ifft(np.multiply(wgXft,gausswin), axis=1)

    # ahora calcula la componente de DC (frecuencia cero) y la agrega al resultado
    xSTdc = np.mean(x)*np.ones((1,N))
    xST = np.concatenate((xSTdc, xST))

    t = np.arange(0, N)/Fs
    freq = np.arange(0, Nh)/N*Fs


    # regresa la transformada S de la señal de entrada
    return(xST, t, freq)


def iS(xST):
    M,N = xST.shape
    Nh = np.int64(N/2)

    ss = np.transpose(xST.sum(axis=1))

    # corrige el problema de signo en la parte imaginaria
    ss_real = np.real(ss)
    ss_imag = np.imag(ss)
    ss_imag[0:Nh] = -1*ss_imag[0:Nh]
    ss = ss_real + 1j*ss_imag

    # calcula la transformada inversa de Fourier
    xiST = 2*np.real(ifft(ss, n=N))  

    return(xiST)