import pcg_denoising.utils.dataset_writer as DR
import pcg_denoising.utils.time_frequency as TF


WAVELET_TRANSFORMS = {"WSST": TF.make_wsst, "CWT": TF.make_cwt}


def build_spectrograms(target_files, noisy_files, save_path, sr, nperseg, noverlap, fft_size, frames=64):
    n_freq = fft_size // 2 + 1
    transform = TF.make_stft(sr, nperseg, noverlap, fft_size)
    return DR.build_pair(save_path, target_files, noisy_files, sr=sr, transform=transform, freq_window=n_freq, time_window=frames)


def build_wavelet(method, target_files, noisy_files, save_path, overlap, sr, wavelet, NV, freq_window, time_window, N, padtype, scaletype, preset, downsample):
    transform = WAVELET_TRANSFORMS[method](wavelet, NV, N, padtype, scaletype, preset, downsample)
    return DR.build_pair(save_path, target_files, noisy_files, sr=sr, transform=transform, freq_window=freq_window, overlap=overlap, time_window=time_window)


def build_stockwell(target_files, noisy_files, save_path, sr, window):
    t_window = int(window)
    f_window = int(window / 2 + 1)
    transform = TF.make_stockwell(sr)
    return DR.build_pair(save_path, target_files, noisy_files, sr=sr, transform=transform, freq_window=f_window, time_window=t_window, per_window=True)
