"""Shared deterministic Log-Mel frontend; output order is time, mel, channel."""
from functools import lru_cache
import numpy as np


def feature_shape(audio, duration=None):
    samples = round(audio['sample_rate'] * (audio['duration_sec'] if duration is None else duration))
    frame = round(audio['sample_rate'] * audio['frame_length_ms'] / 1000)
    hop = round(audio['sample_rate'] * audio['hop_length_ms'] / 1000)
    return (1 + (samples - frame) // hop, audio['n_mels'], 1)


@lru_cache(maxsize=16)
def mel_filter(sample_rate, fft_length, n_mels):
    import librosa
    return librosa.filters.mel(sr=sample_rate, n_fft=fft_length, n_mels=n_mels,
                               fmin=0, fmax=sample_rate / 2, htk=False, norm='slaney').T


def log_mel(waveform, audio):
    """Periodic Hann 400 samples, hop 160, right-zero-pad FFT to 512, no centering.

    Slaney mel and area normalization; natural log(max(mel, epsilon)); no
    per-clip normalization, so 1s/2s and calibration use identical scaling.
    """
    y = np.asarray(waveform, dtype=np.float32)
    frame = round(audio['sample_rate'] * audio['frame_length_ms'] / 1000)
    hop = round(audio['sample_rate'] * audio['hop_length_ms'] / 1000)
    if y.ndim != 1 or len(y) < frame or not np.all(np.isfinite(y)):
        raise ValueError('Log-Mel needs a finite mono waveform of at least one frame')
    windows = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    hann = np.hanning(frame + 1)[:-1].astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windows * hann, n=audio['fft_length'], axis=-1)).astype(np.float32)
    if audio['spectrum'] == 'power':
        spectrum = spectrum ** 2
    mel = spectrum @ mel_filter(audio['sample_rate'], audio['fft_length'], audio['n_mels'])
    return np.log(np.maximum(mel, audio['log_offset'])).astype(np.float32)[..., None]
