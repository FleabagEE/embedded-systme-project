"""Waveform augmentation: pitch-preserving speed, crop/pad, then unclipped gain."""
import numpy as np


def crop_or_pad(waveform, samples, rng=None):
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < samples:
        extra = samples - len(waveform)
        return np.pad(waveform, (extra // 2, extra - extra // 2))
    maximum = len(waveform) - samples
    start = int(rng.integers(maximum + 1)) if rng is not None else maximum // 2
    return waveform[start:start + samples].copy()


def speed_perturb(waveform, factor):
    """factor>1 shortens audio without shifting pitch; silence handles tiny inputs."""
    if factor <= 0:
        raise ValueError('Speed factor must be positive')
    if len(waveform) < 32 or not np.any(waveform):
        length = max(1, round(len(waveform) / factor))
        return np.full(length, waveform[0] if len(waveform) else 0, dtype=np.float32)
    import librosa
    n_fft = min(2048, 2 ** int(np.floor(np.log2(len(waveform)))))
    return librosa.effects.time_stretch(waveform, rate=factor, n_fft=n_fft).astype(np.float32)


def prepare_waveform(waveform, config, training=False, rng=None, duration=None):
    rng = np.random.default_rng() if rng is None else rng
    a, aug = config['audio'], config['augmentation']
    enabled = training and aug['enabled']
    if enabled and rng.random() < aug['speed_probability']:
        waveform = speed_perturb(waveform, rng.uniform(aug['speed_min'], aug['speed_max']))
    # Random training crop even when gain/speed augmentation is disabled.
    waveform = crop_or_pad(waveform, round(a['sample_rate'] * (a['duration_sec'] if duration is None else duration)), rng if training else None)
    if enabled and rng.random() < aug['gain_probability']:
        db = rng.uniform(aug['gain_db_min'], aug['gain_db_max'])
        waveform = waveform * np.float32(10 ** (db / 20))
    return waveform.astype(np.float32)


def evaluation_crops(waveform, samples, count=1):
    if count == 1 or len(waveform) <= samples:
        return [crop_or_pad(waveform, samples)]
    starts = np.unique(np.linspace(0, len(waveform) - samples, count).round().astype(int))
    return [waveform[start:start + samples] for start in starts]
