"""Seeded Compose augmentation. Cropping precedes waveform transforms; masks follow Log-Mel."""
from __future__ import annotations
import numpy as np


def crop_or_pad(waveform, samples, rng=None):
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < samples:
        extra = samples - len(waveform)
        return np.pad(waveform, (extra // 2, extra - extra // 2))
    maximum = len(waveform) - samples
    start = int(rng.integers(maximum + 1)) if rng is not None else maximum // 2
    result = waveform[start:start + samples].copy()
    # Do not select an entirely silent training crop from an otherwise nonzero clip.
    if rng is not None and not np.any(result) and np.any(waveform):
        active = np.flatnonzero(waveform)
        anchor = int(active[int(rng.integers(len(active)))])
        start = max(0, min(maximum, anchor - samples // 2))
        result = waveform[start:start + samples].copy()
    return result


def preserve_length(waveform, samples):
    if len(waveform) <= samples:
        return crop_or_pad(waveform, samples)
    # Retain the highest-energy window after slowing down, rather than dropping
    # a brief event at an edge with an arbitrary crop. Energy is not a semantic label.
    energy = np.concatenate(([0.], np.cumsum(np.square(waveform, dtype=np.float64))))
    start = int(np.argmax(energy[samples:] - energy[:-samples]))
    return np.asarray(waveform[start:start + samples], np.float32)


def speed_perturb(waveform, factor):
    """Pitch-preserving stretch: factor > 1 shortens the signal."""
    if factor <= 0:
        raise ValueError('Speed factor must be positive')
    if len(waveform) < 32 or not np.any(waveform):
        return np.full(max(1, round(len(waveform) / factor)), waveform[0] if len(waveform) else 0, np.float32)
    import librosa
    return librosa.effects.time_stretch(waveform, rate=factor,
        n_fft=min(2048, 2 ** int(np.floor(np.log2(len(waveform)))))).astype(np.float32)


def time_shift(y, rng, sample_rate, max_seconds, **_):
    maximum = min(len(y) - 1, round(sample_rate * max_seconds))
    return np.roll(y, int(rng.integers(-maximum, maximum + 1))) if maximum > 0 else y.copy()


def gain(y, rng, min_db, max_db, **_):
    return y * np.float32(10 ** (rng.uniform(min_db, max_db) / 20))


def background_noise(y, rng, min_snr_db, max_snr_db, source='gaussian', noise_provider=None, **_):
    noise = (noise_provider(rng, len(y)) if source == 'training_audio' and noise_provider is not None
             else rng.standard_normal(len(y)).astype(np.float32))
    signal_rms = np.sqrt(np.mean(np.square(y, dtype=np.float64)))
    noise_rms = np.sqrt(np.mean(np.square(noise, dtype=np.float64)))
    if signal_rms < 1e-12 or noise_rms < 1e-12:
        return y.copy()
    scale = signal_rms / (noise_rms * 10 ** (rng.uniform(min_snr_db, max_snr_db) / 20))
    return y + noise * np.float32(scale)


def pitch_shift(y, rng, sample_rate, min_semitones, max_semitones, **_):
    if len(y) < 32 or not np.any(y):
        return y.copy()
    import librosa
    return librosa.effects.pitch_shift(y, sr=sample_rate,
        n_steps=rng.uniform(min_semitones, max_semitones),
        n_fft=min(2048, 2 ** int(np.floor(np.log2(len(y)))))).astype(np.float32)


def time_stretch(y, rng, min_rate, max_rate, **_):
    return preserve_length(speed_perturb(y, rng.uniform(min_rate, max_rate)), len(y))


def frequency_mask(x, rng, max_bins, **_):
    out = x.copy()
    maximum = min(max_bins, max(0, x.shape[1] // 5))
    width = int(rng.integers(maximum + 1))
    if width:
        start = int(rng.integers(x.shape[1] - width + 1))
        out[:, start:start + width] = np.mean(x)
        peak = int(np.argmax(np.max(x, axis=(0, 2))))
        out[:, peak] = x[:, peak]  # Keep a salient band for narrow-band events.
    return out


def time_mask(x, rng, max_frames, reference_frames, **_):
    out = x.copy()
    maximum = min(round(max_frames * x.shape[0] / reference_frames), max(0, x.shape[0] // 5))
    width = int(rng.integers(maximum + 1))
    if width:
        start = int(rng.integers(x.shape[0] - width + 1))
        out[start:start + width] = np.mean(x)
        peak = int(np.argmax(np.max(x, axis=(1, 2))))
        out[peak] = x[peak]  # Do not completely mask a single dominant transient.
    return out


TRANSFORMS = {'RandomTimeShift': time_shift, 'RandomGain': gain,
              'AddBackgroundNoise': background_noise, 'PitchShift': pitch_shift,
              'TimeStretch': time_stretch, 'FrequencyMask': frequency_mask, 'TimeMask': time_mask}
PARAMETERS = {'RandomTimeShift': {'max_seconds'}, 'RandomGain': {'min_db', 'max_db'},
              'AddBackgroundNoise': {'min_snr_db', 'max_snr_db', 'source'},
              'PitchShift': {'min_semitones', 'max_semitones'}, 'TimeStretch': {'min_rate', 'max_rate'},
              'FrequencyMask': {'max_bins'}, 'TimeMask': {'max_frames', 'reference_frames'}}


def transform_config(config):
    """Translate old augmentation records when reading historical experiments."""
    if 'waveform' in config:
        return config
    return {'enabled': config['enabled'], 'waveform': {
        'TimeStretch': {'enabled': True, 'probability': config.get('speed_probability', .5),
                        'min_rate': config.get('speed_min', .9), 'max_rate': config.get('speed_max', 1.1)},
        'RandomGain': {'enabled': True, 'probability': config.get('gain_probability', .8),
                       'min_db': config.get('gain_db_min', -6), 'max_db': config.get('gain_db_max', 6)}}, 'spectrogram': {}}


def validate_augmentation(config):
    config = transform_config(config)
    if type(config['enabled']) is not bool:
        raise ValueError('augmentation.enabled must be boolean')
    for stage, allowed in [('waveform', set(TRANSFORMS) - {'FrequencyMask', 'TimeMask'}),
                           ('spectrogram', {'FrequencyMask', 'TimeMask'})]:
        for name, params in config.get(stage, {}).items():
            if name not in allowed or set(params) - ({'enabled', 'probability'} | PARAMETERS.get(name, set())):
                raise ValueError(f'Unknown {stage} transform or parameter: {name}')
            if type(params.get('enabled')) is not bool or not 0 <= params.get('probability', -1) <= 1:
                raise ValueError(f'{name} needs enabled and probability in [0,1]')
            if PARAMETERS[name] - set(params) - {'source'}:
                raise ValueError(f'Missing parameters for {name}')
            for low, high in [('min_db', 'max_db'), ('min_snr_db', 'max_snr_db'),
                              ('min_semitones', 'max_semitones'), ('min_rate', 'max_rate')]:
                if low in params and (not np.isfinite([params[low], params[high]]).all() or params[low] > params[high]):
                    raise ValueError(f'Invalid range for {name}')
            if name == 'TimeStretch' and params['min_rate'] <= 0:
                raise ValueError('TimeStretch rates must be positive')
            if name == 'AddBackgroundNoise' and (params['min_snr_db'] < 0 or params.get('source', 'gaussian') not in ('gaussian', 'training_audio')):
                raise ValueError('Noise needs nonnegative SNR and gaussian/training_audio source')
            for key in ('max_seconds', 'max_bins', 'max_frames'):
                if key in params and (not np.isfinite(params[key]) or params[key] < 0):
                    raise ValueError(f'{name}.{key} must be nonnegative')
            if name == 'TimeMask' and params['reference_frames'] <= 0:
                raise ValueError('reference_frames must be positive')


class Compose:
    """Independent Bernoulli decisions on every call; no internal RNG or cached result."""
    def __init__(self, transforms, enabled=True):
        self.transforms = transforms
        self.enabled = enabled

    def __call__(self, value, rng, **context):
        value = np.asarray(value, np.float32).copy()
        if self.enabled:
            for name, settings in self.transforms.items():
                if settings['enabled'] and rng.random() < settings['probability']:
                    params = {k: v for k, v in settings.items() if k not in ('enabled', 'probability')}
                    value = TRANSFORMS[name](value, rng=rng, **params, **context)
        return np.asarray(value, np.float32)


def prepare_waveform(waveform, config, training=False, rng=None, duration=None, noise_provider=None):
    rng = np.random.default_rng() if rng is None else rng
    a = config['audio']
    seconds = a['duration_sec'] if duration is None else duration
    # Crop first, then online transforms, including stretch with length preservation.
    waveform = crop_or_pad(waveform, round(a['sample_rate'] * seconds), rng if training else None)
    if training:
        aug = transform_config(config['augmentation'])
        waveform = Compose(aug['waveform'], aug['enabled'])(waveform, rng,
                        sample_rate=a['sample_rate'], noise_provider=noise_provider)
    return waveform.astype(np.float32)


def evaluation_crops(waveform, samples, count=1):
    if count == 1 or len(waveform) <= samples:
        return [crop_or_pad(waveform, samples)]
    starts = np.unique(np.linspace(0, len(waveform) - samples, count).round().astype(int))
    return [waveform[start:start + samples] for start in starts]
