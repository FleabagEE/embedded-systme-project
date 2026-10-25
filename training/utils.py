"""Configuration, deterministic runtime setup and atomic experiment IO."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

WIDTHS = {'1': 1.0, '5': 0.5, '25': 0.25}


def width_token(value):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError('model.width must be the exact token 1, 5, 25, or all')
    token = str(value)
    if token not in (*WIDTHS, 'all'):
        raise ValueError('model.width must be the exact token 1, 5, 25, or all')
    return token


def canonical_name(config, width=None):
    return f"DSCNN_{config['audio']['n_mels']}_{config['model']['conv_type']}_{width_token(config['model']['width'] if width is None else width)}"


def duration_token(duration):
    return f'{float(duration):g}s'


def training_durations(config):
    return config['training'].get('duration_secs') or [config['audio']['duration_sec']]


def export_durations(config):
    q = config['quantization']
    return q.get('output_duration_secs') or [q.get('output_duration_sec', config['audio']['duration_sec'])]


def run_duration_token(config):
    return '_'.join(duration_token(d) for d in training_durations(config))


def set_override(config, expression):
    if '=' not in expression:
        raise ValueError(f'Expected key=value override: {expression}')
    key, value = expression.split('=', 1)
    path = key.split('.')
    target = config
    for part in path[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f'Unknown configuration key: {key}')
        target = target[part]
    if path[-1] not in target:
        raise ValueError(f'Unknown configuration key: {key}')
    target[path[-1]] = yaml.safe_load(value)


def load_config(path, overrides=()):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    for override in overrides:
        set_override(config, override)
    validate_config(config)
    return config


def validate_config(c):
    sections = ('dataset', 'audio', 'augmentation', 'model', 'training', 'all_width_training', 'quantization', 'runtime')
    if not isinstance(c, dict) or any(not isinstance(c.get(key), dict) for key in sections):
        raise ValueError(f'Configuration needs mappings for: {", ".join(sections)}')
    a, m, t, q = (c[k] for k in ('audio', 'model', 'training', 'quantization'))
    m['width'] = width_token(m['width'])
    if a['sample_rate'] != 16000 or a['duration_sec'] not in (1, 2) or a['n_mels'] not in (40, 64):
        raise ValueError('Use 16000 Hz, duration_sec 1 or 2, and n_mels 40 or 64')
    if m['conv_type'] not in ('SC', 'DW') or not m['global_average_pooling']:
        raise ValueError('conv_type must be SC/DW and global_average_pooling must be true')
    if m['num_blocks'] != len(m['base_channels']) or not m['base_channels'] or any(type(x) is not int or x < 1 for x in m['base_channels']):
        raise ValueError('base_channels must contain one positive integer per num_blocks')
    if type(m['kernel_size']) is not int or m['kernel_size'] < 1 or m['num_classes'] < 2:
        raise ValueError('kernel_size must be a positive integer and num_classes >= 2')
    if type(m['num_classes']) is not int:
        raise ValueError('num_classes must be an integer')
    if a['center'] or a['normalization'] != 'none' or a['mono'] != 'mean' or a['padding'] != 'center':
        raise ValueError('Frontend supports center=false, normalization=none, mono=mean, padding=center')
    frame = round(a['frame_length_ms'] * a['sample_rate'] / 1000)
    hop = round(a['hop_length_ms'] * a['sample_rate'] / 1000)
    if frame < 2 or hop < 1 or a['fft_length'] < frame or a['log_offset'] <= 0 or a['spectrum'] not in ('power', 'magnitude'):
        raise ValueError('Invalid STFT/log configuration')
    folds = [c['dataset'][f'{s}_folds'] for s in ('train', 'val', 'test')]
    if any(not f or len(set(f)) != len(f) or any(type(x) is not int or not 1 <= x <= 10 for x in f) for f in folds):
        raise ValueError('Each split needs unique integer folds in [1,10]')
    if any(set(folds[i]) & set(folds[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Train, validation and test folds must be disjoint')
    if set(folds[0]) & {9, 10}:
        raise ValueError('Folds 9 and 10 are excluded from training and PTQ calibration')
    aug = c['augmentation']
    if any(not 0 <= aug[k] <= 1 for k in ('speed_probability', 'gain_probability')):
        raise ValueError('Augmentation probabilities must be in [0,1]')
    if not 0 < aug['speed_min'] <= aug['speed_max'] or aug['gain_db_min'] > aug['gain_db_max']:
        raise ValueError('Invalid augmentation ranges')
    if t['epochs'] < 1 or t['batch_size'] < 1 or t['learning_rate'] <= 0 or t['early_stopping_patience'] < 0 or t['optimizer'] not in ('adam', 'sgd'):
        raise ValueError('Invalid training settings (optimizer must be adam or sgd)')
    if any(type(t[key]) is not int for key in ('epochs', 'batch_size', 'early_stopping_patience', 'seed')) or t['seed'] < 0:
        raise ValueError('epochs, batch_size, patience and seed must be integers; seed must be nonnegative')
    aw = c['all_width_training']
    if [width_token(w) for w in aw['width_order']] != list(WIDTHS) or not aw['shared_weights']:
        raise ValueError('all requires shared_weights=true and width_order=[1,5,25]')
    if len(aw['loss_weights']) != 3 or any(not math.isfinite(w) or w <= 0 for w in aw['loss_weights']):
        raise ValueError('Provide three finite positive width loss weights')
    for label, values in [('training.duration_secs', training_durations(c)), ('quantization.output_duration_secs', export_durations(c))]:
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or any(type(d) not in (int, float) or d not in (1, 2) for d in values):
            raise ValueError(f'{label} must contain unique durations 1 and/or 2')
    if t.get('monitor', 'val_loss') not in ('val_accuracy', 'val_macro_f1', 'val_loss'):
        raise ValueError('training.monitor must be val_accuracy, val_macro_f1, or val_loss')
    if t.get('lr_patience', 3) < 1 or not 0 < t.get('lr_factor', 0.5) < 1 or t.get('min_learning_rate', 1e-5) <= 0:
        raise ValueError('Invalid learning-rate plateau settings')
    if type(c.get('evaluation', {}).get('num_crops', 1)) is not int or c.get('evaluation', {}).get('num_crops', 1) < 1:
        raise ValueError('evaluation.num_crops must be a positive integer')
    if q['representative_samples'] < 1 or not q['int8_in_and_out']:
        raise ValueError('INT8 export needs representative_samples>0, int8_in_and_out=true, duration 1/2')
    if type(q['representative_samples']) is not int:
        raise ValueError('representative_samples must be an integer')
    if c['runtime']['device'] not in ('cpu', 'gpu', 'auto'):
        raise ValueError('runtime.device must be cpu, gpu, or auto')
    return c


def configure_runtime(config):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'urbansound-numba-cache'))
    import tensorflow as tf
    from training.runtime import select_device
    select_device(tf, config['runtime'])
    seed = config['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if config['runtime']['deterministic']:
        tf.config.experimental.enable_op_determinism()
        if platform.system() == 'Darwin' and tf.config.get_visible_devices('GPU'):
            print('Runtime: deterministic ops requested on Metal. Unsupported deterministic kernels may fail; '
                  'GPU reproducibility and speed are backend-dependent. CPU helpers may still be used.', flush=True)
    return tf


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content.encode() if isinstance(content, str) else content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def software_versions():
    import importlib.metadata
    versions = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}
    for package in ('tensorflow', 'tensorflow-metal', 'keras', 'numpy', 'librosa', 'soundfile', 'PyYAML', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        versions['git_commit'] = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        versions['git_commit'] = None
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ('training', 'quantization', 'evaluation', 'data')
               for p in sorted((root / folder).glob('*.py'))}
    versions['source_sha256'] = digest(sources)
    versions['requirements_lock_sha256'] = hashlib.sha256((root / 'requirements-lock.txt').read_bytes()).hexdigest()
    return versions
