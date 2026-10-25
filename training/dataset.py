"""Metadata splits and bounded-memory, per-epoch waveform/feature batches."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from training.augment import prepare_waveform, evaluation_crops
from training.features import log_mel
from training.utils import atomic_write, digest, training_durations


@dataclass(frozen=True)
class Record:
    path: Path
    fold: int
    label: int
    class_name: str

    @property
    def file_id(self):
        return f'fold{self.fold}/{self.path.name}'


def load_metadata(root):
    root = Path(root).expanduser().resolve()
    metadata = root / 'metadata' / 'UrbanSound8K.csv'
    if not metadata.is_file():
        raise FileNotFoundError(f'Expected local metadata at {metadata}; dataset is never downloaded automatically')
    records, classes, seen = [], {}, set()
    with metadata.open(newline='') as stream:
        for row in csv.DictReader(stream):
            fold, label, name = int(row['fold']), int(row['classID']), row['class']
            filename = row['slice_file_name']
            if not 1 <= fold <= 10 or Path(filename).name != filename:
                raise ValueError(f'Invalid metadata fold or filename: {row}')
            path = root / 'audio' / f'fold{fold}' / filename
            if (fold, filename) in seen:
                raise ValueError(f'Duplicate original recording: {path}')
            seen.add((fold, filename))
            if label in classes and classes[label] != name:
                raise ValueError(f'Inconsistent class name for classID {label}')
            classes[label] = name
            records.append(Record(path, fold, label, name))
    if not records or sorted(classes) != list(range(len(classes))):
        raise ValueError('Expected nonempty data with contiguous native classIDs starting at zero')
    return records, dict(sorted(classes.items()))


def partition(records, config, split):
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Unknown split: {split}')
    folds = set(config['dataset'][f'{split}_folds'])
    return [r for r in records if r.fold in folds]


def read_waveform(path, sample_rate=16000):
    # soundfile converts PCM to normalized float exactly once.
    y, native_rate = sf.read(path, dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    if not np.all(np.isfinite(y)):
        raise ValueError(f'Nonfinite audio: {path}')
    if native_rate != sample_rate and len(y):
        import librosa
        y = librosa.resample(y, orig_sr=native_rate, target_sr=sample_rate, res_type='soxr_hq')
    return np.asarray(y, dtype=np.float32)


def cached_waveform(record, config):
    cache = config['dataset'].get('cache_dir')
    if not cache:
        return read_waveform(record.path, config['audio']['sample_rate'])
    stat = record.path.stat()
    key = digest({'path': str(record.path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                  'audio': config['audio'], 'frontend_version': 1, 'resampler': 'soxr_hq',
                  'stage': 'normalized-mono-resampled-waveform'})
    path = Path(cache) / f'{key}.npy'
    if path.exists():
        try:
            y = np.load(path, allow_pickle=False)
            if y.ndim == 1 and y.dtype == np.float32 and np.all(np.isfinite(y)):
                return y
        except (ValueError, OSError, EOFError):
            pass
    y = read_waveform(record.path, config['audio']['sample_rate'])
    buffer = io.BytesIO()
    np.save(buffer, y, allow_pickle=False)
    atomic_write(path, buffer.getvalue())
    return y


class AudioDataset:
    def __init__(self, records, config, split, *, limit=None):
        self.config, self.split = config, split
        self.records = partition(records, config, split)
        # Seeded shuffle avoids CSV-order bias; it does not balance class counts.
        self.records = [self.records[i] for i in np.random.default_rng(config['training']['seed']).permutation(len(self.records))]
        if limit is not None:
            if limit < 1:
                raise ValueError('Dataset limit must be positive')
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f'No examples in split {split}')

    def batches(self, epoch=0, batch_size=None, training=None, duration=None):
        rng = np.random.default_rng(np.random.SeedSequence([self.config['training']['seed'], epoch]))
        training = self.split == 'train' if training is None else training
        order = rng.permutation(len(self.records)) if training else np.arange(len(self.records))
        size = batch_size or self.config['training']['batch_size']
        x, y, ids = [], [], []
        batch_index = 0
        durations = training_durations(self.config)
        for index in order:
            r = self.records[index]
            wave = cached_waveform(r, self.config)
            seconds = (durations[(epoch + batch_index) % len(durations)]
                       if training and duration is None else self.config['audio']['duration_sec'] if duration is None else duration)
            if training:
                crops = [prepare_waveform(wave, self.config, True, rng, duration=seconds)]
            else:
                crops = evaluation_crops(wave, round(self.config['audio']['sample_rate'] * seconds), self.config.get('evaluation', {}).get('num_crops', 1))
            for crop in crops:
                x.append(log_mel(crop, self.config['audio']))
                y.append(r.label)
                ids.append(r.file_id)
                if len(x) == size:
                    yield np.stack(x), np.asarray(y, np.int32), ids
                    x, y, ids = [], [], []
                    batch_index += 1
        if x:
            yield np.stack(x), np.asarray(y, np.int32), ids

    def representative(self, count):
        if self.split != 'train':
            raise ValueError('Calibration must use training folds only')
        # Training fold membership with deterministic inference preprocessing, no augmentation.
        for r in self.records[:count]:
            if r.fold in (9, 10) or r.fold not in self.config['dataset']['train_folds']:
                raise ValueError('Held-out fold in calibration')
            wave = prepare_waveform(cached_waveform(r, self.config), self.config, training=False)
            yield log_mel(wave, self.config['audio'])[None]


class SyntheticDataset:
    """Waveform-based smoke data; its metrics are always marked synthetic."""
    def __init__(self, config, split, count=12):
        self.config, self.split, self.count = config, split, count
        self.records = []

    def batches(self, epoch=0, batch_size=None, training=None, duration=None):
        training = self.split == 'train' if training is None else training
        rng = np.random.default_rng(self.config['training']['seed'] + {'train': 0, 'val': 10000, 'test': 20000}[self.split])
        augmentation_rng = np.random.default_rng(self.config['training']['seed'] + epoch)
        size = batch_size or self.config['training']['batch_size']
        x, y, ids = [], [], []
        durations = training_durations(self.config)
        for i in range(self.count):
            label = i % self.config['model']['num_classes']
            t = np.arange(round(self.config['audio']['sample_rate'] * 2.2), dtype=np.float32) / self.config['audio']['sample_rate']
            wave = (0.2 * np.sin(2 * np.pi * (150 + label * 150) * t) + rng.normal(0, 0.02, len(t))).astype(np.float32)
            seconds = (durations[(epoch + i // size) % len(durations)]
                       if training and duration is None else self.config['audio']['duration_sec'] if duration is None else duration)
            wave = prepare_waveform(wave, self.config, training, augmentation_rng, duration=seconds)
            x.append(log_mel(wave, self.config['audio']))
            y.append(label)
            ids.append(f'synthetic/{self.split}/{i}')
            if len(x) == size or i == self.count - 1:
                yield np.stack(x), np.asarray(y, np.int32), ids
                x, y, ids = [], [], []

    def representative(self, count):
        if self.split != 'train':
            raise ValueError('Synthetic calibration still requires split=train')
        source = SyntheticDataset(self.config, 'train', min(count, self.count))
        for x, _, _ in source.batches(batch_size=1, training=False):
            yield x


def make_datasets(config, synthetic=False, limit=None):
    if synthetic:
        classes = {i: f'synthetic_{i}' for i in range(config['model']['num_classes'])}
        return ({s: SyntheticDataset(config, s, limit or 12) for s in ('train', 'val')}, classes)
    records, classes = load_metadata(config['dataset']['root'])
    if len(classes) != config['model']['num_classes']:
        raise ValueError(f"Metadata has {len(classes)} classes, model.num_classes={config['model']['num_classes']}")
    return ({s: AudioDataset(records, config, s, limit=limit) for s in ('train', 'val')}, classes)
