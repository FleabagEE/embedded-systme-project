"""Metadata splits and bounded-memory, per-epoch waveform/feature batches."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from training.augment import prepare_waveform, evaluation_crops
from training.features import extract_features
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
                  'sample_rate': config['audio']['sample_rate'], 'mono': 'mean', 'frontend_version': 2, 'resampler': 'soxr_hq',
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


class BatchDataset:
    """Sequential loader; RNG streams are keyed by seed, epoch, retrieval and worker ID."""
    def _setup(self, config, split, worker_id=0):
        self.config, self.split, self.worker_id = config, split, worker_id
        self._retrievals = {}

    def batches(self, epoch=0, batch_size=None, training=None, duration=None):
        training = self.split == 'train' if training is None else training
        if training and self.split != 'train':
            raise ValueError('Augmentation is only allowed for the training split')
        visit = self._retrievals.get(epoch, 0) if training else 0
        if training:
            self._retrievals[epoch] = visit + 1
        rng = np.random.default_rng(np.random.SeedSequence([
            self.config['training']['seed'], epoch, visit, self.worker_id]))
        order = rng.permutation(self.count) if training else np.arange(self.count)
        size = batch_size or self.config['training']['batch_size']
        durations = training_durations(self.config)
        x, y, ids, batch_index = [], [], [], 0
        for index in order:
            wave, label, file_id = self.example(int(index))
            seconds = (durations[(epoch + batch_index) % len(durations)] if training and duration is None
                       else self.config['audio']['duration_sec'] if duration is None else duration)
            if training:
                crops = [prepare_waveform(wave, self.config, True, rng, duration=seconds,
                         noise_provider=lambda r, n: self.noise(r, n, int(index)))]
            else:
                crops = evaluation_crops(wave, round(self.config['audio']['sample_rate'] * seconds),
                                         self.config.get('evaluation', {}).get('num_crops', 1))
            for crop in crops:
                x.append(extract_features(crop, self.config, training=training, rng=rng))
                y.append(label)
                ids.append(file_id)
                if len(x) == size:
                    yield np.stack(x), np.asarray(y, np.int32), ids
                    x, y, ids = [], [], []
                    batch_index += 1
        if x:
            yield np.stack(x), np.asarray(y, np.int32), ids

    def noise(self, rng, samples, exclude):
        return rng.standard_normal(samples).astype(np.float32)

    def representative(self, count):
        if self.split != 'train':
            raise ValueError('Calibration must use training folds only')
        for index in range(min(count, self.count)):
            wave, _, _ = self.example(index)
            wave = prepare_waveform(wave, self.config, training=False)
            yield extract_features(wave, self.config, training=False)[None]


class AudioDataset(BatchDataset):
    def __init__(self, records, config, split, *, limit=None, worker_id=0):
        self._setup(config, split, worker_id)
        selected = partition(records, config, split)
        self.records = [selected[i] for i in np.random.default_rng(config['training']['seed']).permutation(len(selected))]
        if limit is not None:
            if limit < 1:
                raise ValueError('Dataset limit must be positive')
            self.records = self.records[:limit]
        self.count = len(self.records)
        if not self.count:
            raise ValueError(f'No examples in split {split}')

    def example(self, index):
        record = self.records[index]
        if self.split == 'train' and (record.fold not in self.config['dataset']['train_folds'] or record.fold in (9, 10)):
            raise ValueError('Held-out fold in training/calibration')
        return cached_waveform(record, self.config), record.label, record.file_id

    def noise(self, rng, samples, exclude):
        if self.split != 'train':
            raise ValueError('Background audio must come from training folds only')
        candidates = [i for i in range(self.count) if i != exclude]
        if not candidates:
            return super().noise(rng, samples, exclude)
        wave, _, _ = self.example(candidates[int(rng.integers(len(candidates)))])
        if len(wave) == 0:
            return np.zeros(samples, np.float32)
        if len(wave) < samples:
            wave = np.tile(wave, int(np.ceil(samples / len(wave))))
        start = int(rng.integers(len(wave) - samples + 1))
        return wave[start:start + samples]


class SyntheticDataset(BatchDataset):
    """Waveform-based smoke data; never label its metrics as empirical dataset accuracy."""
    def __init__(self, config, split, count=12, worker_id=0):
        self._setup(config, split, worker_id)
        self.count, self.records = count, []

    def example(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([
            self.config['training']['seed'], {'train': 0, 'val': 10000, 'test': 20000}[self.split], index]))
        label = index % self.config['model']['num_classes']
        t = np.arange(round(self.config['audio']['sample_rate'] * 2.2), dtype=np.float32) / self.config['audio']['sample_rate']
        wave = (0.2 * np.sin(2 * np.pi * (150 + label * 150) * t) + rng.normal(0, 0.02, len(t))).astype(np.float32)
        return wave, label, f'synthetic/{self.split}/{index}'


def make_datasets(config, synthetic=False, limit=None):
    if synthetic:
        classes = {i: f'synthetic_{i}' for i in range(config['model']['num_classes'])}
        return ({s: SyntheticDataset(config, s, limit or 12) for s in ('train', 'val')}, classes)
    records, classes = load_metadata(config['dataset']['root'])
    if len(classes) != config['model']['num_classes']:
        raise ValueError(f"Metadata has {len(classes)} classes, model.num_classes={config['model']['num_classes']}")
    return ({s: AudioDataset(records, config, s, limit=limit) for s in ('train', 'val')}, classes)
