"""Deterministic host-side validation windows; no model predictions are cached."""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import time

import numpy as np

from evaluation.metrics import classification_metrics
from training.features import extract_features
from training.utils import validation_protocol


def window_starts(num_samples, window_samples, stride_samples, include_last=True):
    """Use integer sample positions; short/empty clips produce one padded window."""
    if any(type(v) is not int for v in (num_samples, window_samples, stride_samples)):
        raise ValueError('Window positions require integer sample counts')
    if num_samples < 0 or window_samples < 1 or stride_samples < 1:
        raise ValueError('Invalid window length or stride')
    if num_samples <= window_samples:
        return [0]
    last = num_samples - window_samples
    starts = list(range(0, last + 1, stride_samples))
    if include_last and starts[-1] != last:
        starts.append(last)
    return starts


def sliding_windows(waveform, samples, stride, include_last=True):
    for start in window_starts(len(waveform), samples, stride, include_last):
        window = np.asarray(waveform[start:start + samples], np.float32)
        # Sliding windows anchor short clips at sample zero; legacy fixed crops
        # retain their existing centered-padding implementation in dataset.py.
        if len(window) < samples:
            window = np.pad(window, (0, samples - len(window)))
        yield window


class ValidationData:
    """Reuse per-run waveform/features on temporary disk across widths and epochs.

    The source dataset and preprocessing config are fixed for this object's life.
    Cache paths are private to the object/run, never shared across experiments.
    Fixed validation delegates to the unchanged BatchDataset.batches path.
    """
    def __init__(self, dataset, config, cache_dir=None):
        self.dataset, self.config = dataset, copy.deepcopy(config)
        self.protocol = validation_protocol(config)
        self._temporary = None
        self.cache_dir = None
        self.identities = {}
        cache = config.get('validation', {}).get('cache_features', True)
        if self.protocol['method'] == 'sliding_window' and cache:
            if cache_dir is None:
                self._temporary = tempfile.TemporaryDirectory(prefix='urbansound-validation-')
                self.cache_dir = Path(self._temporary.name)
            else:
                # Always reserve a private cache, even when a parent is supplied.
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                self._temporary = tempfile.TemporaryDirectory(prefix='validation-', dir=cache_dir)
                self.cache_dir = Path(self._temporary.name)

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _features(self, index, duration):
        path = self.cache_dir / f'features-{index}-{duration:g}.npy' if self.cache_dir else None
        if path is not None and path.exists():
            label, file_id = self.identities[index]
            return np.load(path, mmap_mode='r', allow_pickle=False), label, file_id
        wave_path = self.cache_dir / f'wave-{index}.npy' if self.cache_dir else None
        if wave_path is not None and wave_path.exists():
            wave = np.load(wave_path, mmap_mode='r', allow_pickle=False)
            label, file_id = self.identities[index]
        else:
            wave, label, file_id = self.dataset.example(index)
            if wave_path is not None:
                np.save(wave_path, wave, allow_pickle=False)
                self.identities[index] = label, file_id
        sr = self.config['audio']['sample_rate']
        windows = sliding_windows(wave, round(duration * sr), round(self.protocol['stride_seconds'] * sr),
                                  self.protocol['include_last_window'])
        # Extract each window exactly: arbitrary final offsets need not align with
        # the STFT hop, so slicing a complete-clip spectrogram is not equivalent.
        features = np.stack([extract_features(w, self.config, training=False) for w in windows])
        if path is not None:
            np.save(path, features, allow_pickle=False)
        return features, label, file_id

    def batches(self, duration):
        if self.protocol['method'] == 'fixed':
            yield from self.dataset.batches(training=False, duration=duration)
            return
        x, labels, ids = [], [], []
        size = self.protocol['batch_size']
        for index in range(self.dataset.count):
            features, label, file_id = self._features(index, duration)
            for feature in features:
                x.append(feature); labels.append(label); ids.append(file_id)
                if len(x) == size:
                    yield np.stack(x), np.asarray(labels, np.int32), ids
                    x, labels, ids = [], [], []
        if x:
            yield np.stack(x), np.asarray(labels, np.int32), ids


def evaluate_configurations(predictors, data, classes, duration, *, synthetic=False, from_logits=True,
                            include_loss=False):
    """Preprocess each batch once, then invoke every current model/width on it."""
    started = time.perf_counter()
    outputs = {key: [] for key in predictors}
    inference = {key: 0. for key in predictors}
    labels, ids, preprocessing = [], [], 0.
    batches = iter(data.batches(duration))
    while True:
        before = time.perf_counter()
        try:
            x, y, file_ids = next(batches)
        except StopIteration:
            preprocessing += time.perf_counter() - before
            break
        preprocessing += time.perf_counter() - before
        labels.extend(y.tolist()); ids.extend(file_ids)
        for key, predict in predictors.items():
            before = time.perf_counter()
            # Conversion synchronizes accelerator outputs before measuring time.
            output = np.asarray(predict(x))
            inference[key] += time.perf_counter() - before
            outputs[key].extend(output)
    results = {}
    for key, values in outputs.items():
        values = np.asarray(values)
        metric = classification_metrics(labels, values, ids, classes, getattr(data.dataset, 'split', 'val'), synthetic,
                                        from_logits=from_logits, aggregation=data.protocol['aggregation'])
        if data.protocol['method'] == 'sliding_window':
            metric['aggregation'] = f"{data.protocol['aggregation']} across deterministic sliding windows per original clip, then argmax."
        metric.update(validation_protocol=data.protocol, num_windows=len(labels), inference_seconds=inference[key])
        if include_loss:
            import tensorflow as tf
            metric['window_loss'] = float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                labels, values, from_logits=from_logits)))
        results[key] = metric
    stats = {'elapsed_seconds': time.perf_counter() - started, 'preprocessing_seconds': preprocessing,
             'inference_seconds': sum(inference.values()), 'num_clips': len(set(ids)), 'num_windows': len(labels),
             'num_window_predictions': len(labels) * len(predictors)}
    return results, stats
