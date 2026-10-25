import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from training.dataset import Record, AudioDataset, load_metadata, partition, read_waveform, cached_waveform
from tests.common import config


class SplitTests(unittest.TestCase):
    def test_mixed_duration_batches_and_original_file_multicrop(self):
        c = config()
        c['training']['duration_secs'] = [1, 2]
        c['training']['batch_size'] = 2
        c['evaluation']['num_crops'] = 3
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for i in range(4):
                path = Path(tmp) / f'{i}.wav'
                sf.write(path, np.sin(np.arange(40000) * 0.1).astype(np.float32) * 0.1, 16000)
                records.append(Record(path, 1, i % 2, 'a'))
            train = AudioDataset(records, c, 'train')
            self.assertEqual([x.shape[1] for x, _, _ in train.batches(0)], [98, 198])
            self.assertEqual([x.shape[1] for x, _, _ in train.batches(1)], [198, 98])
            ids = [file_id for _, _, group in train.batches(training=False, duration=1) for file_id in group]
            self.assertEqual(len(ids), 12)
            self.assertEqual(len(set(ids)), 4)
            self.assertEqual(len(list(train.representative(4))), 4)

    def test_metadata_and_official_fold_partitions(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'metadata').mkdir()
            with (root / 'metadata/UrbanSound8K.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['slice_file_name', 'fold', 'classID', 'class'])
                writer.writeheader()
                for fold in range(1, 11):
                    writer.writerow(dict(slice_file_name=f'{fold}.wav', fold=fold, classID=(fold - 1) % 2, **{'class': ['a', 'b'][(fold - 1) % 2]}))
            records, classes = load_metadata(root)
            self.assertEqual(classes, {0: 'a', 1: 'b'})
            splits = {s: {r.fold for r in partition(records, c, s)} for s in ('train', 'val', 'test')}
            self.assertEqual(splits, {'train': set(range(1, 9)), 'val': {9}, 'test': {10}})
            validation = AudioDataset(records, c, 'val')
            with self.assertRaises(ValueError):
                list(validation.representative(1))

    def test_normalized_stereo_resampling_cache_and_calibration(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stereo.wav'
            sf.write(path, np.tile([0.25, 0.75], (8000, 1)), 8000, subtype='PCM_16')
            y = read_waveform(path)
            self.assertEqual(y.shape, (16000,))
            self.assertAlmostEqual(float(y[100:-100].mean()), 0.5, places=3)
            r = Record(path, 1, 0, 'a')
            c['dataset']['cache_dir'] = str(Path(tmp) / 'cache')
            np.testing.assert_array_equal(cached_waveform(r, c), cached_waveform(r, c))
            with patch('training.dataset.read_waveform', side_effect=AssertionError('cache miss')):
                self.assertEqual(cached_waveform(r, c).shape, (16000,))
            samples = list(AudioDataset([r], c, 'train').representative(5))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].shape, (1, 98, 40, 1))
            # Changed source invalidates the deterministic cache.
            sf.write(path, np.zeros((8000, 2)), 8000, subtype='PCM_16')
            np.testing.assert_array_equal(cached_waveform(r, c), np.zeros(16000))
