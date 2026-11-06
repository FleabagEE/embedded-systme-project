import copy
import unittest
from unittest.mock import patch, Mock
import numpy as np
from training.augment import (Compose, crop_or_pad, prepare_waveform, speed_perturb,
                              background_noise, time_mask, frequency_mask)
from training.features import extract_features
from training.dataset import SyntheticDataset
from tests.common import config


class MaximumRNG:
    def integers(self, high):
        return high - 1


class AugmentationTests(unittest.TestCase):
    def test_crop_pad_and_deterministic_evaluation(self):
        for length in (16000, 24000, 32000):
            for wave in (np.ones(31), np.arange(40000)):
                self.assertEqual(crop_or_pad(wave, length).shape, (length,))
        np.testing.assert_array_equal(crop_or_pad(np.ones(2), 6), [0, 0, 1, 1, 0, 0])
        c = config(); c['augmentation']['enabled'] = True
        wave = np.arange(40000, dtype=np.float32)
        np.testing.assert_array_equal(prepare_waveform(wave, c), prepare_waveform(wave, c))

    def test_crop_precedes_transforms_and_multiple_transforms_apply(self):
        c = config()
        c['augmentation'] = {'enabled': True, 'waveform': {
            'RandomGain': {'enabled': True, 'probability': 1, 'min_db': 6, 'max_db': 6},
            'TimeStretch': {'enabled': True, 'probability': 1, 'min_rate': 1.1, 'max_rate': 1.1}}, 'spectrogram': {}}
        with patch('training.augment.speed_perturb', side_effect=lambda y, rate: y[:round(len(y) / rate)]) as speed:
            result = prepare_waveform(np.ones(50000), c, True, np.random.default_rng(2), duration=1.5)
            self.assertEqual(len(speed.call_args.args[0]), 24000)
            self.assertEqual(result.shape, (24000,))
            self.assertAlmostEqual(float(result.max()), 10 ** (6 / 20), places=5)
            prepare_waveform(np.ones(50000), c, False)
            self.assertEqual(speed.call_count, 1)

    def test_independent_probabilities_and_global_disable(self):
        transforms = {'RandomGain': {'enabled': True, 'probability': .5, 'min_db': 6, 'max_db': 6},
                      'RandomTimeShift': {'enabled': True, 'probability': .5, 'max_seconds': .2}}
        rng = Mock(); rng.random.side_effect = [.1, .9]; rng.uniform.return_value = 6
        wave = np.arange(100, dtype=np.float32)
        result = Compose(transforms)(wave, rng, sample_rate=16000)
        np.testing.assert_allclose(result, wave * 10 ** (6 / 20), rtol=1e-6)
        rng.integers.assert_not_called()
        np.testing.assert_array_equal(Compose(transforms, False)(wave, rng, sample_rate=16000), wave)

    def test_dynamic_retrieval_epoch_seed_and_worker_streams(self):
        c = config(); c['training']['batch_size'] = 2
        c['augmentation'] = {'enabled': True, 'waveform': {
            'RandomGain': {'enabled': True, 'probability': 1, 'min_db': -6, 'max_db': 6}}, 'spectrogram': {}}
        dataset = SyntheticDataset(c, 'train', 2)
        first = next(dataset.batches(0))[0]
        second = next(dataset.batches(0))[0]
        epoch1 = next(dataset.batches(1))[0]
        np.testing.assert_array_equal(first, next(SyntheticDataset(c, 'train', 2).batches(0))[0])
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, epoch1))
        self.assertFalse(np.array_equal(first, next(SyntheticDataset(c, 'train', 2, worker_id=1).batches(0))[0]))
        val = SyntheticDataset(c, 'val', 2)
        np.testing.assert_array_equal(next(val.batches())[0], next(val.batches())[0])
        with self.assertRaises(ValueError):
            next(val.batches(training=True))
        np.testing.assert_array_equal(next(dataset.representative(1)), next(dataset.representative(1)))

    def test_all_augmentation_stages_are_bypassed_for_validation_test_and_calibration(self):
        c = config(); c['augmentation']['enabled'] = True
        c['audio']['duration_sec'] = 1.5
        with patch('training.augment.Compose.__call__', side_effect=AssertionError('augmentation leaked')):
            for split in ('val', 'test'):
                source = SyntheticDataset(c, split, 2)
                first = next(source.batches())[0]
                np.testing.assert_array_equal(first, next(source.batches())[0])
                self.assertEqual(first.shape[1], 148)
            representative = next(SyntheticDataset(c, 'train', 2).representative(1))
            self.assertEqual(representative.shape, (1,148,40,1))

    def test_masks_scale_with_length_and_never_remove_all_features(self):
        for frames, expected in [(98, 10), (148, 15), (198, 20)]:
            x = np.arange(frames * 40, dtype=np.float32).reshape(frames, 40, 1)
            masked = time_mask(x, MaximumRNG(), max_frames=10, reference_frames=100)
            self.assertTrue(expected - 1 <= np.any(masked != x, axis=(1, 2)).sum() <= expected)
            masked = frequency_mask(x, MaximumRNG(), max_bins=4)
            self.assertTrue(3 <= np.any(masked != x, axis=(0, 2)).sum() <= 4)
            self.assertTrue(np.any(masked != masked.mean()))
        event = np.full((148,40,1), -10., np.float32)
        event[-1,-1] = 10.
        masked = time_mask(frequency_mask(event, MaximumRNG(), max_bins=4), MaximumRNG(), max_frames=10, reference_frames=100)
        self.assertEqual(float(masked[-1,-1,0]), 10.)

    def test_noise_snr_and_normalization_after_masks(self):
        wave = np.sin(np.arange(24000) * .1).astype(np.float32)
        noisy = background_noise(wave, np.random.default_rng(4), min_snr_db=20, max_snr_db=20)
        actual = 20 * np.log10(np.sqrt(np.mean(wave ** 2)) / np.sqrt(np.mean((noisy - wave) ** 2)))
        self.assertAlmostEqual(float(actual), 20, places=4)
        c = config(); c['audio'].update(normalization='fixed', normalization_mean=1., normalization_std=2.)
        raw = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
        with patch('training.features.log_mel', return_value=raw), patch('training.augment.Compose') as compose:
            compose.return_value.side_effect = lambda x, rng: x + 3
            actual = extract_features(wave, c, training=True, rng=np.random.default_rng(2))
            np.testing.assert_array_equal(actual, (raw + 3 - 1) / 2)
            compose.reset_mock()
            np.testing.assert_array_equal(extract_features(wave, c), (raw - 1) / 2)
            compose.assert_not_called()

    def test_pitch_preserving_speed_pitch_shift_and_degenerate_audio(self):
        wave = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        for factor in (.9, 1.1):
            out = speed_perturb(wave, factor)
            self.assertEqual(len(out), round(len(wave) / factor))
            self.assertLess(abs(np.argmax(abs(np.fft.rfft(out))) * 16000 / len(out) - 440), 10)
        from training.augment import pitch_shift
        shifted = pitch_shift(wave, np.random.default_rng(1), sample_rate=16000, min_semitones=1, max_semitones=1)
        peak = np.argmax(abs(np.fft.rfft(shifted)))
        self.assertLess(abs(peak - 440 * 2 ** (1 / 12)), 10)
        for wave in (np.array([], np.float32), np.zeros(8, np.float32), np.ones(1, np.float32)):
            self.assertTrue(np.all(np.isfinite(speed_perturb(wave, 1.1))))
