import unittest
from unittest.mock import patch, Mock
import numpy as np
from training.augment import crop_or_pad, prepare_waveform, speed_perturb
from tests.common import config


class AugmentationTests(unittest.TestCase):
    def test_crop_pad_and_determinism(self):
        for length in (16000, 32000):
            for wave in (np.ones(31), np.arange(40000)):
                self.assertEqual(crop_or_pad(wave, length).shape, (length,))
        np.testing.assert_array_equal(crop_or_pad(np.ones(2), 6), [0, 0, 1, 1, 0, 0])
        c = config()
        c['augmentation']['enabled'] = True
        y = np.arange(40000, dtype=np.float32)
        np.testing.assert_array_equal(prepare_waveform(y, c), prepare_waveform(y, c))

    def test_speed_then_crop_then_gain_and_train_only(self):
        c = config()
        c['augmentation'].update(enabled=True, speed_probability=1, gain_probability=1)
        rng = Mock()
        rng.random.return_value = 0
        rng.uniform.side_effect = [1.1, 6]
        rng.integers.return_value = 0
        with patch('training.augment.speed_perturb', return_value=np.ones(20000, np.float32)) as speed:
            result = prepare_waveform(np.ones(32000), c, True, rng)
            speed.assert_called_once()
            self.assertEqual(result.shape, (16000,))
            np.testing.assert_allclose(result, 10 ** (6 / 20), rtol=1e-6)
            self.assertGreater(result.max(), 1)  # Gain intentionally does not clip.
            prepare_waveform(np.ones(32000), c, False, rng)
            self.assertEqual(speed.call_count, 1)

    def test_zero_probabilities_and_random_crop(self):
        c = config()
        c['augmentation'].update(enabled=True, speed_probability=0, gain_probability=0)
        with patch('training.augment.speed_perturb') as speed:
            first = prepare_waveform(np.arange(50000), c, True, np.random.default_rng(1))
            second = prepare_waveform(np.arange(50000), c, True, np.random.default_rng(2))
            self.assertFalse(np.array_equal(first, second))
            speed.assert_not_called()

    def test_real_pitch_preserving_speed_and_degenerate_audio(self):
        wave = np.sin(2 * np.pi * 440 * np.arange(8000) / 16000).astype(np.float32)
        for factor in (0.9, 1.1):
            out = speed_perturb(wave, factor)
            self.assertEqual(len(out), round(len(wave) / factor))
            peak = np.argmax(abs(np.fft.rfft(out))) * 16000 / len(out)
            self.assertLess(abs(peak - 440), 10)
        for wave in (np.array([], np.float32), np.zeros(8, np.float32), np.ones(1, np.float32)):
            self.assertTrue(np.all(np.isfinite(speed_perturb(wave, 1.1))))
