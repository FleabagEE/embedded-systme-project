import unittest
import numpy as np
from training.features import feature_shape, log_mel
from tests.common import config


class FeatureTests(unittest.TestCase):
    def test_shapes_and_finite_reproducible_silence(self):
        c = config()
        for duration in (1, 2):
            for mels in (40, 64):
                c['audio'].update(duration_sec=duration, n_mels=mels)
                y = np.zeros(16000 * duration, np.float32)
                out = log_mel(y, c['audio'])
                self.assertEqual(out.shape, (98 if duration == 1 else 198, mels, 1))
                self.assertEqual(out.shape, feature_shape(c['audio']))
                self.assertTrue(np.all(np.isfinite(out)))
                np.testing.assert_array_equal(out, log_mel(y, c['audio']))
                np.testing.assert_allclose(out, np.log(1e-6), rtol=1e-6)

    def test_reference_stft_and_mel(self):
        import tensorflow as tf
        import librosa
        c = config()
        y = np.random.default_rng(2).normal(size=16000).astype(np.float32)
        stft = tf.signal.stft(y, frame_length=400, frame_step=160, fft_length=512)
        power = abs(stft.numpy()) ** 2
        mel = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, htk=False, norm='slaney')
        expected = np.log(np.maximum(power @ mel.T, 1e-6))
        np.testing.assert_allclose(log_mel(y, c['audio'])[..., 0], expected, atol=3e-5, rtol=3e-5)
