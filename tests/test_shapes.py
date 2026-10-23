import unittest
import numpy as np
from tests.common import config
from training.dscnn import build_model, architecture_metadata


class ShapeTests(unittest.TestCase):
    def test_legacy_dw_stem_remains_compatible(self):
        from training.dscnn import fix_input_shape
        c = config()
        c['model'].pop('standard_stem')
        model = build_model(c, '5')
        self.assertIsNotNone(model.get_layer('depthwise_0'))
        fixed = fix_input_shape(model, c, 2)
        x = np.ones((1, 198, 40, 1), np.float32)
        np.testing.assert_allclose(model(x).numpy(), fixed(x).numpy(), atol=1e-6)

    def test_same_weights_accept_both_temporal_lengths(self):
        c = config()
        for conv in ('SC', 'DW'):
            for mels in (40, 64):
                c['model']['conv_type'], c['audio']['n_mels'] = conv, mels
                for width in ('1', '5', '25'):
                    model = build_model(c, width)
                    before = model.get_weights()
                    for frames in (98, 198):
                        out = model(np.ones((2, frames, mels, 1), np.float32))
                        self.assertEqual(tuple(out.shape), (2, 10))
                    for a, b in zip(before, model.get_weights()):
                        np.testing.assert_array_equal(a, b)
                    meta = architecture_metadata(model, c, width, 1)
                    self.assertGreater(meta['estimated_macs'], 0)
                    self.assertIsNone(meta['tensor_arena_bytes'])
