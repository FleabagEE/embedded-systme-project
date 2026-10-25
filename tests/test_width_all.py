import unittest
import numpy as np
import tensorflow as tf
from tests.common import config
from training.dscnn import SharedCNN, channels
from training.losses import width_losses, shared_train_step


class SharedWidthTests(unittest.TestCase):
    def test_shared_identity_joint_loss_gradients_update_and_freeze(self):
        for conv, standard_stem in (('SC', True), ('DW', True), ('DW', False)):
            c = config()
            c['model'].update(conv_type=conv, width='all', standard_stem=standard_stem)
            model = SharedCNN(c)
            x = tf.random.normal([2, 98, 40, 1])
            y = tf.constant([2, 5])
            references = [b.kernel for b in model.blocks]
            with tf.GradientTape(persistent=True) as tape:
                total, losses, _ = width_losses(model, x, y)
                weighted, _, _ = width_losses(model, x, y, weights=(1., 2., 3.))
            self.assertAlmostEqual(float(total), sum(float(v) for v in losses.values()), places=5)
            self.assertAlmostEqual(float(weighted), sum(float(losses[w]) * n for w, n in zip(('1', '5', '25'), (1, 2, 3))), places=5)
            total_gradient = tape.gradient(total, model.blocks[0].kernel).numpy()
            subgradients = [tape.gradient(losses[w], model.blocks[0].kernel).numpy() for w in ('1', '5', '25')]
            np.testing.assert_allclose(total_gradient, sum(subgradients), atol=1e-5, rtol=1e-5)
            for w, gradient in zip(('1', '5', '25'), subgradients):
                count = channels(c['model']['base_channels'], w)[0]
                self.assertGreater(np.abs(gradient[..., :count]).sum(), 0)
                np.testing.assert_array_equal(gradient[..., count:], 0)
            # The second block must also slice its INPUT channel axis.
            half = tape.gradient(losses['5'], model.blocks[1].kernel).numpy()
            half_input = channels(c['model']['base_channels'], '5')[0]
            np.testing.assert_array_equal(half[:, :, half_input:, :], 0)
            if conv == 'DW':
                depthwise = tape.gradient(losses['25'], model.blocks[1].depthwise_kernel).numpy()
                quarter_input = channels(c['model']['base_channels'], '25')[0]
                self.assertGreater(np.abs(depthwise[:, :, :quarter_input]).sum(), 0)
                np.testing.assert_array_equal(depthwise[:, :, quarter_input:], 0)
            classifier = tape.gradient(losses['25'], model.classifier_kernel).numpy()
            quarter_output = channels(c['model']['base_channels'], '25')[-1]
            np.testing.assert_array_equal(classifier[quarter_output:, :], 0)
            before = model.blocks[0].kernel.numpy().copy()
            optimizer = tf.keras.optimizers.SGD(0.01)
            shared_train_step(model, optimizer, x, y)
            self.assertEqual(int(optimizer.iterations), 1)
            self.assertFalse(np.array_equal(before, model.blocks[0].kernel.numpy()))
            for w in ('1', '5', '25'):
                for b, reference in zip(model.blocks, references):
                    self.assertIs(b.kernel, reference)
                frozen = model.freeze(w)
                for frames in (98, 198):
                    probe = tf.random.normal([2, frames, 40, 1])
                    np.testing.assert_allclose(model(probe, width=w).numpy(), frozen(probe).numpy(), atol=1e-5, rtol=1e-5)
                self.assertFalse(any(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in frozen.layers))
