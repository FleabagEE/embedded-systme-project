import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from training.control import training_callbacks
from tests.common import config


class TrainingControlTests(unittest.TestCase):
    def test_accuracy_checkpoint_patience_and_lr_reduction(self):
        c = config()
        settings = {**c['training'], 'monitor': 'val_accuracy', 'early_stopping_patience': 2, 'lr_patience': 1}
        model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1, use_bias=False)])
        optimizer = tf.keras.optimizers.SGD(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.weights.h5'
            callbacks, early = training_callbacks(model, optimizer, settings, path)
            waits = []
            for epoch, (accuracy, loss) in enumerate(zip([0.44, 0.52, 0.50, 0.49], [1.0, 1.1, 0.9, 0.8])):
                model.set_weights([np.array([[epoch + 1]], np.float32)])
                for callback in callbacks:
                    callback.on_epoch_end(epoch, {'val_accuracy': accuracy, 'val_loss': loss})
                waits.append(early.wait)
            self.assertEqual(waits, [0, 0, 1, 2])
            self.assertTrue(model.stop_training)
            self.assertEqual(early.best_epoch, 1)
            self.assertLess(float(optimizer.learning_rate.numpy()), 0.01)
            model.load_weights(path)
            self.assertEqual(float(model.get_weights()[0][0, 0]), 2)
