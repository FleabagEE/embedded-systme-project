import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from training.control import training_callbacks
from tests.common import config


class TrainingControlTests(unittest.TestCase):
    def test_mixed_accuracy_is_equal_duration_mean_not_pooled(self):
        from training.train import validate_epoch
        c = config()
        c['training']['duration_secs'] = [1, 1.5, 2]
        class Dataset:
            def batches(self, **kwargs):
                duration = kwargs['duration']
                count = {1: 2, 1.5: 1, 2: 7}[duration]
                yield np.zeros((count, 2, 2, 1), np.float32), np.full(count, int(duration == 1.5), np.int32), [str(i) for i in range(count)]
        class Model:
            def __call__(self, x, **kwargs):
                return tf.constant([[2., 0.]] * len(x))
        row = validate_epoch(Model(), Dataset(), c, ['5'], {0:'a',1:'b'}, True)
        self.assertAlmostEqual(row['val_accuracy'], 2/3)
        self.assertNotAlmostEqual(row['val_accuracy'], 9/10)
        self.assertEqual([row[f'val_accuracy_{d}s'] for d in ('1','1.5','2')], [1.,0.,1.])

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

    def test_progress_uses_actual_checkpoint_patience_and_recent_eta(self):
        from training.control import TrainingProgress
        c = config()
        settings = {**c['training'],'early_stopping_patience':2,'eta_window':2,'eta_min_epochs':2}
        model = tf.keras.Sequential([tf.keras.layers.Input((1,)),tf.keras.layers.Dense(1,use_bias=False)])
        optimizer = tf.keras.optimizers.SGD(.01)
        output, rows = [], []
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'best.weights.h5'
            callbacks,early = training_callbacks(model,optimizer,settings,checkpoint)
            clock = iter([0,0,10,10,30,30,60,60,100])
            progress = TrainingProgress(early,6,settings,[1,1.5,2],['1','5','25'],clock=lambda:next(clock),printer=output.append)
            progress.set_model(model); progress.on_train_begin()
            for epoch,accuracy in enumerate((.4,.5,.49,.48)):
                model.set_weights([np.array([[epoch+1]],np.float32)])
                row = {'accuracy':.6,'val_accuracy':accuracy,'val_loss':1/(epoch+1)}
                row.update({f'val_accuracy_{w}_{d}s':accuracy for w in ('1','5','25') for d in ('1','1.5','2')})
                progress.on_epoch_begin(epoch)
                for callback in callbacks: callback.on_epoch_end(epoch,row)
                progress.on_epoch_end(epoch,row); rows.append(row)
            self.assertEqual([r['patience_used'] for r in rows],[0,0,1,2])
            self.assertEqual([r['checkpoint_improved'] for r in rows],[True,True,False,False])
            self.assertEqual([r['epoch_seconds'] for r in rows],[10,20,30,40])
            self.assertIsNone(rows[0]['eta_seconds'])
            self.assertEqual(rows[1]['eta_seconds'],15*4)
            self.assertEqual(rows[2]['eta_seconds'],25*3)
            self.assertEqual(rows[3]['eta_seconds'],0)
            self.assertEqual(rows[-1]['training_elapsed_seconds'],100)
            self.assertEqual(rows[-1]['best_epoch'],2)
            self.assertIn('Patience: 2/2',output[-2])
            self.assertIn('Width 0.25',output[-1]); self.assertIn('1.5s',output[-1])
            model.load_weights(checkpoint)
            self.assertEqual(model.get_weights()[0].item(),2)

    def test_mobile_shared_validation_equal_mean_of_nine_configurations(self):
        from training.train import validate_epoch
        from tests.test_models_extended import mobile_config
        c = mobile_config()
        class Dataset:
            def batches(self, **kwargs):
                d = kwargs['duration']; n = {1:1,1.5:2,2:8}[d]
                yield np.full((n,1,1,1),d,np.float32),np.zeros(n,np.int32),[str(i) for i in range(n)]
        class Model:
            def __call__(self,x,width,**kwargs):
                right = {'1':(1,1.5,2),'5':(1,1.5),'25':(1,)}
                scores = [1.,0.] if float(x[0,0,0,0]) in right[width] else [0.,1.]
                return tf.constant([scores]*len(x))
        row = validate_epoch(Model(),Dataset(),c,['1','5','25'],{0:'a',1:'b'},True)
        self.assertAlmostEqual(row['val_accuracy'],6/9)
        self.assertEqual(sum(row[f'val_accuracy_{w}_{d}s'] for w in ('1','5','25') for d in ('1','1.5','2')),6)
        self.assertNotAlmostEqual(row['val_accuracy'],(11+3+1)/33)
