"""Small deterministic models cover shared kernels, BN isolation and static export."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tensorflow as tf

from tests.common import config
from training.models import build_model, build_shared_model, fix_input_shape, architecture_metadata
from training.losses import width_losses, shared_train_step, sample_widths
from training.utils import validate_config


def mobile_config():
    c = config()
    c['model'].update(architecture='mobilenetv2', width='all', stem_channels=4,
                      stage_channels=[4,8], stage_strides=[1,2], stage_repeats=[1,1], pooling='mix')
    c['training']['duration_secs'] = [1,1.5,2]
    c['quantization']['output_duration_secs'] = []
    return c


class ExtendedModelTests(unittest.TestCase):
    def test_dscnn_pools_preserve_shared_outputs_and_standard_stem(self):
        for conv in ('SC','DW'):
            for pooling in ('avg','max','mix'):
                with self.subTest(conv=conv,pooling=pooling):
                    c = config(); c['model'].update(width='all',conv_type=conv,pooling=pooling)
                    shared = build_shared_model(c)
                    for width in ('1','5','25'):
                        static = shared.freeze(width)
                        self.assertIsInstance(static.get_layer('conv_0'),tf.keras.layers.Conv2D)
                        self.assertFalse(any(isinstance(l,tf.keras.layers.BatchNormalization) for l in static.layers))
                        for frames in (98,148,198):
                            x = tf.random.normal((2,frames,40,1))
                            np.testing.assert_allclose(shared(x,width=width),static(x),atol=1e-5,rtol=1e-5)
                        rows = static.get_layer('logits').kernel.shape[0]
                        expected = max(1,int(12*{'1':1,'5':.5,'25':.25}[width]))
                        self.assertEqual(rows,expected*(2 if pooling=='mix' else 1))

    def test_legacy_dscnn_missing_pooling_is_identical_avg(self):
        c = config(); c['model'].pop('pooling')
        old = build_model(c)
        c['model']['pooling'] = 'avg'
        new = build_model(c); new.set_weights(old.get_weights())
        x = tf.random.normal((1,98,40,1))
        np.testing.assert_array_equal(old(x),new(x))
        self.assertEqual(old.name,new.name)

    def test_mobile_all_widths_pools_durations_static_equivalence(self):
        for pool in ('avg','max','mix'):
            for mels in (40,64,128):
                c = mobile_config(); c['model']['pooling'] = pool; c['audio']['n_mels'] = mels
                shared = build_shared_model(c)
                params = []
                for width in ('25','5','1'):
                    static = shared.freeze(width)
                    params.append(static.count_params())
                    self.assertEqual(static.layers[-1].name,'probabilities')
                    self.assertIsInstance(static.get_layer('stem_conv'),tf.keras.layers.Conv2D)
                    self.assertIsInstance(static.get_layer('block_0_residual'),tf.keras.layers.Add)
                    self.assertFalse(static.get_layer('block_0_project').use_bias)
                    self.assertEqual(static.get_layer('block_0_expand_relu6').max_value,6)
                    for duration,frames in ((1,98),(1.5,148),(2,198)):
                        x = tf.random.normal((2,frames,mels,1))
                        actual = shared(x,width=width,training=False)
                        self.assertEqual(tuple(actual.shape),(2,10))
                        np.testing.assert_allclose(actual.numpy().sum(axis=1),1.,atol=1e-6)
                        np.testing.assert_allclose(actual,static(x,training=False),atol=1e-5,rtol=1e-5)
                        fixed = fix_input_shape(static,c,duration)
                        self.assertEqual(fixed.input_shape,(1,frames,mels,1))
                        meta = architecture_metadata(fixed,c,width,duration)
                        self.assertGreater(meta['estimated_macs'],0)
                        self.assertEqual(meta['num_parameters'],static.count_params())
                self.assertTrue(params[0] < params[1] < params[2])

    def test_mobile_bn_isolation_and_shared_gradient_accumulation(self):
        c = mobile_config(); model = build_shared_model(c)
        x = tf.random.normal((2,98,40,1)); labels = tf.constant([2,5])
        states = model.stem_bn.states
        half_before = [v.numpy().copy() for v in states['5'].weights]
        quarter_before = [v.numpy().copy() for v in states['25'].weights]
        before_mean = states['1'].moving_mean.numpy().copy()
        model(x,width='1',training=True)
        self.assertFalse(np.array_equal(before_mean,states['1'].moving_mean.numpy()))
        for original,var in zip(half_before+quarter_before,states['5'].weights+states['25'].weights):
            np.testing.assert_array_equal(original,var.numpy())
        self.assertIsNot(states['5'].gamma,states['1'].gamma)
        reference = model.blocks[0].expansion_kernel
        with tf.GradientTape(persistent=True) as tape:
            total,losses,_ = width_losses(model,x,labels)
        gradient = tape.gradient(total,reference).numpy()
        contributions = [tape.gradient(losses[w],reference).numpy() for w in ('1','5','25')]
        np.testing.assert_allclose(gradient,sum(contributions),atol=1e-5,rtol=1e-5)
        quarter = contributions[-1]
        self.assertGreater(np.abs(quarter[:,:,:1,:3]).sum(),0)
        np.testing.assert_array_equal(quarter[:,:,1:,:],0)
        np.testing.assert_array_equal(quarter[:,:,:,3:],0)
        classifier = tape.gradient(losses['25'],model.classifier_kernel).numpy()
        # Mix must slice both avg and max prefixes, leaving the two tails unused.
        self.assertGreater(np.abs(classifier[:2]).sum(),0)
        self.assertGreater(np.abs(classifier[8:10]).sum(),0)
        np.testing.assert_array_equal(classifier[2:8],0)
        np.testing.assert_array_equal(classifier[10:],0)
        optimizer = tf.keras.optimizers.SGD(.01); optimizer.build(model.trainable_variables)
        original = reference.numpy().copy()
        shared_train_step(model,optimizer,x,labels,widths=('1','25'))
        shared_train_step(model,optimizer,tf.random.normal((2,148,40,1)),labels)
        self.assertIs(model.blocks[0].expansion_kernel,reference)
        self.assertEqual(int(optimizer.iterations),2)
        self.assertFalse(np.array_equal(original,reference.numpy()))
        for width in ('1','5','25'):
            np.testing.assert_allclose(model(x,width=width),model.freeze(width)(x),atol=1e-5,rtol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'shared.weights.h5'; model.save_weights(path)
            restored = build_shared_model(c); restored(x); restored.load_weights(path)
            for width in ('1','5','25'):
                np.testing.assert_array_equal(model(x,width=width),restored(x,width=width))

    def test_sandwich_sampling_subset_and_config_validation(self):
        c = mobile_config(); c['all_width_training']['strategy'] = 'sandwich'
        self.assertEqual(sample_widths(c,np.random.default_rng(4)),['1','25','5'])
        c['all_width_training']['sandwich_random_widths'] = 0
        self.assertEqual(sample_widths(c,np.random.default_rng(4)),['1','25'])
        c['model']['width_options'] = ['5','25']
        validate_config(c)
        model = build_shared_model(c)
        self.assertEqual(set(model.stem_bn.states),{'5','25'})
        self.assertEqual(model(tf.ones((1,98,40,1))).shape,(1,10))
        for key,value in (('pooling','invalid'),('expansion_ratio',0),('width_options',['1','1'])):
            bad = copy.deepcopy(c); bad['model'][key] = value
            with self.assertRaises(ValueError): validate_config(bad)

    def test_bn_recalibration_is_train_only_deterministic_and_affine_unchanged(self):
        from quantization.recalibrate_bn import recalibrate_batch_norm
        from training.dataset import SyntheticDataset
        c = mobile_config(); c['model']['width'] = '5'; c['augmentation']['enabled'] = True
        source = SyntheticDataset(c,'train',4)
        model = build_model(c)
        before = [v.numpy().copy() for v in model.trainable_variables]
        report = recalibrate_batch_norm(model,source,c,3)
        self.assertTrue(report['applied']); self.assertFalse(report['augmentation'])
        self.assertEqual(report['durations'],[1,1.5,2])
        for original,v in zip(before,model.trainable_variables): np.testing.assert_array_equal(original,v.numpy())
        first = [v.numpy().copy() for v in model.non_trainable_variables]
        recalibrate_batch_norm(model,source,c,3)
        for original,v in zip(first,model.non_trainable_variables): np.testing.assert_array_equal(original,v.numpy())
        with self.assertRaises(ValueError):
            recalibrate_batch_norm(model,SyntheticDataset(c,'val',4),c,3)
