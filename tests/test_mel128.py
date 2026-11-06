"""128-Mel models work through the normal training and export APIs."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from training.artifacts import read_report


class Mel128Tests(unittest.TestCase):
    def test_mobile_sc_shared_prefix_gradients_freeze_and_legacy_dw(self):
        import tensorflow as tf
        from tests.test_models_extended import mobile_config
        from training.models import build_shared_model, build_model, architecture_metadata
        from training.losses import width_losses, shared_train_step
        from training.utils import canonical_name
        c=mobile_config(); c['audio']['n_mels']=128
        c['model'].update(mobile_spatial_conv='SC',pooling='max')
        model=build_shared_model(c)
        x=tf.random.normal((2,98,128,1)); labels=tf.constant([0,1])
        kernel=model.blocks[0].spatial_kernel
        with tf.GradientTape(persistent=True) as tape:
            total,losses,_=width_losses(model,x,labels)
        gradients=[tape.gradient(losses[w],kernel).numpy() for w in ('1','5','25')]
        np.testing.assert_allclose(tape.gradient(total,kernel),sum(gradients),atol=1e-5,rtol=1e-5)
        np.testing.assert_array_equal(gradients[-1][:,:,3:,:],0)
        np.testing.assert_array_equal(gradients[-1][:,:,:,3:],0)
        before=kernel.numpy().copy()
        shared_train_step(model,tf.keras.optimizers.SGD(.01),x,labels)
        self.assertFalse(np.array_equal(before,kernel.numpy()))
        for w in ('1','5','25'):
            frozen=model.freeze(w)
            self.assertIsInstance(frozen.get_layer('block_0_spatial'),tf.keras.layers.Conv2D)
            self.assertFalse(any(isinstance(l,tf.keras.layers.DepthwiseConv2D) for l in frozen.layers))
            for frames in (98,148,198):
                probe=tf.random.normal((1,frames,128,1))
                np.testing.assert_allclose(model(probe,width=w),frozen(probe),atol=1e-5,rtol=1e-5)
        sc_macs=architecture_metadata(model.freeze('1'),c,'1',1)['estimated_macs']
        legacy=copy.deepcopy(c); legacy['model'].pop('mobile_spatial_conv'); legacy['model']['conv_type']='SC'
        original=build_model(legacy,'1')
        self.assertIsInstance(original.get_layer('block_0_depthwise'),tf.keras.layers.DepthwiseConv2D)
        self.assertEqual(canonical_name(legacy,'1'),'MobileNetV2Tiny_128_1')
        self.assertEqual(canonical_name(c,'1'),'MobileNetV2Tiny_128_SC_1')
        self.assertGreater(sc_macs,architecture_metadata(original,legacy,'1',1)['estimated_macs'])
        with tempfile.TemporaryDirectory() as tmp:
            model.save_weights(Path(tmp)/'shared.weights.h5')
            restored=build_shared_model(c); restored(x); restored.load_weights(Path(tmp)/'shared.weights.h5')
            for w in ('1','5','25'):
                np.testing.assert_array_equal(model(x,width=w),restored(x,width=w))

    def test_mobile_128_shared_mixed_cpu_training_and_fp32_int8_export(self):
        from tests.test_models_extended import mobile_config
        from training.train import train
        from quantization.validate_int8 import TFLiteRunner
        for conv in ('SC','DW'):
            c=mobile_config(); c['model'].update(mobile_spatial_conv=conv,pooling='max')
            c['audio']['n_mels']=128
            c['training'].update(duration_secs=[1,2],batch_size=2)
            c['quantization'].update(representative_samples=2,output_duration_secs=[])
            with tempfile.TemporaryDirectory() as tmp:
                c['runtime']['output_dir']=tmp
                run=train(c,dry_run=True,limit=4)
                self.assertEqual(run.name,'mobilenetv2_001')
                report=read_report(run)
                self.assertEqual(report['state']['status'],'complete')
                self.assertEqual(report['model']['spatial_conv_type'],conv)
                self.assertEqual(len(list(run.rglob('*.tflite'))),12)
                self.assertEqual(len(list(run.rglob('*.json'))),1)
                for relative in report['artifacts']:
                    child=read_report(run/relative)
                    for tag,frames in (('1s',98),('2s',198)):
                        for kind in ('fp32','int8'):
                            exported=child['exports'][tag][kind]
                            runner=TFLiteRunner(run/relative/exported['filename'],require_int8=kind=='int8')
                            self.assertEqual(runner(np.zeros((1,frames,128,1))).shape,(1,10))
                            if conv == 'SC':
                                self.assertNotIn('DEPTHWISE_CONV_2D',exported['micro_compatibility']['operators'])
                            else:
                                self.assertIn('DEPTHWISE_CONV_2D',exported['micro_compatibility']['operators'])
                            self.assertEqual(exported['architecture']['spatial_conv_type'],conv)


    def test_dscnn_128_fp32_and_int8_can_invoke(self):
        from tests.common import config
        from training.models import build_model,fix_input_shape
        from quantization.convert_int8 import convert_model
        from quantization.validate_int8 import TFLiteRunner
        for conv in ('SC','DW'):
            c=config(); c['audio']['n_mels']=128
            c['model']['conv_type'] = conv
            model=fix_input_shape(build_model(c),c,1)
            example=np.zeros((1,98,128,1),np.float32)
            for representative in (None,[example,np.ones_like(example)]):
                runner=TFLiteRunner(model_content=convert_model(model,representative),require_int8=representative is not None)
                self.assertEqual(runner(example).shape,(1,10))

    def test_mel_config_normalization_and_invalid_values(self):
        from training.utils import load_config, SUPPORTED_MEL_BINS
        self.assertEqual(SUPPORTED_MEL_BINS,(40,64,128))
        for value in (128,128.0,'128',' 128 ','128.0'):
            c=load_config(overrides=[f'audio.n_mels={json.dumps(value)}'])
            self.assertEqual(c['audio']['n_mels'],128)
            self.assertIs(type(c['audio']['n_mels']),int)
        for value in (True,128.5,'unsupported',80,float('nan'),'nan'):
            with self.assertRaisesRegex(ValueError,'audio.n_mels'):
                load_config(overrides=[f'audio.n_mels={json.dumps(value)}'])

    def test_single_and_cartesian_cli_checks_do_not_allocate_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'models'
            command=[sys.executable,'-m','training.train','--check-config',
                     '--set','audio.n_mels="128"','--set','model.architecture=mobilenetv2',
                     '--set','model.mobile_spatial_conv=DW','--set','training.duration_secs=[1,1.5,2]',
                     '--set',f'runtime.output_dir={output}']
            checked=subprocess.run(command,capture_output=True,text=True,check=True)
            summary=json.loads(checked.stdout)
            self.assertEqual(summary['n_mels'],128)
            self.assertEqual(summary['feature_shapes'],{'1s':[98,128,1],'1.5s':[148,128,1],'2s':[198,128,1]})
            self.assertEqual(summary['spatial_conv'],'DW')
            self.assertFalse(output.exists())
            space=Path(tmp)/'space.json'
            space.write_text(json.dumps({'parameters':{'audio.n_mels':[64,'128'],
                              'model.architecture':['dscnn','mobilenetv2']}}))
            checked=subprocess.run([sys.executable,'-m','training.sweep','--check-config','--space',str(space),
                                    '--set',f'runtime.output_dir={output}'],capture_output=True,text=True,check=True)
            summary=json.loads(checked.stdout)
            self.assertEqual(summary['configuration_count'],4)
            self.assertEqual(summary['mel_bins'],[64,128])
            self.assertEqual(summary['architectures'],['dscnn','mobilenetv2'])
            self.assertFalse(output.exists())
