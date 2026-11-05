import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import yaml

from training.artifacts import read_report, update_report
from training.registry import create_run, experiment_key
from training.sweep import experiment_changes, run_sweep
from training.utils import load_config, set_override, validate_config


def global_allocate(args):
    config, index = args
    config['model']['architecture'] = 'dscnn' if index % 2 == 0 else 'mobilenetv2'
    return create_run(config, sequence_start=7).name


def preset_space():
    return yaml.safe_load(Path('configs/baseline_queue.yaml').read_text())


class BaselineQueueTests(unittest.TestCase):
    def test_seven_distinct_presets_match_baseline_and_frequency_masks(self):
        space = preset_space()
        expected = [('dscnn','SC',128,8),('dscnn','DW',64,4),('dscnn','DW',128,8),
                    ('mobilenetv2','SC',64,4),('mobilenetv2','SC',128,8),
                    ('mobilenetv2','DW',64,4),('mobilenetv2','DW',128,8)]
        base = load_config(overrides=[f'{k}={json.dumps(v)}' for k,v in space['defaults'].items()])
        keys = []
        for (_,changes), (architecture,conv,mels,mask) in zip(experiment_changes(space),expected):
            c=copy.deepcopy(base)
            for change in changes: set_override(c,change)
            validate_config(c)
            self.assertEqual(c['model']['architecture'],architecture)
            self.assertEqual(c['model']['conv_type'],conv)
            if architecture=='mobilenetv2': self.assertEqual(c['model']['mobile_spatial_conv'],conv)
            self.assertEqual(c['audio']['n_mels'],mels)
            self.assertEqual(c['augmentation']['spectrogram']['FrequencyMask']['max_bins'],mask)
            self.assertTrue(c['augmentation']['enabled'])
            self.assertEqual(c['training']['duration_secs'],[1,2])
            self.assertEqual(c['training']['batch_size'],64)
            self.assertEqual(c['training']['epochs'],50)
            self.assertEqual(c['training']['seed'],42)
            self.assertEqual(c['model']['width'],'all')
            self.assertEqual(c['model']['pooling'],'max')
            self.assertEqual(c['validation']['method'],'fixed')
            self.assertEqual(c['quantization']['bn_recalibration_samples'],0)
            self.assertEqual(c['runtime']['device'],'auto')
            self.assertTrue(c['runtime']['deterministic'])
            keys.append(experiment_key(c))
        self.assertEqual(len(keys),7); self.assertEqual(len(set(keys)),7)

    def test_preview_does_not_train_or_allocate_and_cli_overrides_win_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'models'
            results=run_sweep('config.yaml',preset_space(),[f'runtime.output_dir={output}','training.epochs=2'],
                              preview=True,runner=lambda *a,**k:self.fail('preview started training'))
            self.assertEqual(len(results),7)
            self.assertTrue(all(row['status']=='planned' and row['epochs']==2 for row in results))
            self.assertFalse(output.exists())

    def test_global_sequence_is_unique_concurrent_and_never_reused(self):
        c=load_config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir']=tmp
            with multiprocessing.get_context('spawn').Pool(3) as pool:
                names=pool.map(global_allocate,[(c,i) for i in range(6)])
            self.assertEqual(sorted(int(n.rsplit('_',1)[1]) for n in names),list(range(7,13)))
            last=next(n for n in names if n.endswith('_012'))
            shutil.rmtree(Path(tmp)/last)
            c['model']['architecture']='mobilenetv2'
            next_run=create_run(c,sequence_start=7)
            self.assertEqual(next_run.name,'mobilenetv2_013')
            self.assertEqual(read_report(next_run)['experiment']['sequence_policy'],'global')
            with self.assertRaises(ValueError): create_run(c,sequence_start=0)

    def test_ordered_queue_continues_failures_and_skips_completed_without_reusing_ids(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            overrides=[f'runtime.output_dir={tmp}']
            def runner(command,**kwargs):
                calls.append(command)
                candidate=load_config('config.yaml',[command[i+1] for i,v in enumerate(command) if v=='--set'])
                label=command[command.index('--run-id')+1]
                start=int(command[command.index('--sequence-start')+1])
                run=create_run(candidate,label,sequence_start=start)
                failed=len(calls)==2
                p=run/'fixture.weights'; p.write_bytes(b'unit fixture, not a trained model')
                update_report(run,state={'status':'failed' if failed else 'complete'},
                              configuration_hash=experiment_key(candidate),
                              file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()})
                return subprocess.CompletedProcess(command,1 if failed else 0)
            first=run_sweep('config.yaml',preset_space(),overrides,runner=runner)
            self.assertEqual([r['experiment_id'] for r in first],['dscnn_007','dscnn_008','dscnn_009',
                'mobilenetv2_010','mobilenetv2_011','mobilenetv2_012','mobilenetv2_013'])
            self.assertEqual([r['status'] for r in first],['complete','failed',*['complete']*5])
            second=run_sweep('config.yaml',preset_space(),overrides,runner=runner,skip_completed=True)
            self.assertEqual(len(calls),8)
            self.assertEqual(second[1]['experiment_id'],'dscnn_014')
            self.assertEqual(second[0]['status'],'skipped')
            self.assertEqual(read_report(Path(tmp)/'dscnn_008')['state']['status'],'failed')

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

    def test_mobile_sc_128_shared_mixed_cpu_training_and_fp32_int8_export(self):
        from tests.test_models_extended import mobile_config
        from training.train import train
        from quantization.validate_int8 import TFLiteRunner
        c=mobile_config(); c['model'].update(mobile_spatial_conv='SC',pooling='max')
        c['audio']['n_mels']=128
        c['training'].update(duration_secs=[1,2],batch_size=2)
        c['quantization'].update(representative_samples=2,output_duration_secs=[])
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir']=tmp
            run=train(c,dry_run=True,limit=4,sequence_start=7)
            self.assertEqual(run.name,'mobilenetv2_007')
            report=read_report(run)
            self.assertEqual(report['state']['status'],'complete')
            self.assertEqual(report['model']['spatial_conv_type'],'SC')
            self.assertEqual(len(list(run.rglob('*.tflite'))),12)
            self.assertEqual(len(list(run.rglob('*.json'))),1)
            for relative in report['artifacts']:
                child=read_report(run/relative)
                for tag,frames in (('1s',98),('2s',198)):
                    for kind in ('fp32','int8'):
                        exported=child['exports'][tag][kind]
                        runner=TFLiteRunner(run/relative/exported['filename'],require_int8=kind=='int8')
                        self.assertEqual(runner(np.zeros((1,frames,128,1))).shape,(1,10))
                        self.assertNotIn('DEPTHWISE_CONV_2D',exported['micro_compatibility']['operators'])
                        self.assertEqual(exported['architecture']['spatial_conv_type'],'SC')

    def test_dscnn_128_fp32_and_int8_can_invoke(self):
        from tests.common import config
        from training.models import build_model,fix_input_shape
        from quantization.convert_int8 import convert_model
        from quantization.validate_int8 import TFLiteRunner
        c=config(); c['audio']['n_mels']=128
        model=fix_input_shape(build_model(c),c,1)
        example=np.zeros((1,98,128,1),np.float32)
        for representative in (None,[example,np.ones_like(example)]):
            runner=TFLiteRunner(model_content=convert_model(model,representative),require_int8=representative is not None)
            self.assertEqual(runner(example).shape,(1,10))
