"""Validation protocols are deterministic host preprocessing, independent of training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from tests.common import config
from tests.test_models_extended import mobile_config
from training.augment import evaluation_crops
from training.dataset import SyntheticDataset, BatchDataset
from training.features import extract_features
from training.models import build_shared_model
from training.train import validate_epoch
from training.utils import validation_protocol, validate_config, load_config
from evaluation.metrics import classification_metrics
from evaluation.validation import window_starts, sliding_windows, ValidationData, evaluate_configurations


class WaveDataset(BatchDataset):
    def __init__(self, c, lengths=(64000, 10000), split='val'):
        self._setup(c, split)
        self.count, self.calls = len(lengths), 0
        self.waves = [np.random.default_rng(i).normal(0,.1,n).astype(np.float32) for i,n in enumerate(lengths)]

    def example(self, index):
        self.calls += 1
        return self.waves[index], index, f'clip-{index}'


class ValidationTests(unittest.TestCase):
    def test_integer_window_counts_end_inclusion_dedup_and_rounding(self):
        self.assertEqual(window_starts(40,10,5),list(range(0,31,5)))
        self.assertEqual(window_starts(41,10,6),[0,6,12,18,24,30,31])
        self.assertEqual(window_starts(41,10,6,False),[0,6,12,18,24,30])
        self.assertEqual(window_starts(10,10,5),[0])
        for duration,count in ((1,7),(1.5,6),(2,5)):
            starts = window_starts(64000,round(duration*16000),8000)
            self.assertEqual(len(starts),count)
            self.assertEqual(starts[-1]+round(duration*16000),64000)
            self.assertEqual(len(set(starts)),len(starts))
        starts = window_starts(64000,16000,round(.3333333*16000))
        self.assertEqual(starts[:3],[0,5333,10666])
        with self.assertRaises(ValueError): window_starts(4,2,0)
        with self.assertRaises(ValueError): window_starts(4,2,.5)

    def test_short_and_empty_audio_right_zero_padding(self):
        np.testing.assert_array_equal(list(sliding_windows(np.array([1,2],np.float32),5,2)),[[1,2,0,0,0]])
        np.testing.assert_array_equal(list(sliding_windows(np.array([],np.float32),5,2)),[[0,0,0,0,0]])
        # The old fixed protocol still uses centered padding, exactly as before.
        np.testing.assert_array_equal(evaluation_crops(np.array([1,2],np.float32),5,3)[0],[0,1,2,0,0])

    def test_fixed_batches_crop_positions_features_and_metrics_are_unchanged(self):
        c = config(); c['evaluation']['num_crops'] = 3
        # All of these settings must be ignored by fixed validation.
        c['validation'].update(stride_seconds=-1,aggregation='unused',batch_size=0,short_clip_padding='unused')
        validate_config(c)
        source = WaveDataset(c)
        for duration in (1,1.5,2):
            with ValidationData(source,c) as prepared:
                actual = list(prepared.batches(duration))
            old = list(source.batches(training=False,duration=duration))
            for (x,y,ids),(a,b,files) in zip(actual,old):
                np.testing.assert_array_equal(x,a); np.testing.assert_array_equal(y,b); self.assertEqual(ids,files)
            expected = np.stack([extract_features(crop,c) for wave in source.waves
                                 for crop in evaluation_crops(wave,round(16000*duration),3)])
            np.testing.assert_array_equal(np.concatenate([b[0] for b in actual]),expected)
            with ValidationData(source,c) as prepared:
                predict = lambda x: np.tile([.2,.8],(len(x),1))
                result,_ = evaluate_configurations({'model':predict},prepared,{0:'a',1:'b'},duration,from_logits=False)
            labels = np.concatenate([b[1] for b in old]).tolist()
            ids = sum([b[2] for b in old],[])
            legacy = classification_metrics(labels,predict(expected),ids,{0:'a',1:'b'},'val',from_logits=False)
            for key in legacy: self.assertEqual(result['model'][key],legacy[key])

    def test_sliding_cache_and_batch_size_preserve_inputs_without_augmentation(self):
        c = config(); c['validation'].update(method='sliding_window',batch_size=3)
        c['augmentation']['enabled'] = True
        source = WaveDataset(c)
        with patch('training.augment.Compose.__call__',side_effect=AssertionError('augmentation leaked')):
            with ValidationData(source,c) as prepared, patch('evaluation.validation.extract_features',wraps=extract_features) as extract:
                snapshots = {d:list(prepared.batches(d)) for d in (1,1.5,2)}
                calls = extract.call_count
                for d in (1,1.5,2):
                    repeated = list(prepared.batches(d))
                    np.testing.assert_array_equal(np.concatenate([b[0] for b in snapshots[d]]),np.concatenate([b[0] for b in repeated]))
                self.assertEqual(extract.call_count,calls)
                self.assertEqual(source.calls,2) # Decoded once, across all durations/epochs.
                cache_dir = prepared.cache_dir
            self.assertFalse(cache_dir.exists())
            plain = copy.deepcopy(c); plain['validation'].update(batch_size=1,cache_features=False)
            with ValidationData(source,plain) as uncached:
                self.assertIsNone(uncached.cache_dir)
                for d in (1,1.5,2):
                    actual = list(uncached.batches(d))
                    np.testing.assert_array_equal(np.concatenate([b[0] for b in snapshots[d]]),np.concatenate([b[0] for b in actual]))
        self.assertEqual(len(sum([b[2] for b in snapshots[1]],[])),8) # 7 long + 1 short.

    def test_aggregation_is_one_prediction_per_clip_with_macro_f1(self):
        values = np.array([[.9,.1,0],[.9,.1,0],[.01,.99,0],[.2,.8,0]])
        ids, labels, classes = ['a','a','a','b'],[0,0,0,1],{0:'a',1:'b',2:'absent'}
        mean = classification_metrics(labels,values,ids,classes,'val',from_logits=False)
        maximum = classification_metrics(labels,values,ids,classes,'val',from_logits=False,aggregation='max_probability')
        self.assertEqual(mean['num_original_files'],2)
        self.assertEqual(mean['accuracy'],1)
        self.assertAlmostEqual(mean['macro_f1'],2/3)
        self.assertEqual(maximum['accuracy'],.5)
        self.assertAlmostEqual(maximum['macro_f1'],2/9)
        np.testing.assert_allclose(classification_metrics(labels,np.log(np.maximum(values,1e-20)),ids,classes,'val')['accuracy'],mean['accuracy'])

    def test_training_and_calibration_inputs_are_independent_of_validation_method(self):
        c = config(); c['augmentation']['enabled'] = True
        other = copy.deepcopy(c); other['validation']['method'] = 'sliding_window'
        for training in (True,False):
            a,b = SyntheticDataset(c,'train',2),SyntheticDataset(other,'train',2)
            first = next(a.batches(training=True))[0] if training else next(a.representative(1))
            second = next(b.batches(training=True))[0] if training else next(b.representative(1))
            np.testing.assert_array_equal(first,second)

    def test_both_architectures_all_width_duration_metrics_and_fresh_predictions(self):
        for architecture in ('dscnn','mobilenetv2'):
            for method in ('fixed','sliding_window'):
                c = mobile_config(); c['model']['architecture'] = architecture
                c['validation'].update(method=method,batch_size=2)
                model = build_shared_model(c)
                source = SyntheticDataset(c,'val',2)
                model(tf.zeros((1,98,40,1)))
                original = [v.numpy().copy() for v in model.weights]
                with ValidationData(source,c) as prepared:
                    row = validate_epoch(model,source,c,['1','5','25'],{i:str(i) for i in range(10)},True,prepared)
                    expected = [row[f'val_accuracy_{w}_{d}s'] for w in ('1','5','25') for d in ('1','1.5','2')]
                    self.assertAlmostEqual(row['val_accuracy'],np.mean(expected))
                    self.assertEqual(row['val_num_clips'],2)
                    self.assertEqual(row['val_num_windows'],18 if method=='sliding_window' else 6)
                    self.assertEqual(row['val_num_window_predictions'],3*row['val_num_windows'])
                    self.assertGreater(row['validation_seconds'],0)
                    for a,v in zip(original,model.weights): np.testing.assert_array_equal(a,v.numpy())
                    # Reuse features but invoke the new predictor, never prior outputs.
                    calls = []
                    def predict(x):
                        calls.append(len(x)); return np.tile(np.eye(10)[1],(len(x),1))
                    result,_ = evaluate_configurations({'changed':predict},prepared,{i:str(i) for i in range(10)},1,from_logits=False)
                    self.assertGreater(len(calls),0); self.assertEqual(result['changed']['accuracy'],.5)

    def test_invalid_sliding_config_and_legacy_missing_section(self):
        c = config(); c['validation']['method'] = 'sliding_window'
        for key,value in (('method','other'),('stride_seconds',0),('stride_seconds',1e-9),('stride_seconds',float('nan')),
                          ('aggregation','mean_logits'),('batch_size',0),('include_last_window',1),('short_clip_padding','reflect')):
            bad=copy.deepcopy(c); bad['validation'][key]=value
            with self.assertRaises(ValueError): validate_config(bad)
        c.pop('validation')
        self.assertEqual(validation_protocol(c)['method'],'fixed')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'old.json'; p.write_text(json.dumps(c))
            self.assertEqual(load_config(p)['validation']['method'],'fixed')
            self.assertEqual(load_config(p,['validation.method=sliding_window'])['validation']['method'],'sliding_window')

    def test_completed_fingerprint_distinguishes_protocol_parameters(self):
        from training.registry import experiment_key,find_completed,create_run
        from training.artifacts import update_report
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            run = create_run(c); p=run/'weights.bin'; p.write_bytes(b'test fixture')
            import hashlib
            update_report(run,state={'status':'complete'},configuration_hash=experiment_key(c),
                          file_hashes={'weights.bin':hashlib.sha256(p.read_bytes()).hexdigest()})
            self.assertEqual(find_completed(c),run)
            slide = copy.deepcopy(c); slide['validation']['method']='sliding_window'
            self.assertIsNone(find_completed(slide))
            baseline = experiment_key(slide)
            for key,value in (('stride_seconds',.25),('aggregation','max_probability'),('include_last_window',False),('batch_size',1)):
                changed=copy.deepcopy(slide); changed['validation'][key]=value
                self.assertNotEqual(experiment_key(changed),baseline)

    def test_checkpoint_uses_equal_configuration_clip_accuracy_not_window_pooling(self):
        from types import SimpleNamespace
        from training.control import training_callbacks
        c = mobile_config()
        c['validation']['method']='sliding_window'
        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__(); self.stage=self.add_weight('stage',shape=(),initializer='zeros')
            def call(self,x,width='1',training=False):
                correct = (float(x[0,0,0,0]) != 1) if self.stage.numpy()==0 else (float(x[0,0,0,0]) == 1)
                return tf.constant([[1.,0.] if correct else [0.,1.]]*len(x))
        def batches(duration):
            n = 100 if duration==1 else 1
            yield np.full((n,1,1,1),duration,np.float32),np.zeros(n,np.int32),['one-clip']*n
        prepared=SimpleNamespace(dataset=SimpleNamespace(split='val'),protocol=validation_protocol(c),batches=batches)
        model=Model(); model(tf.ones((1,1,1,1)))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'best.weights.h5'
            callbacks,early=training_callbacks(model,tf.keras.optimizers.SGD(.01),c['training'],checkpoint)
            scores=[]
            for epoch in range(2):
                model.stage.assign(float(epoch))
                row=validate_epoch(model,prepared.dataset,c,['1','5','25'],{0:'a',1:'b'},True,prepared)
                scores.append(row['val_accuracy'])
                for callback in callbacks: callback.on_epoch_end(epoch,row)
            np.testing.assert_allclose(scores,[2/3,1/3])
            self.assertEqual(early.best_epoch,0)
            self.assertEqual(early.wait,1)
            model.load_weights(checkpoint)
            self.assertEqual(float(model.stage.numpy()),0.)
