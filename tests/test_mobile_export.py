import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from tests.test_models_extended import mobile_config
from training.artifacts import read_report, config_path
from training.train import train
from quantization.validate_int8 import TFLiteRunner


class MobileExportTests(unittest.TestCase):
    def test_shared_mixed_training_exports_reports_and_recalibration(self):
        c = mobile_config()
        c['training']['batch_size'] = 2
        c['validation'].update(method='sliding_window', batch_size=2)
        c['all_width_training'].update(strategy='sandwich',sandwich_random_widths=1)
        c['quantization'].update(representative_samples=3,bn_recalibration_samples=3)
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            run = train(c,dry_run=True,limit=6)
            self.assertEqual(run.name,'mobilenetv2_001')
            report = read_report(run)
            self.assertEqual(report['state']['status'],'complete')
            self.assertEqual(report['model']['architecture'],'mobilenetv2')
            self.assertEqual(report['model']['pooling'],'mix')
            self.assertEqual(report['model']['expansion_ratio'],3)
            self.assertEqual(report['model']['width_options'],[1.,.5,.25])
            self.assertTrue(report['model']['batch_norm']['switchable'])
            self.assertEqual(len(report['subnets']),3)
            self.assertEqual(list(run.rglob('*.json')),[run/'report.json'])
            self.assertEqual(list(run.rglob('*.csv')),[run/'history.csv'])
            self.assertNotIn('history',report)
            self.assertLess((run/'report.json').stat().st_size,200000)
            self.assertEqual(report['checkpoint']['monitor'],'val_accuracy')
            self.assertEqual(report['checkpoint']['best_epoch'],1)
            self.assertGreater(report['timing']['training_elapsed_seconds'],0)
            self.assertEqual(len(report['timing']['epoch_elapsed_seconds']),1)
            self.assertIsNone(report['test'])
            self.assertEqual(report['validation']['protocol']['method'], 'sliding_window')
            self.assertEqual(report['validation']['num_clips'], 6)
            self.assertEqual(report['validation']['num_windows'], 6 * (4 + 3 + 2))
            self.assertEqual(report['validation']['num_window_predictions'], 3 * 6 * (4 + 3 + 2))
            self.assertGreater(report['validation']['runtime_seconds'], 0)
            by_config = report['validation']['by_configuration']
            self.assertAlmostEqual(report['checkpoint']['best_value'],np.mean([m['accuracy'] for d in by_config.values() for m in d.values()]))
            with (run/'history.csv').open() as stream: row = next(csv.DictReader(stream))
            for d in ('1','1.5','2'): self.assertGreater(int(row[f'train_samples_{d}s']),0)
            with (Path(tmp)/'summary.csv').open() as stream: summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary),1); self.assertEqual(summary[0]['pooling'],'mix')
            self.assertEqual(float(summary[0]['val_accuracy']),report['validation']['accuracy'])
            for relative in report['artifacts']:
                path = run/relative; child = read_report(path)
                self.assertEqual(config_path(path),run/'report.json')
                self.assertTrue(child['bn_recalibration']['applied'])
                self.assertTrue(child['metadata']['equivalence_checked_before_bn_recalibration'])
                model = tf.keras.models.load_model(path/'best.keras',compile=False)
                for d,frames in (('1',98),('1.5',148),('2',198)):
                    for kind in ('fp32','int8'):
                        exported = child['exports'][f'{d}s'][kind]
                        file = path/exported['filename']
                        self.assertEqual(file.stat().st_size,exported['byte_size'])
                        self.assertEqual(exported['sha256'],child['file_hashes'][str(file.relative_to(path))])
                        self.assertGreater(exported['architecture']['num_parameters'],0)
                        self.assertGreater(exported['architecture']['estimated_macs'],0)
                        audit = exported['micro_compatibility']
                        self.assertEqual(audit['status'],'static_checks_passed')
                        self.assertIn('AddConcatenation',audit['required_resolver_calls'])
                        self.assertFalse(audit['firmware_invocation_verified'])
                        self.assertIsNone(audit['tensor_arena_bytes'])
                        runner = TFLiteRunner(file,require_int8=kind=='int8')
                        self.assertEqual(runner.input['shape_signature'].tolist(),[1,frames,40,1])
                        probe = np.zeros((1,frames,40,1),np.float32)
                        scores = runner(probe)
                        self.assertEqual(scores.shape,(1,10))
                        if kind=='fp32': np.testing.assert_allclose(scores,model(probe),atol=1e-5,rtol=1e-4)
                    evaluation = child['evaluation'][f'{d}s']['val']
                    self.assertIn('int8_minus_fp32_accuracy',evaluation['comparison'])
                    self.assertFalse(evaluation['int8_tflite']['empirical_dataset_result'])
            json.loads((run/'report.json').read_text(),parse_constant=lambda v:self.fail(v))
            self.assertEqual(train(c,dry_run=True,limit=6,skip_completed=True),run)
