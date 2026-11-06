import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.common import config
from training.train import train
from training.utils import read_json
from training.artifacts import read_report, validation_metrics
from quantization.validate_int8 import TFLiteRunner


class QuantizationIntegrationTests(unittest.TestCase):
    def test_waveform_smoke_epoch_checkpoint_exports_and_int8_invoke(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            c['training']['batch_size'] = 2
            c['training']['duration_secs'] = [1, 1.5, 2]
            c['quantization']['output_duration_secs'] = [1, 1.5, 2]
            c['quantization']['representative_samples'] = 4
            run = train(c, dry_run=True, limit=6, run_id='integration')
            report = read_report(run)
            status = report['state']
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['epochs_executed'], 1)
            self.assertFalse(status['held_out_test_evaluated'])
            self.assertTrue((run / 'best.keras').exists())
            self.assertTrue((Path(tmp) / 'summary.csv').exists())
            self.assertFalse((Path(tmp) / 'model_registry.csv').exists())
            self.assertFalse((run / 'config.yaml').exists())
            self.assertFalse((run / 'legacy_config.yaml').exists())
            runner = TFLiteRunner(run / 'exports/1s/int8.tflite', require_int8=True)
            self.assertEqual(runner(np.zeros([1, 98, 40, 1])).shape, (1, 10))
            details = report['exports']['1s']['int8']
            self.assertEqual(details['input']['dtype'], 'int8')
            self.assertEqual(details['output']['dtype'], 'int8')
            self.assertTrue(details['cpu_invocation_success'])
            metrics = validation_metrics(run)
            for fmt in ('keras_fp32', 'fp32_tflite', 'int8_tflite'):
                self.assertFalse(metrics[fmt]['empirical_dataset_result'])
                self.assertEqual(metrics[fmt]['data_source'], 'synthetic_smoke_only')
            from training.dataset import SyntheticDataset
            from evaluation.evaluate import evaluate_artifact
            import copy
            import tensorflow as tf
            model = tf.keras.models.load_model(run / 'best.keras', compile=False)
            probe = np.random.default_rng(1).normal(size=(1, 98, 40, 1)).astype(np.float32)
            fp32 = TFLiteRunner(run / 'exports/1s/fp32.tflite')
            np.testing.assert_allclose(fp32(probe), model(probe).numpy(), rtol=1e-4, atol=1e-5)
            runner2 = TFLiteRunner(run / 'exports/2s/int8.tflite', require_int8=True)
            self.assertEqual(runner2(np.zeros([1, 198, 40, 1])).shape, (1, 10))
            self.assertEqual(details['frozen_weight_sha256'], report['exports']['2s']['int8']['frozen_weight_sha256'])
            self.assertEqual(list(run.rglob('*.json')), [run / 'report.json'])
            self.assertLess((run / 'report.json').stat().st_size, 60000)
            self.assertNotIn('history', report)
            self.assertNotIn('confusion_matrix', metrics['keras_fp32'])
            import csv
            with (run / 'history.csv').open() as stream:
                history = list(csv.DictReader(stream))
            self.assertGreater(int(history[0]['train_samples_1s']), 0)
            self.assertGreater(int(history[0]['train_samples_2s']), 0)
            self.assertGreater(int(history[0]['train_samples_1.5s']), 0)
            self.assertIn('val_accuracy_1s', history[0])
            self.assertIn('val_accuracy_2s', history[0])
            self.assertEqual(int(history[0]['patience_remaining']), c['training']['early_stopping_patience'])
            self.assertAlmostEqual(float(history[0]['val_accuracy']), np.mean([float(history[0][f'val_accuracy_{d}s']) for d in ('1', '1.5', '2')]))
            self.assertEqual(TFLiteRunner(run / 'exports/1.5s/int8.tflite', require_int8=True)(np.zeros([1,148,40,1])).shape, (1,10))
            self.assertEqual(report['checkpoint']['monitor'], 'val_accuracy')
            self.assertEqual(report['checkpoint']['best_epoch'], 1)
            self.assertIsNone(report['test'])
            self.assertEqual(train(c, dry_run=True, limit=6, skip_completed=True), run)
            with (Path(tmp) / 'summary.csv').open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(float(summary[0]['val_accuracy']), report['validation']['accuracy'])
            self.assertEqual(len(report['file_hashes']), 8)
            c2 = copy.deepcopy(c)
            c2['audio']['duration_sec'] = 2
            evaluation = evaluate_artifact(run, SyntheticDataset(c2, 'val', 4), c2,
                                           {i: f'synthetic_{i}' for i in range(10)}, duration=2, synthetic=True)
            self.assertEqual(evaluation['int8_tflite']['evaluation_duration_sec'], 2)
            self.assertEqual(evaluation['int8_tflite']['trained_duration_sec'], 1)
            from training.registry import upsert_registry, register_artifact
            upsert_registry(tmp, {'artifact_id':'compatibility-test'})
            register_artifact(run)
            with (Path(tmp) / 'model_registry.csv').open() as stream:
                records = list(csv.DictReader(stream))
            config_paths = [Path(row['config_path']) for row in records if row['run_id']]
            self.assertEqual(len(config_paths), 3)
            self.assertTrue(all(path.name == 'legacy_config.yaml' for path in config_paths))
            from training.utils import load_config
            self.assertEqual(load_config(config_paths[0])['model']['width'], '5')
            # A different host evaluation protocol preserves the trained report,
            # its original metrics, exports and default CREST/registry meaning.
            previous = read_report(run)
            sliding = copy.deepcopy(c)
            sliding['validation'].update(method='sliding_window', aggregation='max_probability', batch_size=2)
            changed = evaluate_artifact(run, SyntheticDataset(sliding, 'val', 6), sliding,
                                       {i: f'synthetic_{i}' for i in range(10)}, duration=1, synthetic=True)
            current = read_report(run)
            self.assertEqual(previous['evaluation']['1s']['val'], current['evaluation']['1s']['val'])
            self.assertEqual(previous['config'], current['config'])
            self.assertEqual(previous['file_hashes'], current['file_hashes'])
            self.assertEqual(len([key for key in current['evaluation']['1s'] if key.startswith('val_sliding_window_')]), 1)
            self.assertEqual(changed['int8_tflite']['num_original_files'], 6)
            self.assertEqual(changed['int8_tflite']['num_windows'], 24)
            self.assertEqual(changed['int8_tflite']['validation_protocol']['aggregation'], 'max_probability')

    def test_shared_smoke_exports_three_static_subnets(self):
        import yaml
        c = config()
        c['model']['width'] = 'all'
        c['training']['batch_size'] = 2
        c['training']['duration_secs'] = [1.5]
        c['quantization']['output_duration_secs'] = [1.5]
        c['quantization']['formats'] = ['int8']
        c['quantization']['representative_samples'] = 2
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            run = train(c, dry_run=True, limit=2, run_id='shared_integration')
            # Regression: Keras must not mutate caller config into Trackable wrappers.
            yaml.safe_dump(c)
            paths = read_report(run)['artifacts']
            self.assertEqual(len(paths), 3)
            self.assertTrue((run / 'shared_best.weights.h5').exists())
            for width, relative in zip(('1', '5', '25'), paths):
                artifact = run / relative
                meta = read_report(artifact)['metadata']
                self.assertEqual(meta['effective_width'], width)
                self.assertEqual(meta['trained_parent'], 'DSCNN_40_DW_all')
                runner = TFLiteRunner(artifact / 'exports/1.5s/int8.tflite', require_int8=True)
                self.assertFalse((artifact / 'exports/1.5s/fp32.tflite').exists())
                self.assertEqual(runner(np.zeros([1, 148, 40, 1])).shape, (1, 10))
                metrics = validation_metrics(artifact)
                self.assertIn('shared_supernet_fp32', metrics)
                self.assertLessEqual(meta['frozen_supernet_max_abs_error'], 1e-5)
