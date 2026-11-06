import copy
import csv
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from training.registry import create_run, write_summary
from training.artifacts import read_report, update_report, config_path
from training.utils import load_config
from tests.common import config


def allocate_worker(args):
    config, _ = args
    return create_run(config).name


class ExperimentTests(unittest.TestCase):
    def test_concurrent_process_allocation_and_never_reuse(self):
        c = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            with multiprocessing.get_context('spawn').Pool(3) as pool:
                names = pool.map(allocate_worker, [(c,i) for i in range(6)])
            self.assertEqual(len(set(names)), 6)
            self.assertEqual(sorted(names), [f'dscnn_{i:03}' for i in range(1,7)])
            shutil.rmtree(Path(tmp) / 'dscnn_006')
            self.assertEqual(create_run(c).name, 'dscnn_007')
            update_report(Path(tmp) / 'dscnn_007', state={'status':'failed'})
            self.assertEqual(create_run(c).name, 'dscnn_008')

    def test_report_config_roundtrip_and_summary_source(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            run = create_run(c)
            self.assertEqual(load_config(config_path(run)), c)
            self.assertIsInstance(load_config(config_path(run))['audio']['log_offset'], float)
            update_report(run, state={'status':'complete'}, checkpoint={'best_epoch':2},
                          validation={'accuracy':.6,'macro_f1':.5,'per_duration':{'1s':{'accuracy':.6}}})
            rows = write_summary(tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['val_accuracy'], read_report(run)['validation']['accuracy'])
            self.assertIsNone(rows[0]['num_parameters'])

    def test_failed_experiment_retains_report_before_dataset_loading(self):
        from training.train import train
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            with patch('training.dataset.make_datasets', side_effect=FileNotFoundError('fixture dataset missing')):
                with self.assertRaises(FileNotFoundError):
                    train(c)
            report = read_report(Path(tmp) / 'dscnn_001')
            self.assertEqual(report['state']['status'], 'failed')
            self.assertEqual(report['failure']['type'], 'FileNotFoundError')
            self.assertIsNone(report['validation'])
            self.assertIsNotNone(report['state']['finished_at'])
            self.assertTrue((Path(tmp) / 'summary.csv').exists())

    def test_sweep_continues_after_invalid_combination(self):
        import subprocess
        from training.sweep import run_sweep
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            path = Path(tmp) / 'input.json'
            # YAML also accepts JSON; the canonical config loader supports JSON snapshots.
            path.write_text(json.dumps(c))
            def runner(command, **kwargs):
                overrides = [command[i+1] for i,x in enumerate(command) if x == '--set']
                candidate = load_config(path, overrides)
                label = command[command.index('--run-id')+1]
                run = create_run(candidate, label)
                update_report(run, state={'status':'complete'}, validation=None)
                return subprocess.CompletedProcess(command,0)
            results = run_sweep(path, {'parameters':{'model.conv_type':['invalid','SC']}}, runner=runner)
            self.assertEqual([r['status'] for r in results], ['failed','complete'])
            self.assertIn('model.conv_type', read_report(Path(tmp)/results[0]['experiment_id'])['attempted_overrides'][0])
