import tempfile
import unittest
from training.utils import canonical_name, load_config, validate_config, width_token
from training.registry import create_run, upsert_registry
from training.train import parser
from tests.common import config


class NamingTests(unittest.TestCase):
    def test_names_and_cli_overrides(self):
        for conv in ('SC', 'DW'):
            for width in ('1', '5', '25', 'all'):
                c = load_config('configs/baseline.yaml', [f'model.conv_type={conv}', f'model.width={width}', 'audio.n_mels=64'])
                self.assertEqual(canonical_name(c), f'DSCNN_64_{conv}_{width}')
        args = parser().parse_args(['--smoke-test', '--set', 'audio.duration_sec=2'])
        self.assertTrue(args.dry_run)
        self.assertEqual(load_config(args.config, args.set)['audio']['duration_sec'], 2)
        for value in (5.0, '5.0', 'dw', True, 'ALL'):
            with self.assertRaises(ValueError):
                width_token(value)

    def test_invalid_overrides(self):
        for change in ('audio.n_mels=80', 'model.width=0.5', 'no.such.key=3', 'dataset.train_folds=[1,9]', 'dataset.val_folds=[1]', 'all_width_training.shared_weights=false'):
            with self.assertRaises(ValueError):
                load_config('configs/baseline.yaml', [change])

    def test_collision_duration_and_registry(self):
        c = config()
        with tempfile.TemporaryDirectory() as tmp:
            c['runtime']['output_dir'] = tmp
            first = create_run(c, 'test')
            with self.assertRaises(FileExistsError):
                create_run(c, 'test')
            c['audio']['duration_sec'] = 2
            second = create_run(c, 'test')
            self.assertNotEqual(first, second)
            row = {'artifact_id': 'one', 'run_id': 'test', 'status': 'trained'}
            upsert_registry(tmp, row)
            upsert_registry(tmp, {**row, 'status': 'int8_exported'})
            import csv
            from pathlib import Path
            with (Path(tmp) / 'model_registry.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'int8_exported')
