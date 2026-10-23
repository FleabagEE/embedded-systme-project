import tempfile
import unittest
from pathlib import Path
from training.artifacts import compact, read_report
from training.utils import save_json


class ArtifactTests(unittest.TestCase):
    def test_lossless_compaction_keeps_weights_and_conflicting_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'best.keras').write_bytes(b'weights must stay unchanged')
            save_json(root / 'metadata.json', {'trained_duration_sec': 1, 'effective_width': '5'})
            save_json(root / 'status.json', {'status': 'complete'})
            save_json(root / 'exports/1s/int8.json', {'input': {'dtype': 'int8'}})
            save_json(root / 'evaluation/1s/val.json', {'keras_fp32': {'accuracy': 0.5}})
            save_json(root / 'metrics.json', {'keras_fp32': {'accuracy': 0.4}})
            before = read_report(root)
            self.assertEqual(compact(root), 5)
            self.assertEqual(read_report(root), before)
            self.assertEqual((root / 'best.keras').read_bytes(), b'weights must stay unchanged')
            self.assertEqual(list(root.rglob('*.json')), [root / 'run.json'])
            self.assertIn('legacy_metrics_summary', read_report(root))

    def test_active_run_not_compacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json(root / 'status.json', {'status': 'training'})
            with self.assertRaisesRegex(RuntimeError, 'active training'):
                compact(root)
            self.assertTrue((root / 'status.json').exists())
