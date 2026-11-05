import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class ColabTests(unittest.TestCase):
    def notebook_sources(self):
        nb = json.loads(Path('notebooks/colab_training.ipynb').read_text())
        return [''.join(cell['source']) for cell in nb['cells']]

    def test_augmentation_controls_before_experiments_and_valid_overrides(self):
        from training.utils import load_config
        sources = self.notebook_sources()
        augmentation = next(i for i, text in enumerate(sources) if text.startswith('#@title Data augmentation config'))
        experiment = next(i for i, text in enumerate(sources) if text.startswith('#@title Experiment controls'))
        self.assertLess(augmentation, experiment)
        namespace = {}
        exec(sources[augmentation], namespace)
        self.assertEqual(namespace['augmentation_overrides'], {'augmentation.enabled': True})
        build = namespace['augmentation_cli_overrides']
        settings = namespace['augmentation_settings']
        self.assertEqual(build('disabled', True, settings), {'augmentation.enabled': False})
        settings['spectrogram']['TimeMask'].update(enabled=False, probability=0.3)
        settings['waveform']['RandomGain']['max_db'] = 3.0
        overrides = build('custom', True, settings)
        config = load_config('config.yaml', [f'{key}={json.dumps(value)}' for key, value in overrides.items()])
        self.assertFalse(config['augmentation']['spectrogram']['TimeMask']['enabled'])
        self.assertEqual(config['augmentation']['waveform']['RandomGain']['max_db'], 3.0)
        self.assertFalse(build('custom', False, settings)['augmentation.enabled'])

    def test_single_and_batch_cli_receive_latest_augmentation_settings(self):
        sources = self.notebook_sources()
        augmentation = next(text for text in sources if text.startswith('#@title Data augmentation config'))
        experiment = next(text for text in sources if text.startswith('#@title Experiment controls'))
        launch = next(text for text in sources if text.startswith('#@title Train through the repository CLI'))
        for mode in ('single', 'batch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                run = Mock(return_value=SimpleNamespace(returncode=0))
                namespace = {'json': json, 'DATASET_ROOT': Path(tmp)/'data', 'OUTPUT_ROOT': Path(tmp)/'models'}
                exec(augmentation, namespace)
                exec(experiment, namespace)
                # Updating this cell after Experiment controls must still affect launch.
                namespace['augmentation_settings']['waveform']['PitchShift']['probability'] = 0.7
                namespace['augmentation_overrides'] = namespace['augmentation_cli_overrides']('custom', True, namespace['augmentation_settings'])
                namespace.update(experiment_mode=mode, TRAINING_PYTHON='python', REPO_DIR=tmp, TRAIN_ENV={},
                                 subprocess=SimpleNamespace(run=run), Path=lambda *_: Path(tmp)/'sweep.json', print=lambda *args: None)
                exec(launch, namespace)
                command = run.call_args.args[0]
                overrides = dict(command[i+1].split('=',1) for i, value in enumerate(command) if value == '--set')
                self.assertEqual(json.loads(overrides['augmentation.waveform.PitchShift.probability']), 0.7)
                self.assertEqual(json.loads(overrides['augmentation.enabled']), True)
                self.assertIn('training.train' if mode == 'single' else 'training.sweep', command)

    def test_notebook_valid_python_cli_and_no_credentials(self):
        path = Path('notebooks/colab_training.ipynb')
        nb = json.loads(path.read_text())
        self.assertEqual(nb['nbformat'], 4)
        source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),f'notebook-cell-{i}','exec')
        self.assertIn('training.train',source)
        self.assertIn('training.sweep',source)
        self.assertIn('colab-training',source)
        self.assertIn("'runtime.device': 'auto'",source)
        self.assertIn('run_final_test = False',source)
        self.assertIn('/content/drive/MyDrive/EmbeddedSystem/models',source)
        self.assertNotRegex(source,r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|AKIA[A-Z0-9]{16}')
        self.assertFalse(Path('configs/baseline.yaml').exists())

    def test_architecture_pooling_and_shared_options_reach_both_cli_modes(self):
        from training.utils import load_config
        sources = self.notebook_sources()
        controls = next(s for s in sources if s.startswith('#@title Experiment controls'))
        launch = next(s for s in sources if s.startswith('#@title Train through the repository CLI'))
        for architecture,token in (('DS-CNN','dscnn'),('MobileNetV2-Tiny','mobilenetv2')):
            for pooling in ('avg','max','mix'):
                for mode in ('single','batch'):
                    with self.subTest(architecture=architecture,pooling=pooling,mode=mode), tempfile.TemporaryDirectory() as tmp:
                        local_controls = controls.replace("architecture = 'DS-CNN'",f'architecture = {architecture!r}').replace("pooling = 'avg'",f'pooling = {pooling!r}')
                        run = Mock(return_value=SimpleNamespace(returncode=0))
                        ns = {'json':json,'DATASET_ROOT':Path(tmp)/'data','OUTPUT_ROOT':Path(tmp)/'models'}
                        exec(local_controls,ns)
                        ns.update(experiment_mode=mode,augmentation_overrides={'augmentation.enabled':True},
                                  TRAINING_PYTHON='python',REPO_DIR=tmp,TRAIN_ENV={},subprocess=SimpleNamespace(run=run),
                                  Path=lambda *_:Path(tmp)/'sweep.json',print=lambda *a:None)
                        exec(launch,ns)
                        command = run.call_args.args[0]
                        overrides = [command[i+1] for i,v in enumerate(command) if v=='--set']
                        c = load_config('config.yaml',overrides)
                        self.assertEqual(c['model']['architecture'],token)
                        self.assertEqual(c['model']['pooling'],pooling)
                        self.assertEqual(c['model']['expansion_ratio'],3)
                        self.assertEqual(c['all_width_training']['strategy'],'all')
                        self.assertEqual(c['quantization']['bn_recalibration_samples'],0)
                        if mode=='batch':
                            matrix = json.loads((Path(tmp)/'sweep.json').read_text())['parameters']
                            self.assertEqual(matrix['model.architecture'],[token])
                            self.assertEqual(matrix['model.pooling'],[pooling])
