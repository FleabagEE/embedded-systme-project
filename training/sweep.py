"""Run Cartesian sweeps or ordered presets through the existing training CLI."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

from training.artifacts import read_report, update_report, utc_now
from training.registry import create_run, find_completed, write_summary
from training.utils import load_config, set_override, validate_config


def experiment_changes(space):
    """Ordered presets are not expanded as a Cartesian product."""
    if 'experiments' in space:
        if 'parameters' in space:
            raise ValueError('Choose experiments or parameters, not both')
        entries = space['experiments']
        if not isinstance(entries, list) or not entries:
            raise ValueError('experiments must be a nonempty list')
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get('overrides'), dict):
                raise ValueError('Each experiment needs an overrides mapping')
            yield entry.get('name'), [f'{key}={json.dumps(value)}' for key,value in entry['overrides'].items()]
    else:
        parameters = space.get('parameters', {})
        if any(not isinstance(values, list) or not values for values in parameters.values()):
            raise ValueError('Each sweep parameter needs a nonempty list of choices')
        for values in itertools.product(*parameters.values()):
            yield None, [f'{key}={json.dumps(value)}' for key,value in zip(parameters,values)]


def run_sweep(config_path, space, overrides=(), *, skip_completed=False, dry_run=False,
              limit=None, runner=subprocess.run, sequence_start=None, preview=False):
    defaults = space.get('defaults', {})
    if not isinstance(defaults, dict):
        raise ValueError('Sweep defaults must be a mapping of configuration overrides')
    base_overrides = [*[f'{k}={json.dumps(v)}' for k,v in defaults.items()], *overrides]
    base = load_config(config_path, base_overrides)
    sequence_start = space.get('sequence_start') if sequence_start is None else sequence_start
    if sequence_start is not None and (type(sequence_start) is not int or sequence_start < 1):
        raise ValueError('sequence_start must be a positive integer')
    results = []
    for preset, changes in experiment_changes(space):
        config = copy.deepcopy(base)
        label = f'{preset or "batch"}_{uuid.uuid4().hex}'
        try:
            for change in changes:
                set_override(config, change)
            validate_config(config)
            if preview:
                m = config['model']
                results.append({'preset':preset, 'status':'planned', 'architecture':m['architecture'],
                    'spatial_conv':m.get('mobile_spatial_conv','DW') if m['architecture']=='mobilenetv2' else m['conv_type'],
                    'n_mels':config['audio']['n_mels'],
                    'frequency_mask_max_bins':config['augmentation']['spectrogram']['FrequencyMask']['max_bins'],
                    'width':m['width'], 'durations':config['training']['duration_secs'],
                    'pooling':m['pooling'], 'validation':config['validation']['method'],
                    'epochs':config['training']['epochs'], 'batch_size':config['training']['batch_size']})
                continue
            completed = find_completed(config, dry_run, limit) if skip_completed else None
            if completed:
                results.append({'experiment_id':completed.name, 'preset':preset, 'status':'skipped'})
                print(f'Skipped {completed}', flush=True)
                continue
            print(f'Preset: {preset or "Cartesian configuration"}', flush=True)
            command = [sys.executable, '-m', 'training.train', '--config', str(config_path), '--run-id', label]
            for override in [*base_overrides, *changes]:
                command += ['--set', override]
            if sequence_start is not None:
                command += ['--sequence-start', str(sequence_start)]
            if dry_run:
                command.append('--dry-run')
            if limit is not None:
                command += ['--limit', str(limit)]
            outcome = runner(command, check=False)
            root = Path(config['runtime']['output_dir'])
            reports = [p for p in root.glob('*/report.json') if read_report(p.parent).get('experiment', {}).get('label') == label]
            if not reports:
                raise RuntimeError(f'Training process exited {outcome.returncode} before publishing a report')
            run = reports[0].parent
            report = read_report(run)
            status = report['state']['status']
            if (outcome.returncode != 0 or status != 'complete') and status not in ('failed', 'interrupted'):
                update_report(run, state={**report['state'], 'status': 'failed', 'finished_at': utc_now()},
                              failure={'type': 'ProcessExit', 'message': f'Training process exited {outcome.returncode}'})
                status = 'failed'
            if preset:
                update_report(run, experiment={**report['experiment'], 'preset':preset})
            results.append({'experiment_id':run.name, 'preset':preset, 'status':status,
                            'val_accuracy':(report.get('validation') or {}).get('accuracy'),
                            'artifact_path':str(run.resolve())})
        except Exception as exc:
            if preview:
                raise # Preview never reserves IDs or publishes a failed experiment.
            failure_config = copy.deepcopy(base)
            if config['model'].get('architecture') in ('dscnn','mobilenetv2'):
                failure_config['model']['architecture'] = config['model']['architecture']
            run = create_run(failure_config, label, sequence_start=sequence_start)
            update_report(run, attempted_overrides=changes,
                          state={**read_report(run)['state'], 'status': 'failed', 'finished_at': utc_now()},
                          failure={'type': type(exc).__name__, 'message': str(exc)[:2000]})
            results.append({'experiment_id':run.name, 'preset':preset, 'status':'failed'})
            print(f'{run.name} failed: {exc}; continuing sweep', flush=True)
        write_summary(base['runtime']['output_dir'])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--space', default='configs/search_space.yaml')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sequence-start', type=int)
    parser.add_argument('--preview', action='store_true', help='Validate and list configurations without training or allocating IDs')
    args = parser.parse_args()
    with open(args.space) as stream:
        space = yaml.safe_load(stream)
    results = run_sweep(args.config, space, args.set, skip_completed=args.skip_completed,
                        dry_run=args.dry_run, limit=args.limit, sequence_start=args.sequence_start, preview=args.preview)
    print(json.dumps(results, indent=2))
    if any(row['status'] == 'failed' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
