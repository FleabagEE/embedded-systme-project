"""Run a Cartesian sweep through the existing CLI, one fresh process per experiment."""
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


def run_sweep(config_path, space, overrides=(), *, skip_completed=False, dry_run=False,
              limit=None, runner=subprocess.run):
    base = load_config(config_path, overrides)
    parameters = space.get('parameters', {})
    if any(not isinstance(values, list) or not values for values in parameters.values()):
        raise ValueError('Each sweep parameter needs a nonempty list of choices')
    combinations = itertools.product(*parameters.values())
    results = []
    for values in combinations:
        changes = [f'{key}={json.dumps(value)}' for key, value in zip(parameters, values)]
        config = copy.deepcopy(base)
        label = f'batch_{uuid.uuid4().hex}'
        try:
            for change in changes:
                set_override(config, change)
            validate_config(config)
            completed = find_completed(config, dry_run, limit) if skip_completed else None
            if completed:
                results.append({'experiment_id': completed.name, 'status': 'skipped'})
                print(f'Skipped {completed}', flush=True)
                continue
            command = [sys.executable, '-m', 'training.train', '--config', str(config_path), '--run-id', label]
            for override in [*overrides, *changes]:
                command += ['--set', override]
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
            results.append({'experiment_id': run.name, 'status': status})
        except Exception as exc:
            # Invalid combinations are also explicit failed experiments, with null metrics.
            # Keep the raw attempted overrides if config validation never succeeded.
            failure_config = copy.deepcopy(base)
            run = create_run(failure_config, label)
            update_report(run, attempted_overrides=changes,
                          state={**read_report(run)['state'], 'status': 'failed', 'finished_at': utc_now()},
                          failure={'type': type(exc).__name__, 'message': str(exc)[:2000]})
            results.append({'experiment_id': run.name, 'status': 'failed'})
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
    args = parser.parse_args()
    with open(args.space) as stream:
        space = yaml.safe_load(stream)
    results = run_sweep(args.config, space, args.set, skip_completed=args.skip_completed,
                        dry_run=args.dry_run, limit=args.limit)
    print(json.dumps(results, indent=2))
    if any(row['status'] == 'failed' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
