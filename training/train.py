"""Train one CNN/shared CNN on configurable durations, then export fixed subnets."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import time

import numpy as np

from training.artifacts import update_report, utc_now, finalize_hashes, read_report
from training.utils import (WIDTHS, atomic_write, canonical_name, configure_runtime,
                            duration_token, export_durations, load_config,
                            software_versions, training_durations, validate_config, selected_widths, validation_protocol)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', default='config.yaml')
    result.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    result.add_argument('--run-id', help='Optional human label; experiment directories still use unique sequence numbers')
    result.add_argument('--skip-completed', action='store_true', help='Skip a matching completed experiment with verified artifacts')
    result.add_argument('--dry-run', '--smoke-test', dest='dry_run', action='store_true',
                        help='One synthetic waveform epoch plus actual exports; never empirical accuracy')
    result.add_argument('--limit', type=int, help='Diagnostic subset size per split, not a benchmark')
    result.add_argument('--skip-int8', action='store_true', help='Only export FP32')
    return result


def for_duration(dataset, duration):
    source = copy.copy(dataset)
    source.config = copy.deepcopy(dataset.config)
    source.config['audio']['duration_sec'] = duration
    return source


def validate_epoch(model, dataset, config, widths, classes, synthetic, validation_data=None):
    from evaluation.validation import ValidationData, evaluate_configurations
    from training.models import outputs_logits
    if validation_data is None:
        with ValidationData(dataset, config) as prepared:
            return validate_epoch(model, dataset, config, widths, classes, synthetic, prepared)
    row = {'validation_seconds': 0., 'val_preprocessing_seconds': 0., 'val_inference_seconds': 0.,
           'val_num_windows': 0, 'val_num_window_predictions': 0}
    shared = config['model']['width'] == 'all'
    durations = training_durations(config)
    predictors = {w: (lambda x, width=w: model(x, width=width, training=False)) if shared
                  else (lambda x: model(x, training=False)) for w in widths}
    for duration in durations:
        metrics, stats = evaluate_configurations(predictors, validation_data, classes, duration,
            synthetic=synthetic, from_logits=outputs_logits(config), include_loss=True)
        row['validation_seconds'] += stats['elapsed_seconds']
        row['val_preprocessing_seconds'] += stats['preprocessing_seconds']
        row['val_inference_seconds'] += stats['inference_seconds']
        row['val_num_windows'] += stats['num_windows']
        row['val_num_window_predictions'] += stats['num_window_predictions']
        row['val_num_clips'] = stats['num_clips']
        row[f'val_num_windows_{duration_token(duration)}'] = stats['num_windows']
        for width in widths:
            metric = metrics[width]
            tag = f'{width}_{duration_token(duration)}'
            row[f'val_loss_{tag}'] = metric['window_loss']
            row[f'val_accuracy_{tag}'] = metric['accuracy']
            row[f'val_macro_f1_{tag}'] = metric['macro_f1']
            row[f'val_num_clips_{tag}'] = metric['num_original_files']
            row[f'val_num_windows_{tag}'] = metric['num_windows']
        for metric in ('accuracy', 'macro_f1'):
            row[f'val_{metric}_{duration_token(duration)}'] = float(np.mean([row[f'val_{metric}_{w}_{duration_token(duration)}'] for w in widths]))
    for width in widths:
        for metric in ('loss', 'accuracy', 'macro_f1'):
            row[f'val_{metric}_{width}'] = float(np.mean([row[f'val_{metric}_{width}_{duration_token(d)}'] for d in durations]))
    for metric in ('loss', 'accuracy', 'macro_f1'):
        row[f'val_{metric}'] = float(np.mean([row[f'val_{metric}_{w}'] for w in widths]))
    weights = dict(zip(WIDTHS, config['all_width_training']['loss_weights'])) if shared else {widths[0]: 1.}
    row['val_total_loss'] = sum(weights[w] * row[f'val_loss_{w}'] for w in widths)
    row['val_loss'] = row['val_total_loss']
    return row


def train(config, *, dry_run=False, limit=None, run_id=None, skip_int8=False, skip_completed=False):
    from training.registry import create_run, find_completed, experiment_key, write_summary
    config = copy.deepcopy(config)
    config['audio']['duration_sec'] = training_durations(config)[0]
    validate_config(config)
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    if skip_int8:
        config['quantization']['formats'] = ['fp32']
    if skip_completed and (completed := find_completed(config, dry_run, limit)):
        print(f'Skipped completed experiment: {completed}', flush=True)
        return completed
    run = create_run(config, run_id)
    source = 'synthetic_smoke_only' if dry_run else 'UrbanSound8K'
    from training.models import model_metadata
    update_report(run, model=model_metadata(config), configuration_hash=experiment_key(config, dry_run, limit), seed=config['training']['seed'],
                  software_versions=software_versions(), data_source=source,
                  state={**read_report(run)['state'], 'status': 'training', 'started_at': utc_now()})
    print(f'Run: {run}', flush=True)
    tf = None
    try:
        tf = configure_runtime(config)
        from training.runtime import runtime_info, verify_training_device
        verify_training_device(config)
        working_root = config['runtime'].get('work_dir')
        if working_root:
            Path(working_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='urbansound-', dir=working_root) as work:
            with tf.device(runtime_info()['selected_device']):
                result = _train(config, tf, run=run, work=Path(work), dry_run=dry_run, limit=limit, skip_int8=skip_int8)
        write_summary(config['runtime']['output_dir'])
        return result
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        update_report(run, state={**read_report(run)['state'], 'status': 'interrupted' if interrupted else 'failed',
                                  'finished_at': utc_now()},
                      failure={'type': type(exc).__name__, 'message': str(exc)[:2000]})
        write_summary(config['runtime']['output_dir'])
        raise
    finally:
        if tf is not None:
            tf.keras.backend.clear_session()


def _train(config, tf, *, run, work, dry_run=False, limit=None, skip_int8=False):
    from training.dataset import make_datasets
    from training.models import build_shared_model, build_model, architecture_metadata, outputs_logits
    from training.features import feature_shape
    from training.losses import shared_train_step, sample_widths
    from training.registry import register_artifact
    from training.runtime import runtime_info
    from training.control import training_callbacks
    from evaluation.evaluate import evaluate_artifact
    from quantization.convert_int8 import export_artifact

    datasets, classes = make_datasets(config, dry_run, limit)
    from evaluation.validation import ValidationData
    validation_data = ValidationData(datasets['val'], config, cache_dir=work)
    source = 'synthetic_smoke_only' if dry_run else 'UrbanSound8K'
    versions = {**software_versions(), 'runtime': runtime_info()}
    manifest = {'class_mapping': classes, 'diagnostic_subset': dry_run or limit is not None,
                'split_counts_used': {s: d.count if dry_run else len(d.records) for s, d in datasets.items()}}
    if not dry_run:
        path = Path(config['dataset']['root']).expanduser() / 'metadata' / 'UrbanSound8K.csv'
        manifest.update(metadata_path=str(path.resolve()), metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    update_report(run, software_versions=versions, dataset_manifest=manifest)
    shared = config['model']['width'] == 'all'
    widths = selected_widths(config)
    durations = training_durations(config)
    model = build_shared_model(config) if shared else build_model(config)
    model(tf.zeros([1, *feature_shape(config['audio'])]))
    reference_width = max(widths, key=lambda w: WIDTHS[w])
    update_report(run, architecture=architecture_metadata(model, config, reference_width, durations[0]))
    optimizer_type = tf.keras.optimizers.Adam if config['training']['optimizer'] == 'adam' else tf.keras.optimizers.SGD
    optimizer = optimizer_type(learning_rate=config['training']['learning_rate'])
    weights = dict(zip(WIDTHS,config['all_width_training']['loss_weights'])) if shared else {widths[0]:1.}
    best_weights = work / ('shared_best.weights.h5' if shared else 'best.weights.h5')
    epochs = 1 if dry_run else config['training']['epochs']
    callbacks, early = training_callbacks(model, optimizer, config['training'], best_weights, durations=durations, widths=widths, epochs=epochs)

    @tf.function(reduce_retracing=True)
    def step(x, y, active_widths):
        if shared:
            return shared_train_step(model, optimizer, x, y, weights, widths=active_widths)
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=outputs_logits(config)))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss, {widths[0]: loss}, {widths[0]: logits}

    history = []
    for epoch in range(epochs):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        training_started = time.perf_counter()
        width_rng = np.random.default_rng(np.random.SeedSequence([config['training']['seed'],epoch,7919]))
        width_counts = {w:0 for w in widths}
        n, total = 0, 0.
        loss_sum, correct = {w: 0. for w in widths}, {w: 0 for w in widths}
        duration_counts = {d: 0 for d in durations}
        for x, y, _ in datasets['train'].batches(epoch):
            active_widths = tuple(str(w) for w in sample_widths(config,width_rng)) if shared else tuple(widths)
            loss, losses, logits = step(x, y, active_widths)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch + 1}')
            if epoch == 0 and n == 0:
                placement = {w: value.device for w, value in logits.items()}
                update_report(run, first_training_batch={'logits_devices': placement, 'selected_device': versions['runtime']['selected_device'],
                                                         'batch_size': len(y), 'data_source': source})
                print(f'Training first minibatch logits devices: {placement}', flush=True)
            seconds = next(d for d in durations if feature_shape(config['audio'], d)[0] == x.shape[1])
            duration_counts[seconds] += len(y)
            n += len(y); total += float(loss) * len(y)
            for w in active_widths:
                width_counts[w] += len(y)
                loss_sum[w] += float(losses[w]) * len(y)
                correct[w] += int(np.sum(np.argmax(logits[w].numpy(), axis=1) == y))
        row = {'total_loss': total / n}
        row.update({f'train_samples_{duration_token(d)}': count for d, count in duration_counts.items()})
        for w in widths:
            row[f'train_samples_width_{w}'] = width_counts[w]
            row[f'loss_{w}'] = loss_sum[w]/width_counts[w] if width_counts[w] else None
            row[f'accuracy_{w}'] = correct[w]/width_counts[w] if width_counts[w] else None
        row['accuracy'] = float(np.mean([row[f'accuracy_{w}'] for w in widths if width_counts[w]]))
        row['train_seconds'] = time.perf_counter() - training_started
        row.update(validate_epoch(model, datasets['val'], config, widths, classes, dry_run, validation_data))
        if not np.isfinite(row['val_accuracy']):
            raise FloatingPointError('Validation must provide finite val_accuracy before checkpoint callbacks')
        for callback in callbacks:
            callback.on_epoch_end(epoch, row)
        row = {'epoch': epoch + 1, **row}
        row.update(patience_used=int(early.wait),
                   patience_remaining=max(0, config['training']['early_stopping_patience'] - int(early.wait)),
                   best_epoch=int(early.best_epoch) + 1,
                   learning_rate=float(optimizer.learning_rate.numpy()))
        history.append(row)
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerows(history)
        atomic_write(run / 'history.csv', stream.getvalue())
        update_report(run,timing={'training_elapsed_seconds':row['training_elapsed_seconds'],
                                  'epoch_elapsed_seconds':[item['epoch_seconds'] for item in history],
                                  'epoch_training_seconds':[item['train_seconds'] for item in history],
                                  'epoch_validation_seconds':[item['validation_seconds'] for item in history],
                                  'validation_total_seconds':sum(item['validation_seconds'] for item in history),
                                  'eta_method':'moving mean of recent epochs; excludes export; early stopping may finish sooner'})
        if model.stop_training:
            print(f'Early stopping: monitor={early.monitor}; restoring epoch {early.best_epoch + 1}', flush=True)
            break
    model.load_weights(best_weights)
    final_checkpoint = run / best_weights.name
    atomic_write(final_checkpoint, best_weights.read_bytes())
    best_row = history[int(early.best_epoch)]
    update_report(run, validation={'accuracy': best_row['val_accuracy'], 'macro_f1': best_row['val_macro_f1'],
                      'protocol':validation_protocol(config), 'runtime_seconds':best_row['validation_seconds'],
                      'preprocessing_seconds':best_row['val_preprocessing_seconds'],
                      'inference_seconds':best_row['val_inference_seconds'],
                      'num_clips':best_row['val_num_clips'], 'num_windows':best_row['val_num_windows'],
                      'num_window_predictions':best_row['val_num_window_predictions'],
                      'count_scope':'Unique clips; windows summed over durations; window predictions also count each width.',
                      'by_configuration': {w: {duration_token(d): {'accuracy':best_row[f'val_accuracy_{w}_{duration_token(d)}'],
                          'macro_f1':best_row[f'val_macro_f1_{w}_{duration_token(d)}'],
                          'num_clips':best_row[f'val_num_clips_{w}_{duration_token(d)}'],
                          'num_windows':best_row[f'val_num_windows_{w}_{duration_token(d)}']} for d in durations} for w in widths},
                      'per_duration': {duration_token(d): {'accuracy': best_row[f'val_accuracy_{duration_token(d)}'],
                          'macro_f1': best_row[f'val_macro_f1_{duration_token(d)}']} for d in durations}}, checkpoint={'monitor': early.monitor, 'best_value': float(early.best),
                                    'best_epoch': int(early.best_epoch) + 1, 'epochs_executed': len(history),
                                    'weight_path': final_checkpoint.name, 'resume_supported': False, 'skip_completed_supported': True})
    artifacts = []
    for w in widths:
        artifact = run / 'subnets' / canonical_name(config, w) if shared else run
        artifact.mkdir(parents=True, exist_ok=True)
        frozen = model.freeze(w) if shared else model
        maximum_error = 0.
        if shared:
            for duration in durations:
                probe = np.random.default_rng(config['training']['seed']).normal(size=(2, *feature_shape(config['audio'], duration))).astype(np.float32)
                maximum_error = max(maximum_error, float(np.max(np.abs(frozen(probe).numpy() - model(probe, width=w).numpy()))))
                np.testing.assert_allclose(frozen(probe).numpy(), model(probe, width=w).numpy(), rtol=1e-5, atol=1e-5)
        from quantization.recalibrate_bn import recalibrate_batch_norm
        recalibration = recalibrate_batch_norm(frozen,datasets['train'],config,config['quantization'].get('bn_recalibration_samples',0))
        frozen_path = work / f'frozen_{w}.keras'
        frozen.save(frozen_path)
        atomic_write(artifact / 'best.keras', frozen_path.read_bytes())
        meta = {'run_id': run.name, 'canonical_name': canonical_name(config, w), 'requested_width': config['model']['width'],
                'effective_width': w, 'trained_parent': canonical_name(config) if shared else None,
                'parent_run_path': str(run) if shared else None, 'trained_duration_sec': durations[0],
                'training_duration_secs': durations, 'class_mapping': classes, 'data_source': source,
                'diagnostic_subset': dry_run or limit is not None, 'dataset_limit': limit, 'synthetic_count': limit or 12,
                'registry_root': str(Path(config['runtime']['output_dir']).resolve()),
                'frozen_supernet_max_abs_error': maximum_error if shared else None,
                'equivalence_checked_before_bn_recalibration': shared}
        update_report(artifact, metadata=meta, bn_recalibration=recalibration, deployment={'status': 'host_validated_only',
                      'device_measurements': None, 'tensor_arena_bytes': None,
                      'note': 'Static built-in operators audited at export; target firmware registration, optimized kernels and arena require device validation.'}, architecture=architecture_metadata(frozen, config, w, durations[0]))
        for duration in export_durations(config):
            export_artifact(artifact, config, datasets['train'], duration=duration, skip_int8=skip_int8)
            evaluate_artifact(artifact, for_duration(datasets['val'], duration), config, classes, duration=duration, synthetic=dry_run, validation_data=validation_data)
        update_report(artifact, state={**read_report(artifact).get('state', {}), 'status': 'complete',
                                      'data_source': source, 'int8_exported': 'int8' in config['quantization'].get('formats', ['fp32', 'int8'])})
        finalize_hashes(artifact)
        register_artifact(artifact)
        artifacts.append(str(artifact.relative_to(run)))
    validation_data.close()
    finalize_hashes(run)
    update_report(run, artifacts=artifacts, state={**read_report(run)['state'], 'status': 'complete', 'finished_at': utc_now(), 'data_source': source,
                                                 'epochs_executed': len(history), 'int8_exported': 'int8' in config['quantization'].get('formats', ['fp32', 'int8']),
                                                 'held_out_test_evaluated': False})
    print(f'Completed: {run}', flush=True)
    return run

def main():
    args = parser().parse_args()
    train(load_config(args.config, args.set), dry_run=args.dry_run, limit=args.limit,
          run_id=args.run_id, skip_int8=args.skip_int8, skip_completed=args.skip_completed)


if __name__ == '__main__':
    main()
