"""Train one CNN/shared CNN on configurable durations, then export fixed subnets."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
from pathlib import Path
import traceback

import numpy as np

from training.artifacts import update_report
from training.utils import (WIDTHS, atomic_write, canonical_name, configure_runtime,
                            duration_token, export_durations, load_config,
                            software_versions, training_durations)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', default='configs/baseline.yaml')
    result.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    result.add_argument('--run-id')
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


def validate_epoch(model, dataset, config, widths, classes, synthetic):
    import tensorflow as tf
    from evaluation.metrics import classification_metrics
    row = {}
    shared = config['model']['width'] == 'all'
    durations = training_durations(config)
    for duration in durations:
        for width in widths:
            labels, logits, ids = [], [], []
            for x, y, file_ids in dataset.batches(training=False, duration=duration):
                output = model(x, width=width, training=False) if shared else model(x, training=False)
                logits.extend(output.numpy()); labels.extend(y.tolist()); ids.extend(file_ids)
            metric = classification_metrics(labels, np.asarray(logits), ids, classes, 'val', synthetic)
            tag = f'{width}_{duration_token(duration)}'
            row[f'val_loss_{tag}'] = float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, np.asarray(logits), from_logits=True)))
            row[f'val_accuracy_{tag}'] = metric['accuracy']
            row[f'val_macro_f1_{tag}'] = metric['macro_f1']
        row[f'val_accuracy_{duration_token(duration)}'] = float(np.mean([row[f'val_accuracy_{w}_{duration_token(duration)}'] for w in widths]))
    for width in widths:
        for metric in ('loss', 'accuracy', 'macro_f1'):
            row[f'val_{metric}_{width}'] = float(np.mean([row[f'val_{metric}_{width}_{duration_token(d)}'] for d in durations]))
    for metric in ('loss', 'accuracy', 'macro_f1'):
        row[f'val_{metric}'] = float(np.mean([row[f'val_{metric}_{w}'] for w in widths]))
    weights = config['all_width_training']['loss_weights'] if shared else [1.]
    row['val_total_loss'] = sum(weight * row[f'val_loss_{w}'] for w, weight in zip(widths, weights))
    row['val_loss'] = row['val_total_loss']
    return row


def train(config, *, dry_run=False, limit=None, run_id=None, skip_int8=False):
    config = copy.deepcopy(config)
    config['audio']['duration_sec'] = training_durations(config)[0]
    tf = configure_runtime(config)
    from training.runtime import runtime_info, verify_training_device
    verify_training_device(config)
    with tf.device(runtime_info()['selected_device']):
        return _train(config, tf, dry_run=dry_run, limit=limit, run_id=run_id, skip_int8=skip_int8)


def _train(config, tf, *, dry_run=False, limit=None, run_id=None, skip_int8=False):
    from training.dataset import make_datasets
    from training.dscnn import SharedCNN, build_model, architecture_metadata
    from training.features import feature_shape
    from training.losses import shared_train_step
    from training.registry import create_run, register_artifact
    from training.runtime import runtime_info
    from training.control import training_callbacks, epoch_message
    from evaluation.evaluate import evaluate_artifact
    from quantization.convert_int8 import export_artifact

    datasets, classes = make_datasets(config, dry_run, limit)
    run = create_run(config, run_id)
    print(f'Run: {run}', flush=True)
    source = 'synthetic_smoke_only' if dry_run else 'UrbanSound8K'
    versions = {**software_versions(), 'runtime': runtime_info()}
    manifest = {'class_mapping': classes, 'diagnostic_subset': dry_run or limit is not None,
                'split_counts_used': {s: d.count if dry_run else len(d.records) for s, d in datasets.items()}}
    if not dry_run:
        path = Path(config['dataset']['root']).expanduser() / 'metadata' / 'UrbanSound8K.csv'
        manifest.update(metadata_path=str(path.resolve()), metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    update_report(run, software_versions=versions, dataset_manifest=manifest,
                  state={'status': 'training', 'data_source': source})
    try:
        shared = config['model']['width'] == 'all'
        widths = list(WIDTHS) if shared else [config['model']['width']]
        durations = training_durations(config)
        model = SharedCNN(config) if shared else build_model(config)
        model(tf.zeros([1, *feature_shape(config['audio'])]))
        optimizer_type = tf.keras.optimizers.Adam if config['training']['optimizer'] == 'adam' else tf.keras.optimizers.SGD
        optimizer = optimizer_type(learning_rate=config['training']['learning_rate'])
        weights = config['all_width_training']['loss_weights'] if shared else [1.]
        best_weights = run / ('shared_best.weights.h5' if shared else 'best.weights.h5')
        callbacks, early = training_callbacks(model, optimizer, config['training'], best_weights)

        @tf.function(reduce_retracing=True)
        def step(x, y):
            if shared:
                return shared_train_step(model, optimizer, x, y, weights)
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss, {widths[0]: loss}, {widths[0]: logits}

        history = []
        epochs = 1 if dry_run else config['training']['epochs']
        for epoch in range(epochs):
            n, total = 0, 0.
            loss_sum, correct = {w: 0. for w in widths}, {w: 0 for w in widths}
            duration_counts = {d: 0 for d in durations}
            for x, y, _ in datasets['train'].batches(epoch):
                loss, losses, logits = step(x, y)
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
                for w in widths:
                    loss_sum[w] += float(losses[w]) * len(y)
                    correct[w] += int(np.sum(np.argmax(logits[w].numpy(), axis=1) == y))
            row = {'total_loss': total / n}
            row.update({f'train_samples_{duration_token(d)}': count for d, count in duration_counts.items()})
            for w in widths:
                row[f'loss_{w}'], row[f'accuracy_{w}'] = loss_sum[w] / n, correct[w] / n
            row['accuracy'] = float(np.mean([row[f'accuracy_{w}'] for w in widths]))
            row.update(validate_epoch(model, datasets['val'], config, widths, classes, dry_run))
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
            print(epoch_message(epoch + 1, epochs, row, config['training'], durations, widths), flush=True)
            if model.stop_training:
                print(f'Early stopping: monitor={early.monitor}; restoring epoch {early.best_epoch + 1}', flush=True)
                break
        model.load_weights(best_weights)
        update_report(run, checkpoint={'monitor': early.monitor, 'best_value': float(early.best),
                                        'best_epoch': int(early.best_epoch) + 1, 'epochs_executed': len(history),
                                        'weight_path': str(best_weights), 'resume_supported': False})
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
            frozen.save(artifact / 'best.keras')
            meta = {'run_id': run.name, 'canonical_name': canonical_name(config, w), 'requested_width': config['model']['width'],
                    'effective_width': w, 'trained_parent': canonical_name(config) if shared else None,
                    'parent_run_path': str(run) if shared else None, 'trained_duration_sec': durations[0],
                    'training_duration_secs': durations, 'class_mapping': classes, 'data_source': source,
                    'diagnostic_subset': dry_run or limit is not None, 'dataset_limit': limit, 'synthetic_count': limit or 12,
                    'registry_root': str(Path(config['runtime']['output_dir']).resolve()),
                    'frozen_supernet_max_abs_error': maximum_error if shared else None}
            update_report(artifact, metadata=meta, architecture=architecture_metadata(frozen, config, w, durations[0]))
            for duration in sorted(set(export_durations(config)) | set(durations)):
                export_artifact(artifact, config, datasets['train'], duration=duration, skip_int8=skip_int8)
                evaluate_artifact(artifact, for_duration(datasets['val'], duration), config, classes, duration=duration, synthetic=dry_run)
            update_report(artifact, state={'status': 'complete', 'data_source': source, 'int8_exported': not skip_int8})
            register_artifact(artifact)
            artifacts.append(str(artifact.relative_to(run)))
        update_report(run, artifacts=artifacts, state={'status': 'complete', 'data_source': source,
                                                     'epochs_executed': len(history), 'int8_exported': not skip_int8,
                                                     'held_out_test_evaluated': False})
        print(f'Completed: {run}', flush=True)
        return run
    except Exception as exc:
        update_report(run, state={'status': 'failed', 'data_source': source, 'error_type': type(exc).__name__,
                                  'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    args = parser().parse_args()
    train(load_config(args.config, args.set), dry_run=args.dry_run, limit=args.limit,
          run_id=args.run_id, skip_int8=args.skip_int8)


if __name__ == '__main__':
    main()
