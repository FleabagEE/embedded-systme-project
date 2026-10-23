"""TensorFlow device selection and a measured, throwaway model training probe."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform

import numpy as np

_selection = {}


def runtime_info():
    return copy.deepcopy(_selection)


def select_device(tf, runtime):
    """Select before initialization. Auto prefers the first TensorFlow GPU (Metal on Mac)."""
    global _selection
    requested = runtime['device']
    if requested not in ('auto', 'cpu', 'gpu'):
        raise ValueError('runtime.device must be auto, cpu, or gpu')
    gpus = tf.config.list_physical_devices('GPU')
    if requested == 'gpu' and not gpus:
        raise RuntimeError('GPU requested but tf.config.list_physical_devices("GPU") is empty. '
                           'Check Metal access/plugin installation or use runtime.device=auto/cpu.')
    visible = [] if requested == 'cpu' else gpus
    try:
        tf.config.set_visible_devices(visible, 'GPU')
    except RuntimeError as exc:
        raise RuntimeError('TensorFlow devices were already initialized with different visibility. '
                           'Restart Python before changing runtime.device.') from exc
    tf.config.set_soft_device_placement(True)
    selected = '/GPU:0' if visible else '/CPU:0'
    reason = ('GPU preferred and detected' if visible else
              'explicit CPU debugging option' if requested == 'cpu' else 'no TensorFlow GPU detected; CPU fallback')
    _selection = {'framework': 'TensorFlow/Keras', 'requested_device': requested,
                  'selected_device': selected, 'physical_gpus': [d.name for d in gpus],
                  'backend': 'Metal' if visible and platform.system() == 'Darwin' else 'GPU' if visible else 'CPU',
                  'reason': reason, 'deterministic_requested': runtime['deterministic'],
                  'gpu_execution_verified': False}
    print(f"Runtime: TensorFlow {tf.__version__}; selected {selected} ({_selection['backend']}); {reason}", flush=True)
    return selected


def verify_training_device(config):
    """Train a separate real project model for one step and inspect output/gradient placement.

    Uses the configured model family, widths, optimizer and determinism settings.
    Restores the initial seed afterwards, so the probe does not change run inputs
    or initial weights. Host-only initializer/shape helpers may use CPU.
    """
    import tensorflow as tf
    from training.dscnn import SharedCNN, build_model
    from training.features import feature_shape
    from training.losses import width_losses
    if not _selection:
        raise RuntimeError('Call configure_runtime before verifying the training device')
    selected = _selection['selected_device']
    shared = config['model']['width'] == 'all'
    seed = config['training']['seed']
    try:
        with tf.device(selected):
            model = SharedCNN(config) if shared else build_model(config)
            shape = feature_shape(config['audio'])
            # Generate input on the host without consuming the experiment RNG.
            x = tf.convert_to_tensor(np.random.default_rng(seed).normal(size=(2, *shape)).astype(np.float32))
            y = tf.constant([0, 1], tf.int32)
            model(x)
            before = [v.numpy().copy() for v in model.trainable_variables]
            optimizer_type = tf.keras.optimizers.Adam if config['training']['optimizer'] == 'adam' else tf.keras.optimizers.SGD
            optimizer = optimizer_type(learning_rate=config['training']['learning_rate'])

            def gradients():
                with tf.GradientTape() as tape:
                    if shared:
                        loss, _, outputs = width_losses(model, x, y, config['all_width_training']['loss_weights'])
                    else:
                        output = model(x, training=True)
                        outputs = {config['model']['width']: output}
                        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, output, from_logits=True))
                return loss, outputs, tape.gradient(loss, model.trainable_variables)

            loss, outputs, grads = gradients()
            # Probe repeatability at identical inputs/weights without assuming
            # a passing sample guarantees every Metal kernel is deterministic.
            _, repeated, repeated_grads = gradients()
            exact_repeat = (all(np.array_equal(outputs[w].numpy(), repeated[w].numpy()) for w in outputs)
                            and all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(grads, repeated_grads)))
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            after = [v.numpy() for v in model.trainable_variables]
            changed = any(not np.array_equal(a, b) for a, b in zip(before, after))
            output_devices = {w: value.device for w, value in outputs.items()}
            gradient_devices = sorted({g.device for g in grads})
            gpu_executed = (all('GPU:0' in d for d in output_devices.values())
                            and any('GPU:0' in d for d in gradient_devices))
            if not changed or not np.isfinite(float(loss.numpy())) or int(optimizer.iterations) != 1:
                raise RuntimeError('Training probe did not perform a finite one-step weight update')
            if selected == '/GPU:0' and not gpu_executed:
                raise RuntimeError(f'GPU selected but training outputs/gradients were not on GPU: {output_devices}, {gradient_devices}')
            report = {'selected_device': selected, 'gpu_execution_verified': gpu_executed,
                      'logits_devices': output_devices, 'gradient_devices': gradient_devices,
                      'variable_devices': sorted({v.device for v in model.trainable_variables}),
                      'weights_changed': changed, 'optimizer_iterations': int(optimizer.iterations),
                      'loss': float(loss.numpy()), 'repeated_forward_and_gradients_bitwise_equal': exact_repeat,
                      'post_step_weight_sha256': hashlib.sha256(b''.join(w.tobytes() for w in after)).hexdigest(),
                      'deterministic_requested': config['runtime']['deterministic'],
                      'determinism_scope': 'One fixed minibatch repeated before updating weights; not a full-run reproducibility guarantee.'}
            _selection.update({'gpu_execution_verified': gpu_executed, 'training_probe': report})
            print(f"Runtime training probe: {selected}; weights_updated={changed}; GPU_execution_verified={gpu_executed}; "
                  f"repeat_bitwise_equal={exact_repeat}", flush=True)
            return report
    except (tf.errors.OpError, RuntimeError) as exc:
        raise RuntimeError(f'Training-device verification failed on {selected}: {exc}. '
                           'For deterministic-kernel errors, use runtime.device=cpu for strict debugging or '
                           'explicitly choose runtime.deterministic=false for a new GPU run. '
                           'Determinism is never disabled automatically.') from exc
    finally:
        tf.keras.utils.set_random_seed(seed)


def main():
    from training.utils import configure_runtime, load_config, save_json, software_versions
    # With `python -m training.runtime`, configure_runtime imports the canonical
    # module. Read its state too, rather than the separate __main__ namespace.
    from training import runtime as runtime_module
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--output', help='Optional JSON verification report path')
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    configure_runtime(config)
    runtime_module.verify_training_device(config)
    report = {**runtime_module.runtime_info(), 'software_versions': software_versions()}
    if args.output:
        save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
