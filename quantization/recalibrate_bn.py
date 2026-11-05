"""Optional train-only, augmentation-free population BatchNorm recalibration."""
import copy
import numpy as np
import tensorflow as tf
from training.utils import training_durations


def recalibrate_batch_norm(model, dataset, config, samples):
    layers = [layer for layer in model.layers if isinstance(layer,tf.keras.layers.BatchNormalization)]
    if not samples or not layers:
        return {'applied':False,'samples':0,'reason':'disabled' if not samples else 'model has no BatchNorm'}
    if dataset.split != 'train':
        raise ValueError('BN recalibration requires training folds only')
    durations = training_durations(config)
    representatives = []
    per_duration = int(np.ceil(samples/len(durations)))
    for duration in durations:
        source = copy.copy(dataset); source.config = copy.deepcopy(dataset.config)
        source.config['audio']['duration_sec'] = duration
        representatives.append(list(source.representative(per_duration)))
    # Interleave durations rather than filling the budget with the first one.
    selected = [(duration, group[index]) for index in range(per_duration)
                for duration, group in zip(durations, representatives) if index < len(group)][:samples]
    inputs = [features for _, features in selected]
    if not inputs:
        raise ValueError('No training samples for BN recalibration')
    trainables = [v.numpy().copy() for v in model.trainable_variables]
    # Calibrate in topological order. Earlier BNs already use their calibrated
    # population statistics when measuring the next BN's input activations.
    for layer in layers:
        activations = tf.keras.Model(model.inputs,layer.input)
        total = np.zeros(layer.moving_mean.shape,np.float64)
        squared = np.zeros_like(total); count = 0
        for features in inputs:
            values = activations(features,training=False).numpy().astype(np.float64)
            total += values.sum(axis=(0,1,2)); squared += np.square(values).sum(axis=(0,1,2))
            count += int(np.prod(values.shape[:3]))
        mean = total/count
        layer.moving_mean.assign(mean.astype(np.float32))
        layer.moving_variance.assign(np.maximum(0,squared/count-np.square(mean)).astype(np.float32))
    for before, variable in zip(trainables,model.trainable_variables):
        np.testing.assert_array_equal(before,variable.numpy())
    return {'applied':True,'samples':len(inputs),'durations':list(dict.fromkeys(d for d, _ in selected)),
            'samples_per_duration':{str(d):sum(actual == d for actual, _ in selected) for d in durations},
            'split':'train','augmentation':False,
            'method':'layerwise population moments; earlier BN layers evaluated with calibrated statistics',
            'trainable_weights_unchanged':True}
