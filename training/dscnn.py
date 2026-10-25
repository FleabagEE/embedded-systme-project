"""BN-free comparable CNNs, including one genuinely shared nested-width supernet."""
from __future__ import annotations
import math
import copy
import numpy as np
import tensorflow as tf

from training.features import feature_shape
from training.utils import WIDTHS, canonical_name, width_token


def channels(base_channels, width):
    width = width_token(width)
    return [max(1, int(math.floor(n * WIDTHS[width]))) for n in base_channels]


def block_type(config, index):
    # Old checkpoints omit this flag and keep their original all-DW architecture.
    return 'SC' if index == 0 and config['model'].get('standard_stem', False) else config['model']['conv_type']


def build_model(config, width=None, time_frames=None):
    """All blocks use stride 2 SAME, ReLU and bias; no BN or pretrained weights."""
    width = width_token(config['model']['width'] if width is None else width)
    if width == 'all':
        raise ValueError('Use SharedCNN for width=all; build_model freezes one effective width')
    m = config['model']
    inputs = tf.keras.Input(shape=(time_frames, config['audio']['n_mels'], 1), name='log_mel')
    x = inputs
    for i, out_channels in enumerate(channels(m['base_channels'], width)):
        if block_type(config, i) == 'SC':
            x = tf.keras.layers.Conv2D(out_channels, m['kernel_size'], strides=2, padding='same',
                                       kernel_initializer='he_normal', name=f'conv_{i}')(x)
        else:
            x = tf.keras.layers.DepthwiseConv2D(m['kernel_size'], strides=2, padding='same',
                                                use_bias=False, depthwise_initializer='he_normal', name=f'depthwise_{i}')(x)
            x = tf.keras.layers.Conv2D(out_channels, 1, padding='same', kernel_initializer='he_normal', name=f'pointwise_{i}')(x)
        x = tf.keras.layers.ReLU(name=f'relu_{i}')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    outputs = tf.keras.layers.Dense(m['num_classes'], name='logits')(x)
    return tf.keras.Model(inputs, outputs, name=canonical_name(config, width))


class SharedBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size, conv_type, **kwargs):
        super().__init__(**kwargs)
        self.conv_type = conv_type
        if conv_type == 'SC':
            self.kernel = self.add_weight('kernel', (kernel_size, kernel_size, input_channels, output_channels), initializer='he_normal')
        else:
            self.depthwise_kernel = self.add_weight('depthwise_kernel', (kernel_size, kernel_size, input_channels, 1), initializer='he_normal')
            self.kernel = self.add_weight('pointwise_kernel', (1, 1, input_channels, output_channels), initializer='he_normal')
        self.bias = self.add_weight('bias', (output_channels,), initializer='zeros')

    def call(self, inputs, output_channels):
        n_in = inputs.shape[-1]
        if self.conv_type == 'DW':
            x = tf.nn.depthwise_conv2d(inputs, self.depthwise_kernel[:, :, :n_in, :], strides=[1, 2, 2, 1], padding='SAME')
            strides = [1, 1, 1, 1]
        else:
            x, strides = inputs, [1, 2, 2, 1]
        x = tf.nn.conv2d(x, self.kernel[:, :, :n_in, :output_channels], strides=strides, padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(x, self.bias[:output_channels]))


class SharedCNN(tf.keras.Model):
    """Each forward slices the SAME Variable objects, never constructs new networks."""
    def __init__(self, config):
        super().__init__(name=canonical_name(config))
        # Config is metadata, not Trackable state. Tracking a caller-owned dict
        # can mutate its nested values into wrappers that YAML cannot serialize.
        self.config_spec = self._no_dependency(copy.deepcopy(config))
        m = config['model']
        previous = 1
        self.blocks = []
        for i, out_channels in enumerate(m['base_channels']):
            self.blocks.append(SharedBlock(previous, out_channels, m['kernel_size'], block_type(config, i), name=f'shared_block_{i}'))
            previous = out_channels
        self.classifier_kernel = self.add_weight('classifier_kernel', (previous, m['num_classes']), initializer='glorot_uniform')
        self.classifier_bias = self.add_weight('classifier_bias', (m['num_classes'],), initializer='zeros')

    def call(self, inputs, width='1', training=False):
        counts = channels(self.config_spec['model']['base_channels'], width)
        x = inputs
        for block, count in zip(self.blocks, counts):
            x = block(x, output_channels=count)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.matmul(x, self.classifier_kernel[:counts[-1], :]) + self.classifier_bias

    def freeze(self, width, time_frames=None):
        model = build_model(self.config_spec, width, time_frames)
        previous = 1
        for i, (block, count) in enumerate(zip(self.blocks, channels(self.config_spec['model']['base_channels'], width))):
            if block.conv_type == 'DW':
                model.get_layer(f'depthwise_{i}').set_weights([block.depthwise_kernel.numpy()[:, :, :previous, :]])
                layer = model.get_layer(f'pointwise_{i}')
            else:
                layer = model.get_layer(f'conv_{i}')
            layer.set_weights([block.kernel.numpy()[:, :, :previous, :count], block.bias.numpy()[:count]])
            previous = count
        model.get_layer('logits').set_weights([self.classifier_kernel.numpy()[:previous, :], self.classifier_bias.numpy()])
        return model


def fix_input_shape(model, config, duration):
    width = model.name.rsplit('_', 1)[-1]
    fixed = build_model(config, width, feature_shape(config['audio'], duration)[0])
    fixed.set_weights(model.get_weights())
    return fixed


def architecture_metadata(model, config, width, duration):
    shape = feature_shape(config['audio'], duration)
    frames, mels, previous = shape
    macs = 0
    block_shapes = []
    k = config['model']['kernel_size']
    for i, count in enumerate(channels(config['model']['base_channels'], width)):
        frames, mels = math.ceil(frames / 2), math.ceil(mels / 2)
        if block_type(config, i) == 'SC':
            macs += frames * mels * k * k * previous * count
        else:
            macs += frames * mels * (k * k * previous + previous * count)
        block_shapes.append([frames, mels, count])
        previous = count
    macs += previous * config['model']['num_classes']
    return {'num_parameters': model.count_params(), 'estimated_macs': macs,
            'macs_method': 'One multiply-accumulate per kernel multiply; SAME stride-2 outputs; conv/depthwise/dense only; excludes bias/ReLU/GAP/frontend.',
            'duration_sec': duration, 'input_shape': [1, *shape],
            'output_shape': [1, config['model']['num_classes']], 'block_output_shapes': block_shapes,
            'channels': channels(config['model']['base_channels'], width),
            'block_types': [block_type(config, i) for i in range(config['model']['num_blocks'])],
            'normalization': 'BN-free; bias and ReLU after each SC or DW+pointwise block',
            'weight_float32_bytes': sum(int(np.prod(v.shape)) * 4 for v in model.weights),
            'tensor_arena_bytes': None, 'device_latency_ms': None, 'energy_mj': None}
