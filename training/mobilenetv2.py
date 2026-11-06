"""Tiny inverted-residual CNN with genuine shared kernels and switchable BatchNorm."""
from __future__ import annotations
import copy
import math
import numpy as np
import tensorflow as tf

from training.dscnn import channels
from training.pooling import static_pool, shared_pool, classifier_prefix
from training.utils import canonical_name, selected_widths, width_token
from training.features import feature_shape


def bn_parameters(config):
    settings = config['model'].get('batch_norm', {})
    return {'momentum': settings.get('momentum', .9), 'epsilon': settings.get('epsilon', .001)}


def block_specs(config):
    m = config['model']
    previous = m.get('stem_channels', 16)
    specs = []
    for count, stride, repeat in zip(m.get('stage_channels',[16,24,32,48]),
                                    m.get('stage_strides',[1,2,2,1]), m.get('stage_repeats',[1,2,2,1])):
        for index in range(repeat):
            specs.append((previous, count, stride if index == 0 else 1))
            previous = count
    return specs


def build_model(config, width=None, time_frames=None):
    m = config['model']
    width = width_token(m['width'] if width is None else width)
    if width == 'all':
        raise ValueError('Use SharedMobileNetV2 for width=all')
    ratio = m.get('expansion_ratio', 3)
    spatial = 'depthwise' if m.get('mobile_spatial_conv', 'DW') == 'DW' else 'spatial'
    inputs = tf.keras.Input((time_frames, config['audio']['n_mels'], 1),
                            batch_size=1 if time_frames is not None else None, name='log_mel')
    count = channels([m.get('stem_channels',16)], width)[0]
    x = tf.keras.layers.Conv2D(count, 3, strides=2, padding='same', use_bias=False,
                               kernel_initializer='he_normal', name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn', **bn_parameters(config))(x)
    x = tf.keras.layers.ReLU(max_value=6, name='stem_relu6')(x)
    for index, (_, maximum, stride) in enumerate(block_specs(config)):
        prefix = f'block_{index}'
        residual = x
        expanded = int(x.shape[-1]) * ratio
        count = channels([maximum], width)[0]
        x = tf.keras.layers.Conv2D(expanded, 1, use_bias=False, kernel_initializer='he_normal', name=prefix+'_expand')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix+'_expand_bn', **bn_parameters(config))(x)
        x = tf.keras.layers.ReLU(max_value=6, name=prefix+'_expand_relu6')(x)
        if spatial == 'depthwise':
            x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                                               depthwise_initializer='he_normal', name=prefix+'_depthwise')(x)
        else:
            x = tf.keras.layers.Conv2D(expanded, 3, strides=stride, padding='same', use_bias=False,
                                       kernel_initializer='he_normal', name=prefix+'_spatial')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix+'_'+spatial+'_bn', **bn_parameters(config))(x)
        x = tf.keras.layers.ReLU(max_value=6, name=prefix+'_'+spatial+'_relu6')(x)
        x = tf.keras.layers.Conv2D(count, 1, use_bias=False, kernel_initializer='he_normal', name=prefix+'_project')(x)
        # Linear projection: BN, no ReLU after the projection.
        x = tf.keras.layers.BatchNormalization(name=prefix+'_project_bn', **bn_parameters(config))(x)
        if stride == 1 and int(residual.shape[-1]) == count:
            x = tf.keras.layers.Add(name=prefix+'_residual')([residual, x])
    x = static_pool(x, m.get('pooling','avg'))
    logits = tf.keras.layers.Dense(m['num_classes'], name='logits')(x)
    outputs = tf.keras.layers.Softmax(name='probabilities')(logits)
    model = tf.keras.Model(inputs, outputs, name=canonical_name(config, width))
    model.output_logits = False
    return model


class SwitchableBatchNorm(tf.keras.layers.Layer):
    """Separate affine parameters AND moving statistics for each width."""
    def __init__(self, counts, config, **kwargs):
        super().__init__(**kwargs)
        self.states = {}
        for width, count in counts.items():
            layer = tf.keras.layers.BatchNormalization(name=f'bn_w{width}', **bn_parameters(config))
            with tf.name_scope(f'{self.name}_w{width}'):
                layer.build((None, None, None, count))
            self.states[width] = layer

    def call(self, inputs, width, training=False):
        return self.states[width](inputs, training=training)


class SharedInvertedBlock(tf.keras.layers.Layer):
    def __init__(self, input_max, output_max, stride, config, widths, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.ratio = config['model'].get('expansion_ratio', 3)
        self.spatial_type = config['model'].get('mobile_spatial_conv', 'DW')
        expanded = input_max * self.ratio
        self.expansion_kernel = self.add_weight('expand_kernel', (1,1,input_max,expanded), initializer='he_normal')
        if self.spatial_type == 'DW':
            self.depthwise_kernel = self.add_weight('depthwise_kernel', (3,3,expanded,1), initializer='he_normal')
        else:
            self.spatial_kernel = self.add_weight('spatial_kernel', (3,3,expanded,expanded), initializer='he_normal')
        self.projection_kernel = self.add_weight('project_kernel', (1,1,expanded,output_max), initializer='he_normal')
        self.expand_bn = SwitchableBatchNorm({w: channels([input_max],w)[0]*self.ratio for w in widths},config,name='expand_bn')
        spatial_name = 'depthwise_bn' if self.spatial_type == 'DW' else 'spatial_bn'
        setattr(self, spatial_name, SwitchableBatchNorm(
            {w: channels([input_max],w)[0]*self.ratio for w in widths},config,name=spatial_name))
        self.project_bn = SwitchableBatchNorm({w: channels([output_max],w)[0] for w in widths},config,name='project_bn')

    def call(self, inputs, width, count, training=False):
        n_in = int(inputs.shape[-1]); expanded = n_in * self.ratio
        x = tf.nn.conv2d(inputs, self.expansion_kernel[:,:,:n_in,:expanded], strides=1, padding='SAME')
        x = tf.nn.relu6(self.expand_bn(x, width=width, training=training))
        if self.spatial_type == 'DW':
            x = tf.nn.depthwise_conv2d(x, self.depthwise_kernel[:,:,:expanded,:],
                                      strides=[1,self.stride,self.stride,1], padding='SAME')
            x = tf.nn.relu6(self.depthwise_bn(x, width=width, training=training))
        else:
            x = tf.nn.conv2d(x, self.spatial_kernel[:,:,:expanded,:expanded],
                             strides=[1,self.stride,self.stride,1], padding='SAME')
            x = tf.nn.relu6(self.spatial_bn(x, width=width, training=training))
        x = tf.nn.conv2d(x, self.projection_kernel[:,:,:expanded,:count], strides=1, padding='SAME')
        x = self.project_bn(x, width=width, training=training)
        return inputs + x if self.stride == 1 and n_in == count else x


class SharedMobileNetV2(tf.keras.Model):
    def __init__(self, config):
        super().__init__(name=canonical_name(config))
        self.config_spec = self._no_dependency(copy.deepcopy(config))
        self.output_logits = False
        m = config['model']; widths = selected_widths(config)
        maximum = m.get('stem_channels',16)
        self.stem_kernel = self.add_weight('stem_kernel',(3,3,1,maximum),initializer='he_normal')
        self.stem_bn = SwitchableBatchNorm({w:channels([maximum],w)[0] for w in widths},config,name='stem_bn')
        self.blocks = [SharedInvertedBlock(i,o,s,config,widths,name=f'shared_block_{n}')
                       for n,(i,o,s) in enumerate(block_specs(config))]
        maximum = block_specs(config)[-1][1]
        self.classifier_kernel = self.add_weight('classifier_kernel',
            (maximum * (2 if m.get('pooling','avg') == 'mix' else 1),m['num_classes']),initializer='glorot_uniform')
        self.classifier_bias = self.add_weight('classifier_bias',(m['num_classes'],),initializer='zeros')

    def call(self, inputs, width=None, training=False):
        width = selected_widths(self.config_spec)[0] if width is None else width
        m = self.config_spec['model']; mode = m.get('pooling','avg')
        count = channels([m.get('stem_channels',16)],width)[0]
        x = tf.nn.conv2d(inputs,self.stem_kernel[:,:,:,:count],strides=[1,2,2,1],padding='SAME')
        x = tf.nn.relu6(self.stem_bn(x,width=width,training=training))
        for block, (_, maximum, _) in zip(self.blocks, block_specs(self.config_spec)):
            count = channels([maximum],width)[0]
            x = block(x,width=width,count=count,training=training)
        x = shared_pool(x,mode)
        kernel = classifier_prefix(self.classifier_kernel,count,block_specs(self.config_spec)[-1][1],mode)
        return tf.nn.softmax(tf.matmul(x,kernel) + self.classifier_bias)

    def freeze(self, width, time_frames=None):
        model = build_model(self.config_spec,width,time_frames)
        m = self.config_spec['model']; count = channels([m.get('stem_channels',16)],width)[0]
        model.get_layer('stem_conv').set_weights([self.stem_kernel.numpy()[:,:,:,:count]])
        model.get_layer('stem_bn').set_weights(self.stem_bn.states[width].get_weights())
        for index,(block,(_,maximum,_)) in enumerate(zip(self.blocks,block_specs(self.config_spec))):
            expanded = count * block.ratio
            output = channels([maximum],width)[0]
            prefix = f'block_{index}'
            model.get_layer(prefix+'_expand').set_weights([block.expansion_kernel.numpy()[:,:,:count,:expanded]])
            if block.spatial_type == 'DW':
                model.get_layer(prefix+'_depthwise').set_weights([block.depthwise_kernel.numpy()[:,:,:expanded,:]])
                spatial_bn_name, spatial_bn = 'depthwise_bn', block.depthwise_bn
            else:
                model.get_layer(prefix+'_spatial').set_weights([block.spatial_kernel.numpy()[:,:,:expanded,:expanded]])
                spatial_bn_name, spatial_bn = 'spatial_bn', block.spatial_bn
            model.get_layer(prefix+'_project').set_weights([block.projection_kernel.numpy()[:,:,:expanded,:output]])
            for suffix,state in [('expand_bn',block.expand_bn),(spatial_bn_name,spatial_bn),('project_bn',block.project_bn)]:
                model.get_layer(prefix+'_'+suffix).set_weights(state.states[width].get_weights())
            count = output
        kernel = classifier_prefix(self.classifier_kernel,count,block_specs(self.config_spec)[-1][1],m.get('pooling','avg'))
        model.get_layer('logits').set_weights([kernel.numpy(),self.classifier_bias.numpy()])
        return model


def architecture_metadata(model, config, width, duration):
    m = config['model']; shape = feature_shape(config['audio'],duration)
    frames,mels,_ = shape; frames,mels = math.ceil(frames/2),math.ceil(mels/2)
    previous = channels([m.get('stem_channels',16)],width)[0]
    macs = frames*mels*9*previous
    blocks = []
    for input_max, output_max, stride in block_specs(config):
        expanded = previous*m.get('expansion_ratio',3)
        macs += frames*mels*previous*expanded
        frames,mels = math.ceil(frames/stride),math.ceil(mels/stride)
        count = channels([output_max],width)[0]
        spatial_macs = 9*expanded*(expanded if m.get('mobile_spatial_conv','DW') == 'SC' else 1)
        macs += frames*mels*(spatial_macs + expanded*count)
        blocks.append({'shape':[frames,mels,count], 'expanded_channels':expanded,
                       'stride':stride,'residual':stride==1 and previous==count})
        previous = count
    macs += previous*(2 if m.get('pooling','avg')=='mix' else 1)*m['num_classes']
    return {'num_parameters':model.count_params(),'estimated_macs':macs,
            'macs_method':'Analytical conv/depthwise/dense MACs; excludes BN/ReLU6/add/pooling/softmax/frontend; not a device measurement.',
            'duration_sec':duration,'input_shape':[1,*shape],'output_shape':[1,m['num_classes']],
            'pooling':m.get('pooling','avg'),'spatial_conv_type':m.get('mobile_spatial_conv','DW'),
            'blocks':blocks,'tensor_arena_bytes':None,
            'device_latency_ms':None,'energy_mj':None,'batch_norm':True,
            'weight_float32_bytes':sum(int(np.prod(v.shape))*4 for v in model.weights)}
