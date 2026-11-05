"""Architecture dispatch only; training/evaluation/export remain shared."""
from training.utils import WIDTHS, selected_widths


def is_mobilenet(config):
    return config['model'].get('architecture','dscnn') == 'mobilenetv2'


def outputs_logits(config):
    return not is_mobilenet(config)


def build_model(config, width=None, time_frames=None):
    from training import mobilenetv2, dscnn
    return (mobilenetv2 if is_mobilenet(config) else dscnn).build_model(config,width,time_frames)


def build_shared_model(config):
    from training import mobilenetv2, dscnn
    return mobilenetv2.SharedMobileNetV2(config) if is_mobilenet(config) else dscnn.SharedCNN(config)


def fix_input_shape(model, config, duration):
    from training.features import feature_shape
    fixed = build_model(config,model.name.rsplit('_',1)[-1],feature_shape(config['audio'],duration)[0])
    fixed.set_weights(model.get_weights())
    return fixed


def architecture_metadata(model, config, width, duration):
    from training import mobilenetv2, dscnn
    return (mobilenetv2 if is_mobilenet(config) else dscnn).architecture_metadata(model,config,width,duration)


def model_metadata(config):
    m = config['model']; mobile = is_mobilenet(config)
    return {'architecture':m.get('architecture','dscnn'),'pooling':m.get('pooling','avg'),
            'expansion_ratio':m.get('expansion_ratio',3) if mobile else None,
            'spatial_conv_type':m.get('mobile_spatial_conv','DW') if mobile else m['conv_type'],
            'width_options':[WIDTHS[w] for w in selected_widths(config)],
            'batch_norm': {'enabled':mobile,'switchable':mobile and m['width']=='all',
                           **(m.get('batch_norm',{'momentum':.9,'epsilon':.001}) if mobile else {})},
            'output':'probabilities' if mobile else 'logits'}
