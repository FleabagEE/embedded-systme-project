from training.utils import load_config, configure_runtime


def config():
    c = load_config('configs/baseline.yaml')
    c['model'].update(conv_type='DW', width='5', num_blocks=2, base_channels=[8, 12])
    c['audio'].update(n_mels=40, duration_sec=1)
    c['augmentation']['enabled'] = False
    c['dataset']['cache_dir'] = None
    c['runtime']['device'] = 'cpu'
    c['training']['duration_secs'] = []
    c['evaluation']['num_crops'] = 1
    c['quantization']['output_duration_secs'] = [1]
    configure_runtime(c)
    return c
