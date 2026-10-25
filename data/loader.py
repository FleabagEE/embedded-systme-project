"""Inspect local UrbanSound8K; never download on import. Run as python -m data.loader."""
import argparse
from training.dataset import load_metadata, partition
from training.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    records, classes = load_metadata(config['dataset']['root'])
    print({'classes': classes, 'files': len(records),
           'splits': {s: len(partition(records, config, s)) for s in ('train', 'val', 'test')}})


if __name__ == '__main__':
    main()
