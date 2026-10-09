"""Configuration, seeding, device selection and small I/O helpers."""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

def deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_by_path(cfg, dotted_key, value):
    node = cfg
    keys = dotted_key.split('.')
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path, overrides=()):
    """Load a YAML config. ``_base_`` (relative path) enables inheritance.

    ``overrides`` are ``key.sub=value`` strings parsed as YAML scalars.
    """
    path = Path(path)
    with open(path, encoding='utf-8') as file:
        cfg = yaml.safe_load(file) or {}
    base = cfg.pop('_base_', None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Override must look like key=value, got {item!r}')
        key, value = item.split('=', 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def base_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='override config entries, e.g. train.epochs=10')
    parser.add_argument('--data-root', default=None,
                        help='overrides data.root (default: ./data)')
    parser.add_argument('--output-dir', default=None,
                        help='overrides output_dir (default: runs/<project>/<name>)')
    parser.add_argument('--device', default=None, help='auto | cuda | mps | cpu')
    parser.add_argument('--seeds', type=int, nargs='*', default=None,
                        help='overrides the list of seeds')
    parser.add_argument('--smoke', action='store_true',
                        help='tiny run (1-2 epochs, few batches) to check the pipeline')
    return parser


def resolve_config(args, project):
    cfg = load_config(args.config, args.set)
    cfg.setdefault('name', Path(args.config).stem)
    cfg.setdefault('data', {})
    if args.data_root:
        cfg['data']['root'] = args.data_root
    cfg['data'].setdefault('root', 'data')
    if args.device:
        cfg['device'] = args.device
    if args.seeds:
        cfg['seeds'] = args.seeds
    cfg.setdefault('seeds', [0])
    cfg['output_dir'] = args.output_dir or cfg.get(
        'output_dir', os.path.join('runs', project, cfg['name']))
    if args.smoke:
        apply_smoke(cfg)
    return cfg


def apply_smoke(cfg):
    cfg['smoke'] = True
    cfg['seeds'] = cfg['seeds'][:1]
    cfg['output_dir'] = cfg['output_dir'].rstrip('/') + '_smoke'
    for section in ('train', 'pretrain', 'temporal'):
        if section in cfg:
            cfg[section]['epochs'] = min(cfg[section].get('epochs', 2), 2)
            cfg[section]['max_steps_per_epoch'] = 2
            cfg[section]['warmup_epochs'] = 0
            if 'eval_every' in cfg[section]:
                cfg[section]['eval_every'] = 1
    if 'crossfit_epochs' in cfg.get('temporal', {}):
        cfg['temporal']['crossfit_epochs'] = min(cfg['temporal']['crossfit_epochs'], 2)
    cfg['data']['num_workers'] = 0


# ----------------------------------------------------------------------------
# Reproducibility and devices
# ----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Each DataLoader worker derives its NumPy/random seed from torch's seed.
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preference='auto'):
    if preference and preference != 'auto':
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ----------------------------------------------------------------------------
# I/O and logging
# ----------------------------------------------------------------------------

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(to_jsonable(obj), file, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def save_yaml(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(to_jsonable(obj), file, sort_keys=False, allow_unicode=True)


class Logger:
    """Print to stdout and append to a log file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *parts):
        message = ' '.join(str(part) for part in parts)
        stamp = time.strftime('%H:%M:%S')
        line = f'[{stamp}] {message}'
        print(line, flush=True)
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as file:
                file.write(line + '\n')


def describe_environment(device):
    return {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'device': str(device),
        'cuda_device': (torch.cuda.get_device_name(0)
                        if device.type == 'cuda' else None),
    }
