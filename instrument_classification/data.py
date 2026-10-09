"""Surgical-instrument dataset: file-name parsing, splits and Dataset class.

File names look like ``v01_007125_Gr.jpg`` = video 01, frame 7125, Grasper.
"""

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common.data import IMAGE_EXTENSIONS, group_kfold, load_rgb

CLASS_CODES = ['Bi', 'Cl', 'Gr', 'Ho', 'Ir', 'Sc', 'Sp']
CLASS_NAMES = ['Bipolar', 'Clipper', 'Grasper', 'Hook', 'Irrigator',
               'Scissors', 'SpecimenBag']
NUM_CLASSES = len(CLASS_CODES)
_NAME_PATTERN = re.compile(r'^(v\d+)_(\d+)_([A-Za-z]{2})$')


@dataclass(frozen=True)
class Record:
    path: str
    video: str
    frame: int
    label: int


def parse_record(path):
    """Parse a file path into a Record; returns None for non-matching files."""
    stem, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() not in IMAGE_EXTENSIONS:
        return None
    match = _NAME_PATTERN.match(stem)
    if not match:
        return None
    video, frame, code = match.groups()
    if code not in CLASS_CODES:
        raise ValueError(f'Unknown class code {code!r} in {path}')
    return Record(str(path), video, int(frame), CLASS_CODES.index(code))


def scan_directory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing image directory: {directory}')
    records = [parse_record(p) for p in sorted(directory.rglob('*')) if p.is_file()]
    records = [r for r in records if r is not None]
    if not records:
        raise ValueError(f'No labelled images found in {directory}')
    return records


def build_splits(data_cfg, seed=0, fold=0):
    """Return {'train', 'val', 'test'} record lists.

    ``split: official`` uses the provided Dataset/{train,val,test} folders.
    ``split: video_cv`` pools all labelled images and assigns whole videos to
    train / val / test (see ``common.data.group_kfold``), which removes the
    near-duplicate adjacent frames shared between the official splits.
    """
    root = Path(data_cfg['root']) / data_cfg.get('labelled_dir', 'Dataset')
    official = {name: scan_directory(root / name) for name in ('train', 'val', 'test')}
    mode = data_cfg.get('split', 'official')
    if mode == 'official':
        return official
    if mode == 'video_cv':
        records = [r for split in official.values() for r in split]
        train_v, val_v, test_v = group_kfold(
            [r.video for r in records], data_cfg.get('n_folds', 5), fold,
            seed=data_cfg.get('split_seed', 0))
        return {
            'train': [r for r in records if r.video in train_v],
            'val': [r for r in records if r.video in val_v],
            'test': [r for r in records if r.video in test_v],
        }
    raise ValueError(f'Unknown split mode {mode!r}')


def subsample_stratified(records, fraction, seed):
    """Keep ``fraction`` of each class (at least one image), seeded.

    Used for label-efficiency experiments; the subset depends only on the seed,
    so different initialisations trained with the same seed see the same images.
    """
    rng = random.Random(seed)
    kept = []
    for label in sorted({r.label for r in records}):
        members = [r for r in records if r.label == label]
        k = max(1, round(len(members) * fraction))
        kept += rng.sample(members, k)
    return sorted(kept, key=lambda r: r.path)


class InstrumentDataset(Dataset):
    """Returns (image_tensor, label, path).

    With ``preload_size`` every image is decoded once and kept in memory at
    that (h, w) resolution, which removes JPEG decoding from the epoch loop.
    """

    def __init__(self, records, transform, preload_size=None):
        self.records = list(records)
        self.transform = transform
        self.preload_size = preload_size
        self.cache = ([load_rgb(r.path, preload_size) for r in self.records]
                      if preload_size else None)

    def __len__(self):
        return len(self.records)

    @property
    def labels(self):
        return [r.label for r in self.records]

    def __getitem__(self, index):
        record = self.records[index]
        image = self.cache[index] if self.cache else load_rgb(record.path)
        return self.transform(image), torch.tensor(record.label), record.path
