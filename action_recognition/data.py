"""RARP suturing-action dataset: annotations, sequences and frame dataset.

Layout: <root>/<split>/images/<video>/<frame_id>.jpg and
        <root>/<split>/actions/<video>.txt with rows ``start,end,class``
(inclusive ranges in original-video frame numbers; frames sampled at 1 FPS).
"""

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common.data import IMAGE_EXTENSIONS, group_kfold, load_rgb

ACTION_NAMES = [
    'Other', 'Picking up the needle', 'Positioning the needle tip',
    'Pushing the needle through the tissue', 'Pulling the needle out of the tissue',
    'Tying a knot', 'Cutting the suture', 'Returning/dropping the needle',
]
SHORT_NAMES = ['Other', 'PickUp', 'Position', 'Push', 'Pull', 'TieKnot', 'Cut', 'Return']
NUM_ACTIONS = len(ACTION_NAMES)
SPLITS = ('train', 'val', 'test')


def load_ranges(path):
    """Read and validate ``start,end,class`` rows (inclusive, non-overlapping)."""
    ranges = []
    with open(path, encoding='utf-8-sig') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            fields = line.strip().split(',')
            if len(fields) != 3:
                raise ValueError(f'{path}:{line_number}: expected 3 values')
            start, end, class_id = map(int, fields)
            if start < 0 or end < start or not 0 <= class_id < NUM_ACTIONS:
                raise ValueError(f'{path}:{line_number}: invalid annotation')
            ranges.append((start, end, class_id))
    if not ranges:
        raise ValueError(f'No annotations in {path}')
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] <= previous[1]:
            raise ValueError(f'Overlapping ranges in {path}')
    return ranges


def label_for_frame(frame_id, ranges):
    starts = [r[0] for r in ranges]
    index = bisect_right(starts, frame_id) - 1
    if index >= 0 and ranges[index][0] <= frame_id <= ranges[index][1]:
        return ranges[index][2]
    raise ValueError(f'Frame {frame_id} is not covered by any annotation range')


def surgery_id(video_name):
    """``video_15_1`` and ``video_15_2`` are two parts of the same surgery."""
    match = re.match(r'^(video_\d+)(?:_\d+)?$', video_name)
    return match.group(1) if match else video_name


@dataclass
class Sequence:
    name: str
    split: str
    paths: list
    frame_ids: np.ndarray
    labels: np.ndarray
    surgery: str = field(init=False)

    def __post_init__(self):
        self.surgery = surgery_id(self.name)

    def __len__(self):
        return len(self.paths)


def load_sequence(image_dir, action_path, split):
    image_dir = Path(image_dir)
    paths = [p for p in image_dir.iterdir()
             if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.isdigit()]
    paths.sort(key=lambda p: int(p.stem))
    if not paths:
        raise ValueError(f'No numbered frames in {image_dir}')
    ranges = load_ranges(action_path)
    frame_ids = np.array([int(p.stem) for p in paths])
    labels = np.array([label_for_frame(f, ranges) for f in frame_ids])
    return Sequence(image_dir.name, split, [str(p) for p in paths], frame_ids, labels)


def load_split(root, split):
    image_root = Path(root) / split / 'images'
    action_root = Path(root) / split / 'actions'
    if not image_root.is_dir() or not action_root.is_dir():
        raise FileNotFoundError(f'Missing images/ or actions/ under {Path(root) / split}')
    sequences = []
    for image_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        action_path = action_root / f'{image_dir.name}.txt'
        if not action_path.exists():
            raise FileNotFoundError(f'Missing annotation {action_path}')
        sequences.append(load_sequence(image_dir, action_path, split))
    return sequences


def frames_root(data_cfg):
    """Prefer the pre-resized cache produced by ``action_recognition.prepare``."""
    root = Path(data_cfg['root'])
    cached = root / data_cfg.get('frames_dir', 'RARP_1FPS_288x512')
    if cached.is_dir():
        return cached
    original = root / data_cfg.get('original_dir', 'RARP_1FPS')
    print(f'[warning] {cached} not found; decoding full-resolution frames from '
          f'{original} (run `python -m action_recognition.prepare` to speed up).')
    return original


def build_sequences(data_cfg, fold=0):
    """{'train','val','test'} -> list[Sequence].

    ``split: official`` keeps the provided video split. Note that its val video
    ``video_15_2`` comes from the same surgery as train ``video_15_1``.
    ``split: surgery_cv`` pools all videos and holds out whole surgeries.
    """
    root = frames_root(data_cfg)
    official = {split: load_split(root, split) for split in SPLITS}
    mode = data_cfg.get('split', 'official')
    if mode == 'official':
        return official
    if mode == 'surgery_cv':
        everything = [s for split in SPLITS for s in official[split]]
        train_g, val_g, test_g = group_kfold([s.surgery for s in everything],
                                             data_cfg.get('n_folds', 4), fold,
                                             seed=data_cfg.get('split_seed', 0))
        return {'train': [s for s in everything if s.surgery in train_g],
                'val': [s for s in everything if s.surgery in val_g],
                'test': [s for s in everything if s.surgery in test_g]}
    raise ValueError(f'Unknown split mode {mode!r}')


class FrameDataset(Dataset):
    """Frames of several sequences in order; returns (image, label, path)."""

    def __init__(self, sequences, transform):
        self.sequences = list(sequences)
        self.transform = transform
        self.paths = [p for s in self.sequences for p in s.paths]
        self.labels = np.concatenate([s.labels for s in self.sequences]).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_rgb(self.paths[index])
        return self.transform(image), torch.tensor(self.labels[index]), self.paths[index]


def class_counts(sequences):
    labels = np.concatenate([s.labels for s in sequences]) if sequences else np.array([], int)
    return np.bincount(labels, minlength=NUM_ACTIONS)
