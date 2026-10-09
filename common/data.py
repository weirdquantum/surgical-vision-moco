"""Image transforms, data loaders and group-wise cross-validation splits."""

import random

import torch
from PIL import Image
from torchvision import transforms as T

from .utils import seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_rgb(path, size_hw=None):
    """Open an image as RGB; optionally resize to (h, w) and close the file."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        if size_hw is not None:
            image = image.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        return image


def build_transform(cfg, train):
    """Build a torchvision transform from an ``augment`` config section.

    ``cfg['image_size']`` is (height, width). ``policy`` selects either the
    original assignment augmentation (``legacy_*``) or the improved pipeline.
    """
    height, width = cfg['image_size']
    size = (height, width)
    policy = cfg.get('policy', 'modern')
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(size, antialias=True), *normalize])

    if policy == 'legacy_task1':
        # Original Task 1/3 fine-tuning augmentation (kept for reproduction).
        ops = [T.Resize(size, antialias=True), T.RandomHorizontalFlip(),
               T.RandomVerticalFlip(), T.RandomRotation(90)]
    elif policy == 'legacy_task3':
        # Original Task 3 contrastive-view augmentation.
        ops = [T.Resize(size, antialias=True), T.RandomHorizontalFlip(),
               T.RandomRotation(15), T.ColorJitter(0.15, 0.15, 0.1, 0.02)]
    elif policy == 'legacy_task2':
        ops = [T.Resize(size, antialias=True), T.RandomRotation(5),
               T.ColorJitter(0.1, 0.1, 0.1, 0.02)]
    elif policy == 'modern':
        aspect = width / height
        ratio_jitter = cfg.get('ratio_jitter', 0.85)
        ops = [T.RandomResizedCrop(
            size, scale=tuple(cfg.get('crop_scale', (0.6, 1.0))),
            ratio=(aspect * ratio_jitter, aspect / ratio_jitter), antialias=True)]
        if cfg.get('hflip', 0.5) > 0:
            ops.append(T.RandomHorizontalFlip(cfg.get('hflip', 0.5)))
        if cfg.get('rotation', 0):
            ops.append(T.RandomRotation(cfg['rotation']))
        jitter = cfg.get('color_jitter')
        if jitter:
            ops.append(T.RandomApply([T.ColorJitter(*jitter)],
                                     p=cfg.get('color_jitter_p', 0.8)))
        if cfg.get('grayscale_p', 0) > 0:
            ops.append(T.RandomGrayscale(cfg['grayscale_p']))
        if cfg.get('blur_p', 0) > 0:
            kernel = max(3, (int(0.1 * min(size)) // 2) * 2 + 1)
            ops.append(T.RandomApply([T.GaussianBlur(kernel, sigma=(0.1, 2.0))],
                                     p=cfg['blur_p']))
    else:
        raise ValueError(f'Unknown augmentation policy: {policy}')

    ops += normalize
    if policy == 'modern' and cfg.get('random_erasing', 0) > 0:
        ops.append(T.RandomErasing(p=cfg['random_erasing'], scale=(0.02, 0.15)))
    return T.Compose(ops)


def make_loader(dataset, batch_size, shuffle, num_workers=0, seed=0,
                drop_last=False, device=None):
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin = device is not None and torch.device(device).type == 'cuda'
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=pin, generator=generator,
        worker_init_fn=seed_worker, persistent_workers=num_workers > 0,
    )


def group_kfold(groups, n_folds, fold, seed=0):
    """Split unique group ids into (train, val, test) group lists.

    Groups (videos / surgeries) are shuffled deterministically and dealt into
    ``n_folds`` folds. Fold ``fold`` is the test fold, fold ``fold + 1`` is the
    validation fold and the remainder is used for training, so no group ever
    appears in two subsets.
    """
    unique = sorted(set(groups))
    if n_folds < 3 or n_folds > len(unique):
        raise ValueError(f'n_folds must be in [3, {len(unique)}], got {n_folds}')
    if not 0 <= fold < n_folds:
        raise ValueError(f'fold must be in [0, {n_folds})')
    rng = random.Random(seed)
    rng.shuffle(unique)
    folds = [unique[i::n_folds] for i in range(n_folds)]
    test = folds[fold]
    val = folds[(fold + 1) % n_folds]
    train = [g for i, f in enumerate(folds)
             if i not in (fold, (fold + 1) % n_folds) for g in f]
    return sorted(train), sorted(val), sorted(test)
