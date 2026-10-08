"""Pre-resize the 1920x1080 RARP frames once (default 288x512, same 16:9 ratio).

Decoding full-HD JPEGs dominated the original training time; the cache makes
each epoch several times faster and is used automatically by the training code.

    python -m action_recognition.prepare --data-root data
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from common.data import IMAGE_EXTENSIONS


def resize_one(job):
    source, target, width, height = job
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert('RGB').resize((width, height), Image.LANCZOS, reducing_gap=3.0) \
            .save(target, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--source', default='RARP_1FPS')
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    source_root = Path(args.data_root) / args.source
    target_root = Path(args.data_root) / f'{args.source}_{args.height}x{args.width}'
    jobs = []
    for path in source_root.rglob('*'):
        relative = path.relative_to(source_root)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            jobs.append((path, target_root / relative, args.width, args.height))
        elif path.is_file() and path.suffix == '.txt':
            (target_root / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_root / relative)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(resize_one, jobs, chunksize=32))
    print(f'Resized {len(jobs)} frames into {target_root}')


if __name__ == '__main__':
    main()
