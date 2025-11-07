#!/usr/bin/env python
"""
Align originals and anonymized images into two flat folders with matching basenames,
so that eval_utils can compute ArcFace/EasyOCR/FID easily.

Usage:
  python scripts/align_pairs.py \
    --input_json data/unified/unified_val.json \
    --images_root .. \
    --anon_dir outputs/baselines/pixel_16 \
    --out_orig_dir outputs/aligned/pixel_16/orig \
    --out_anon_dir outputs/aligned/pixel_16/anon \
    --max_images 500
"""
import argparse
import json
import os
import shutil
import glob
from typing import Optional


def find_original(images_root: str, file_name: str) -> Optional[str]:
    """Try resolve original image full path under images_root.
    Attempts direct join, then a few known dataset prefixes, finally basename recursive search.
    """
    file_name = file_name.replace('\\', '/')
    cand = os.path.join(images_root, file_name)
    if os.path.exists(cand):
        return cand
    # CCPD often under CCPD2020/
    ccpd = os.path.join(images_root, 'CCPD2020', file_name)
    if os.path.exists(ccpd):
        return ccpd
    # WiderFace splits
    for wr in [
        'WiderFace/WIDER_train/WIDER_train/images',
        'WiderFace/WIDER_val/WIDER_val/images',
        'WiderFace/WIDER_test/WIDER_test/images',
    ]:
        p = os.path.join(images_root, wr, file_name)
        if os.path.exists(p):
            return p
    # PP4AV under PP4AV/images
    ppa = os.path.join(images_root, 'PP4AV/images', file_name)
    if os.path.exists(ppa):
        return ppa
    # Fallback: basename recursive search (may be slow, but bounded if max_images is small)
    base = os.path.basename(file_name)
    matches = glob.glob(os.path.join(images_root, '**', base), recursive=True)
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_json', required=True)
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--anon_dir', required=True)
    ap.add_argument('--out_orig_dir', required=True)
    ap.add_argument('--out_anon_dir', required=True)
    ap.add_argument('--max_images', type=int, default=0)
    args = ap.parse_args()

    with open(args.input_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = coco.get('images', [])
    os.makedirs(args.out_orig_dir, exist_ok=True)
    os.makedirs(args.out_anon_dir, exist_ok=True)

    n = 0
    for img in images:
        if args.max_images and n >= args.max_images:
            break
        fn = str(img.get('file_name', ''))
        base = os.path.basename(fn)
        anon_src = os.path.join(args.anon_dir, base)
        if not os.path.exists(anon_src):
            continue
        orig_src = find_original(args.images_root, fn)
        if not orig_src or not os.path.exists(orig_src):
            continue
        shutil.copyfile(orig_src, os.path.join(args.out_orig_dir, base))
        shutil.copyfile(anon_src, os.path.join(args.out_anon_dir, base))
        n += 1
    print(f'Aligned {n} pairs into {args.out_orig_dir} and {args.out_anon_dir}')


if __name__ == '__main__':
    main()
