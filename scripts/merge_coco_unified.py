#!/usr/bin/env python
"""Merge multiple COCO style annotation jsons (faces / plates / etc.) into a unified COCO file.

Usage (example):

python scripts/merge_coco_unified.py \
  --out data/unified/unified_train.json \
  --image-root data/unified/images \
  --ds CCPD2020=data/CCPD2020/ccpd_coco.json,data/CCPD2020/CCPD2020 \
  --ds WiderFace=data/WiderFace/widerface_train_coco.json,data/WiderFace/WIDER_train \
  --ds PP4AV=data/PP4AV/pp4av_coco.json,data/PP4AV/images

This will:
 1. Load each json.
 2. Copy / link (optional) images into a unified image root (if --copy or --symlink used; by default leaves images in-place and keeps relative paths).
 3. Remap image & annotation IDs to avoid collisions.
 4. Normalize categories so that same category names share the same ID.

If you do NOT want to physically move/copy images, omit --copy/--symlink and provide prefixes so file_name keeps relative subfolder (recommended to avoid duplication).

After merge set in config (planB_colab.yaml):

paths:
  train_images: data/unified/images
  train_annotations: data/unified/unified_train.json

Requirements: only standard library.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_ds_arg(arg: str) -> Tuple[str, str, str]:
    """Parse DS spec: NAME=ann.json,image_root"""
    if '=' not in arg:
        raise ValueError(f"--ds format must be NAME=ann.json,image_root, got: {arg}")
    name, rest = arg.split('=', 1)
    if ',' not in rest:
        raise ValueError(f"--ds format must be NAME=ann.json,image_root, got: {arg}")
    ann, img_root = rest.split(',', 1)
    return name.strip(), ann.strip(), img_root.strip()


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', action='append', required=True, help='Dataset spec NAME=ann.json,image_root (repeatable)')
    ap.add_argument('--out', required=True, help='Output unified COCO json')
    ap.add_argument('--image-root', required=True, help='Unified images root (used as base path in config)')
    ap.add_argument('--strategy', choices=['refer', 'copy', 'symlink'], default='refer', help='refer: keep original file_name with relative prefix; copy/symlink: place under unified root')
    ap.add_argument('--rel-prefix', action='store_true', help='Force file_name to include dataset name prefix (recommended with refer)')
    args = ap.parse_args()

    ds_specs = [parse_ds_arg(d) for d in args.ds]
    ensure_dir(Path(args.out).parent)
    if args.strategy != 'refer':
        ensure_dir(args.image_root)

    unified = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    # category name -> id
    cat_name_to_id: Dict[str, int] = {}
    next_cat_id = 1
    next_img_id = 1
    next_ann_id = 1

    for ds_name, ann_path, img_root in ds_specs:
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # categories
        local_cat_map: Dict[int, int] = {}
        for c in data.get('categories', []):
            cname = c.get('name')
            if cname not in cat_name_to_id:
                cat_name_to_id[cname] = next_cat_id
                unified['categories'].append({'id': next_cat_id, 'name': cname, 'supercategory': c.get('supercategory', '')})
                next_cat_id += 1
            local_cat_map[c['id']] = cat_name_to_id[cname]
        # images
        img_id_map: Dict[int, int] = {}
        for img in data.get('images', []):
            new_id = next_img_id
            next_img_id += 1
            img_id_map[img['id']] = new_id
            orig_file = img.get('file_name')
            # normalize path
            orig_file_norm = orig_file.replace('\\', '/') if isinstance(orig_file, str) else str(orig_file)
            # Decide new file_name
            if args.strategy == 'refer':
                # keep relative path with optional dataset prefix
                if args.rel_prefix:
                    rel_name = f"{ds_name}/{orig_file_norm}"
                else:
                    rel_name = orig_file_norm
                new_file_name = rel_name
            else:
                # copy/symlink into unified root keeping ds_name prefix
                rel_name = f"{ds_name}/{Path(orig_file_norm).name}"
                dest_path = Path(args.image_root) / rel_name
                ensure_dir(dest_path.parent)
                src_path = Path(img_root) / orig_file_norm
                if not dest_path.exists():
                    if args.strategy == 'copy':
                        shutil.copy2(src_path, dest_path)
                    elif args.strategy == 'symlink':
                        try:
                            os.symlink(src_path, dest_path)
                        except FileExistsError:
                            pass
                new_file_name = rel_name
            unified['images'].append({
                'id': new_id,
                'width': img.get('width'),
                'height': img.get('height'),
                'file_name': new_file_name,
                'dataset': ds_name,
            })
        # annotations
        for ann in data.get('annotations', []):
            old_img = ann.get('image_id')
            if old_img not in img_id_map:
                continue
            new_ann = {
                'id': next_ann_id,
                'image_id': img_id_map[old_img],
                'category_id': local_cat_map.get(ann.get('category_id'), ann.get('category_id')),
                'bbox': ann.get('bbox'),
                'area': ann.get('area', 0),
                'iscrowd': ann.get('iscrowd', 0),
            }
            next_ann_id += 1
            unified['annotations'].append(new_ann)

    # Save
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(unified, f)
    print(f"Unified COCO written: {args.out}")
    print(f"Images: {len(unified['images'])}  Annots: {len(unified['annotations'])}  Categories: {len(unified['categories'])}")
    print("Category mapping:")
    for c in unified['categories']:
        print(f"  {c['id']}: {c['name']}")

    # Usage hint for config
    print("\nAdd to config (paths section):")
    print(f"  train_images: {args.image_root}")
    print(f"  train_annotations: {args.out}")
    print(f"  val_images: {args.image_root}  # (or another merged val json)")
    print(f"  val_annotations: <your_val_json>")

if __name__ == '__main__':
    main()
