#!/usr/bin/env python
"""
Merge multiple COCO JSONs, enforce categories (face=1, license_plate=2), filter small boxes,
then split into unified_train/val/test with given ratios.

Usage:
  python scripts/merge_and_sample_coco.py \
    --inputs data/wider_coco.json data/ccpd_coco.json ... \
    --out_dir data/unified \
    --min_side 8 \
    --train_ratio 0.7 --val_ratio 0.15 --seed 42

Artifacts:
  - data/unified/unified_train.json
  - data/unified/unified_val.json
  - data/unified/unified_test.json
"""
import argparse
import json
import os
import random
import glob
from typing import Dict, Any, List

CAT_TARGET = [
    {"id": 1, "name": "face"},
    {"id": 2, "name": "license_plate"},
]
NAME_TO_ID = {c["name"]: c["id"] for c in CAT_TARGET}


def load_coco(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_remap_categories(coco: Dict[str, Any]) -> Dict[int, int]:
    cats = coco.get("categories", [])
    if not cats:
        return {}
    remap: Dict[int, int] = {}
    for c in cats:
        name = c.get("name", "").strip().lower()
        cid = c.get("id")
        if name not in NAME_TO_ID:
            raise ValueError(f"Unknown category '{name}' in input {c}")
        remap[cid] = NAME_TO_ID[name]
    return remap


def filter_small(ann: Dict[str, Any], min_side: int) -> bool:
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    return min(float(w), float(h)) >= float(min_side)


def _fixup_filename_to_existing(fn: str, datasets_root: str) -> str:
    """
    Given a file_name from COCO, try to adjust it with known dataset prefixes so that
    join(datasets_root, adjusted_fn) points to an existing file on disk.
    This avoids copying large datasets and makes a single images_dir workable.
    """
    fn = fn.replace('\\', '/')
    cand = os.path.join(datasets_root, fn)
    if os.path.exists(cand):
        return fn
    # CCPD: might need an extra 'CCPD2020/' prefix
    lower = fn.lower()
    if lower.startswith('ccpd2020/') and not lower.startswith('ccpd2020/ccpd2020/'):
        fn2 = 'CCPD2020/' + fn
        if os.path.exists(os.path.join(datasets_root, fn2)):
            return fn2
    # WiderFace: try each split root
    wider_roots = [
        'WiderFace/WIDER_train/WIDER_train/images',
        'WiderFace/WIDER_val/WIDER_val/images',
        'WiderFace/WIDER_test/WIDER_test/images',
    ]
    for wr in wider_roots:
        fn2 = f"{wr}/{fn}"
        if os.path.exists(os.path.join(datasets_root, fn2)):
            return fn2
    # PP4AV: images under PP4AV/images/
    pp4av_prefix = f"PP4AV/images/{fn}"
    if os.path.exists(os.path.join(datasets_root, pp4av_prefix)):
        return pp4av_prefix
    # no adjustment found; return original
    return fn


def merge(inputs: List[str], min_side: int) -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    next_img_id = 1
    next_ann_id = 1
    datasets_root = os.path.abspath(os.path.join(os.getcwd(), '..')).replace('\\', '/')
    for in_path in inputs:
        coco = load_coco(in_path)
        remap = validate_and_remap_categories(coco)
        local_map: Dict[int, int] = {}
        for img in coco.get("images", []):
            fn = str(img.get("file_name")).replace('\\', '/')
            fn = _fixup_filename_to_existing(fn, datasets_root)
            new_id = next_img_id
            next_img_id += 1
            local_map[img["id"]] = new_id
            images.append({
                "id": new_id,
                "file_name": fn,
                "width": img.get("width"),
                "height": img.get("height"),
            })
        for ann in coco.get("annotations", []):
            if not filter_small(ann, min_side):
                continue
            cid = ann.get("category_id")
            if remap:
                cid = remap.get(cid)
                if cid is None:
                    continue
            img_id_new = local_map.get(ann.get("image_id"))
            if img_id_new is None:
                continue
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            anns.append({
                "id": next_ann_id,
                "image_id": img_id_new,
                "category_id": int(cid),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w) * float(h),
                "iscrowd": int(ann.get("iscrowd", 0)),
            })
            next_ann_id += 1
    return {"images": images, "annotations": anns, "categories": CAT_TARGET}


def split(coco: Dict[str, Any], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, Dict[str, Any]]:
    imgs = coco["images"]
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train + n_val]
    test_imgs = imgs[n_train + n_val:]

    def subset(imgs_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        img_ids = {im["id"] for im in imgs_subset}
        anns = [a for a in coco["annotations"] if a["image_id"] in img_ids]
        return {"images": imgs_subset, "annotations": anns, "categories": CAT_TARGET}

    return {
        "train": subset(train_imgs),
        "val": subset(val_imgs),
        "test": subset(test_imgs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="data/unified")
    ap.add_argument("--min_side", type=int, default=8)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Expand any glob patterns (PowerShell won't expand for Python)
    expanded_inputs = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if matches:
            expanded_inputs.extend(matches)
        else:
            expanded_inputs.append(pat)

    os.makedirs(args.out_dir, exist_ok=True)
    merged = merge(expanded_inputs, min_side=args.min_side)
    parts = split(merged, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)

    out_train = os.path.join(args.out_dir, "unified_train.json")
    out_val = os.path.join(args.out_dir, "unified_val.json")
    out_test = os.path.join(args.out_dir, "unified_test.json")
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(parts["train"], f)
    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(parts["val"], f)
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(parts["test"], f)
    print(f"Unified splits written: {out_train}, {out_val}, {out_test}")


if __name__ == "__main__":
    main()
