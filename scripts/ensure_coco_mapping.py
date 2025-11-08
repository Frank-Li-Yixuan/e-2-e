#!/usr/bin/env python
"""
Merge/validate multiple COCO JSONs into a unified COCO with strict category mapping.
- Enforces categories exactly: [{"id":1,"name":"face"}, {"id":2,"name":"license_plate"}]
- Filters small boxes: min(w,h) < --min_side (default 8) will be dropped
- Reindexes image/annotation IDs; preserves file_name but attempts to fix common dataset-root prefixes

Usage:
    python scripts/ensure_coco_mapping.py \
        --inputs data/pp4av.json data/wider_coco.json data/ccpd_coco.json \
        --output data/unified_coco.json \
        --min_side 8 \
        --datasets_root ..
"""
import argparse
import json
import os
import glob
from typing import List, Dict, Any

CAT_TARGET = [
    {"id": 1, "name": "face"},
    {"id": 2, "name": "license_plate"},
]
NAME_TO_ID = {c["name"]: c["id"] for c in CAT_TARGET}


def load_coco(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fixup_filename_to_existing(fn: str, datasets_root: str) -> str:
    """
    Given a COCO file_name, try to adjust it with known dataset prefixes so that
    os.path.join(datasets_root, adjusted_fn) points to an existing file on disk.
    This lets a single images_root work without copying datasets.
    """
    fn = str(fn).replace('\\', '/')
    cand = os.path.join(datasets_root, fn)
    if os.path.exists(cand):
        return fn
    lower = fn.lower()
    # CCPD sometimes needs a leading 'CCPD2020/'
    if lower.startswith('ccpd2020/') and not lower.startswith('ccpd2020/ccpd2020/'):
        fn2 = 'CCPD2020/' + fn
        if os.path.exists(os.path.join(datasets_root, fn2)):
            return fn2
    # WiderFace: try split roots
    wider_roots = [
        'WiderFace/WIDER_train/WIDER_train/images',
        'WiderFace/WIDER_val/WIDER_val/images',
        'WiderFace/WIDER_test/WIDER_test/images',
    ]
    for wr in wider_roots:
        fn2 = f"{wr}/{fn}"
        if os.path.exists(os.path.join(datasets_root, fn2)):
            return fn2
    # Not found; return original (may be resolved later by caller's images_root)
    return fn


def validate_and_remap_categories(coco: Dict[str, Any]) -> Dict[int, int]:
    """Return mapping from original category_id -> target_id. Raise if unknown names."""
    cats = coco.get("categories", [])
    if not cats:
        # allow empty; caller may only provide images/anns where labels known externally
        return {}
    # build mapping by name
    remap: Dict[int, int] = {}
    for c in cats:
        name = c.get("name", "").strip().lower()
        cid = c.get("id")
        if name not in NAME_TO_ID:
            raise ValueError(f"Unknown category '{name}' in input; expected only face/license_plate")
        remap[cid] = NAME_TO_ID[name]
    return remap


def filter_small_ann(ann: Dict[str, Any], min_side: int) -> bool:
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    return min(w, h) >= float(min_side)


def merge_cocos(inputs: List[str], output: str, min_side: int = 8, datasets_root: str | None = None) -> None:
    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    next_img_id = 1
    next_ann_id = 1

    if datasets_root is None or len(datasets_root) == 0:
        # default: parent of repo root (sibling folders like CCPD2020/, WiderFace/, PP4AV/)
        datasets_root = os.path.abspath(os.path.join(os.getcwd(), '..')).replace('\\', '/')

    for in_path in inputs:
        coco = load_coco(in_path)
        remap = validate_and_remap_categories(coco)
        # index images by file_name to reduce future duplicates across files
        local_imgid_to_new: Dict[int, int] = {}
        for img in coco.get("images", []):
            file_name = str(img.get("file_name", "")).replace('\\', '/')
            # Normalize and attempt to fix known dataset roots to ensure files resolve later
            file_name = _fixup_filename_to_existing(file_name, datasets_root)
            # Use unique compound key (dataset file base + file_name) to avoid collisions by same names from different datasets
            key = f"{os.path.basename(in_path)}:{file_name}"
            new_id = next_img_id
            next_img_id += 1
            local_imgid_to_new[img["id"]] = new_id
            images.append({
                "id": new_id,
                "file_name": file_name,
                "width": img.get("width"),
                "height": img.get("height"),
            })
        for ann in coco.get("annotations", []):
            if not filter_small_ann(ann, min_side=min_side):
                continue
            cid = ann.get("category_id")
            if remap:
                cid = remap.get(cid)
                if cid is None:
                    # unknown category, skip
                    continue
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            new_ann = {
                "id": next_ann_id,
                "image_id": local_imgid_to_new.get(ann.get("image_id")),
                "category_id": int(cid),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            # drop if image did not exist (corrupt inputs)
            if new_ann["image_id"] is None:
                continue
            anns.append(new_ann)
            next_ann_id += 1

    coco_out = {
        "images": images,
        "annotations": anns,
        "categories": CAT_TARGET,
    }
    # Inject minimal info/licenses for downstream tools expecting them
    coco_out.setdefault("info", {"description": "Unified COCO (face=1, license_plate=2)", "version": "1.0"})
    coco_out.setdefault("licenses", [])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco_out, f)
    print(f"Wrote unified COCO with {len(images)} images and {len(anns)} annotations to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input COCO JSON files (supports globs)")
    ap.add_argument("--output", default="data/unified_coco.json")
    ap.add_argument("--min_side", type=int, default=8)
    ap.add_argument("--datasets_root", type=str, default="", help="Root directory where datasets live (so images_root/dataset_path/file_name exists)")
    args = ap.parse_args()
    # Expand globs (PowerShell doesn't expand for Python)
    expanded_inputs: List[str] = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if matches:
            expanded_inputs.extend(matches)
        else:
            expanded_inputs.append(pat)
    merge_cocos(expanded_inputs, args.output, min_side=args.min_side, datasets_root=args.datasets_root)


if __name__ == "__main__":
    main()
