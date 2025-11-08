#!/usr/bin/env python
"""
Adapter to convert PP4AV original annotations to COCO format with categories:
  1: face, 2: license_plate

Assumes PP4AV annotations are text-based per-image files (e.g., annotations/fisheye/*.txt) with format:
  <class_name> x y w h   (one per line) OR dataset-specific simple formats.

This is a best-effort adapter. You may need to customize parsing for your exact PP4AV split.

Usage:
  python scripts/pp4av_to_coco.py \
    --images_dir <PP4AV images folder> \
    --ann_dir <PP4AV annotations dir> \
    --output data/pp4av_coco.json
"""
import argparse
import json
import os
from typing import Dict, Any, List, Tuple

CAT_TARGET = [
    {"id": 1, "name": "face"},
    {"id": 2, "name": "license_plate"},
]
NAME_TO_ID = {c["name"]: c["id"] for c in CAT_TARGET}


def parse_pp4av_txt(path: str) -> List[Tuple[int, float, float, float, float]]:
    """Parse a PP4AV-like txt; expected lines:
    face x y w h
    license_plate x y w h
    Returns list of (cat_id, x, y, w, h)
    """
    items: List[Tuple[int, float, float, float, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            name = parts[0].lower()
            if name not in NAME_TO_ID:
                # ignore unknown classes
                continue
            try:
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                continue
            items.append((NAME_TO_ID[name], x, y, w, h))
    return items


def _iter_image_files(images_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for root, _dirs, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                yield os.path.join(root, fn)


def convert(images_dir: str, ann_dir: str, output: str) -> None:
    images = []
    anns = []
    next_img_id = 1
    next_ann_id = 1

    images_dir = os.path.normpath(images_dir)
    ann_dir = os.path.normpath(ann_dir)

    for img_path in _iter_image_files(images_dir):
        rel = os.path.relpath(img_path, images_dir)
        fn = rel.replace("\\", "/")
        try:
            from PIL import Image  # lazy
            w, h = Image.open(img_path).size
        except Exception:
            w = h = None
        images.append({"id": next_img_id, "file_name": fn, "width": w, "height": h})
        # expected annotation txt mirrors folder structure, matching stem
        stem = os.path.splitext(rel)[0]
        cand = os.path.join(ann_dir, f"{stem}.txt")
        if os.path.exists(cand):
            items = parse_pp4av_txt(cand)
            for cid, x, y, w, h in items:
                anns.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": cid,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                next_ann_id += 1
        next_img_id += 1

    coco = {"images": images, "annotations": anns, "categories": CAT_TARGET}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"PP4AV converted: {len(images)} images, {len(anns)} annotations -> {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--output", default="data/pp4av_coco.json")
    args = ap.parse_args()
    convert(args.images_dir, args.ann_dir, args.output)


if __name__ == "__main__":
    main()
