#!/usr/bin/env python
"""
Filter a COCO json by a list of filenames or by a path prefix to produce a split (e.g., CCPD2020 val).

Examples:
# Using a list of relative paths or basenames (one per line)
python scripts/coco_filter_by_list.py \
  --src data/ccpd_coco.json \
  --list data/ccpd_val_list.txt \
  --out data/ccpd_val.json

# Using a prefix (e.g., images under a folder named 'val')
python scripts/coco_filter_by_list.py \
  --src data/ccpd_coco.json \
  --prefix "val/" \
  --out data/ccpd_val.json

Options:
  --complement   Invert selection (e.g., produce train = all - val)

Matching rules:
- Paths are normalized to forward slashes. List entries can be either full relative paths
  matching the COCO 'file_name' field, or just basenames; both will be matched.
- When --prefix is used, any image whose file_name startswith the given prefix will be selected.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Set


def load_list(list_path: str) -> Set[str]:
    items: Set[str] = set()
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace('\\', '/')
            items.add(s)
            items.add(Path(s).name)  # also allow basename match
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Source COCO json path')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--list', dest='list_path', help='Text file of image paths/basenames, one per line')
    grp.add_argument('--prefix', help='Path prefix to match in file_name (after normalization)')
    ap.add_argument('--out', required=True, help='Output COCO json path')
    ap.add_argument('--complement', action='store_true', help='Invert selection to keep non-matching images')
    args = ap.parse_args()

    with open(args.src, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    select_names: Set[str] = set()
    if args.list_path:
        select_names = load_list(args.list_path)

    # Build selection set by image id
    keep_by_id: Dict[int, bool] = {}
    for img in coco.get('images', []):
        fn = str(img.get('file_name', '')).replace('\\', '/')
        base = Path(fn).name
        match = False
        if args.list_path:
            if fn in select_names or base in select_names:
                match = True
        elif args.prefix:
            match = fn.startswith(args.prefix)
        keep = (not args.complement and match) or (args.complement and not match)
        keep_by_id[img['id']] = keep

    images_new = [img for img in coco.get('images', []) if keep_by_id.get(img['id'], False)]
    id_map = {img['id']: i+1 for i, img in enumerate(images_new)}
    anns_new = []
    next_ann_id = 1
    for ann in coco.get('annotations', []):
        old_iid = ann.get('image_id')
        if old_iid in id_map:
            new_ann = {
                'id': next_ann_id,
                'image_id': id_map[old_iid],
                'category_id': ann.get('category_id'),
                'bbox': ann.get('bbox'),
                'area': ann.get('area', 0),
                'iscrowd': ann.get('iscrowd', 0),
            }
            next_ann_id += 1
            anns_new.append(new_ann)

    # Reindex image ids
    for i, img in enumerate(images_new, start=1):
        img['id'] = i

    out = {
        'images': images_new,
        'annotations': anns_new,
        'categories': coco.get('categories', []),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f"Wrote filtered COCO: {args.out}")
    print(f"Images: {len(images_new)}  Annotations: {len(anns_new)}  Categories: {len(out['categories'])}")


if __name__ == '__main__':
    main()
