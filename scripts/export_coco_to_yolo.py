#!/usr/bin/env python
import argparse, json, os, shutil, random
from pathlib import Path
from typing import Dict, List, Optional

# Convert a COCO JSON (license plate dataset) into a YOLO dataset
# Structure:
#   OUT/images/{train,val}
#   OUT/labels/{train,val}
# Copies images and writes YOLO .txt labels. Splits by ratio.

def _autodetect_category_ids(coco: Dict) -> List[int]:
    # Prefer categories whose name contains 'plate' or 'license/licence'.
    cats = coco.get('categories', []) or []
    anns = coco.get('annotations', []) or []
    if cats:
        name_to_id = {c.get('name', '').lower(): c.get('id') for c in cats}
        # heuristic matches
        matches = [c.get('id') for c in cats if any(k in (c.get('name') or '').lower() for k in ['plate', 'license', 'licence'])]
        if matches:
            return sorted(list({int(i) for i in matches if i is not None}))
        if len(cats) == 1 and cats[0].get('id') is not None:
            return [int(cats[0]['id'])]
    # fallback to most frequent category_id in annotations
    if anns:
        freq: Dict[int, int] = {}
        for a in anns:
            cid = a.get('category_id')
            if cid is None:
                continue
            freq[int(cid)] = freq.get(int(cid), 0) + 1
        if freq:
            # return all categories present if single, else the most frequent one
            if len(freq) == 1:
                return [next(iter(freq.keys()))]
            # most frequent
            top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[0][0]
            return [int(top)]
    # last resort: no filtering
    return []


def coco_to_yolo(coco_json: str, images_root: str, out_dir: str, train_ratio: float = 0.9, seed: int = 42, category_ids: Optional[List[int]] = None):
    out = Path(out_dir)
    img_train = out / 'images' / 'train'
    img_val = out / 'images' / 'val'
    lbl_train = out / 'labels' / 'train'
    lbl_val = out / 'labels' / 'val'
    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    with open(coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # determine category filter
    if category_ids is None:
        category_ids = _autodetect_category_ids(coco)
        if category_ids:
            print(f"Auto-detected category_ids: {category_ids}")
        else:
            print("No category filter applied (using all annotations).")

    imgs = {im['id']: im for im in coco.get('images', [])}
    anns_by_img: Dict[int, List[Dict]] = {im_id: [] for im_id in imgs.keys()}
    for ann in coco.get('annotations', []):
        if ann.get('iscrowd', 0):
            continue
        if category_ids and ann.get('category_id') not in category_ids:
            continue
        img_id = ann['image_id']
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)

    im_ids = list(imgs.keys())
    random.Random(seed).shuffle(im_ids)
    n_train = int(len(im_ids) * train_ratio)
    train_ids = set(im_ids[:n_train])

    copied = 0
    missing = 0
    for idx, im_id in enumerate(im_ids):
        im = imgs[im_id]
        src = Path(images_root) / im['file_name']
        if not src.exists():
            # try basename fallback
            cand = Path(images_root) / Path(im['file_name']).name
            if cand.exists():
                src = cand
            else:
                missing += 1
                continue
        W = im.get('width') or 0
        H = im.get('height') or 0
        split = 'train' if im_id in train_ids else 'val'
        dst_img = (img_train if split=='train' else img_val) / src.name
        dst_lbl = (lbl_train if split=='train' else lbl_val) / (src.stem + '.txt')
        # copy image
        if not dst_img.exists():
            shutil.copy2(str(src), str(dst_img))
        # write labels
        lines = []
        for a in anns_by_img.get(im_id, []):
            x, y, w, h = a['bbox']
            if W <= 0 or H <= 0:
                continue
            # convert to normalized cx,cy,w,h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            # For plates, single class 0
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(dst_lbl, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        copied += 1
    print(f"Prepared YOLO dataset at {out} with {copied} images (train_ratio={train_ratio}). Missing files: {missing}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco_json', required=True, help='Path to COCO annotations JSON')
    ap.add_argument('--images_root', required=True, help='Root directory containing images referenced by COCO file_name')
    ap.add_argument('--out_dir', required=True, help='Output YOLO dataset directory')
    ap.add_argument('--train_ratio', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--category_ids', type=int, nargs='*', default=None, help='COCO category_ids to include; default: auto-detect plate category or use all')
    args = ap.parse_args()
    coco_to_yolo(args.coco_json, args.images_root, args.out_dir, args.train_ratio, args.seed, args.category_ids)

if __name__ == '__main__':
    main()
