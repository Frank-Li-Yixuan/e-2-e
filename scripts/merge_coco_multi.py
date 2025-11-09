#!/usr/bin/env python
"""
Merge multiple COCO JSONs (faces + plates) into a unified train annotations file.

Typical usage (Colab):
  python scripts/merge_coco_multi.py \
    --faces-ann "/content/drive/MyDrive/WiderFace/widerface_train_coco.json" \
    --faces-root "/content/drive/MyDrive/WiderFace/WIDER_train" \
    --plates-ann "/content/drive/MyDrive/PP4AV/pp4av_coco.json" \
    --plates-root "/content/drive/MyDrive/PP4AV/images" \
    --plates-ann "/content/drive/MyDrive/CCPD2020/ccpd_coco.json" \
    --plates-root "/content/drive/MyDrive/CCPD2020/CCPD2020" \
    --common-images-root "/content/drive/MyDrive" \
    --out-train "/content/unified/unified_train.json"

Then set in paths overlay:
  paths.train_images: /content/drive/MyDrive
  paths.train_annotations: /content/unified/unified_train.json

The merged categories are fixed to: 1 -> face, 2 -> license_plate.
All image file_name will be rebased to be relative to --common-images-root.
"""
import argparse, json, os, posixpath
from pathlib import Path
from typing import Dict, List


FACE_CAT = {"id": 1, "name": "face"}
PLATE_CAT = {"id": 2, "name": "license_plate"}


def load_json(p: Path) -> Dict:
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_filenames(coco: Dict) -> None:
    for im in coco.get('images', []):
        fn = str(im.get('file_name', '')).replace('\\', '/')
        im['file_name'] = fn.lstrip('./')


def rebase_file_names(coco: Dict, src_images_root: Path, common_root: Path) -> None:
    """Make file_name relative to common_root by joining with src_images_root.

    If original file_name is already absolute or rooted elsewhere, we still
    build a path by joining src_images_root/file_name and then make it
    relative to common_root.
    """
    for im in coco.get('images', []):
        fn = str(im.get('file_name', ''))
        joined = (src_images_root / fn).resolve()
        try:
            rel = joined.relative_to(common_root.resolve())
        except Exception:
            # Fallback: keep top-level with common_root parent
            rel = Path(joined)
            # As last resort, just keep the filename
            try:
                rel = Path(posixpath.join(src_images_root.name, fn.replace('\\','/')))
            except Exception:
                rel = Path(fn.replace('\\','/'))
        im['file_name'] = rel.as_posix()


def build_id_maps(coco: Dict):
    id_to_fn = {im['id']: im['file_name'] for im in coco.get('images', [])}
    return id_to_fn


def merge_faces_plates(face_coco: Dict, plate_cocos: List[Dict]) -> Dict:
    out = {"images": [], "annotations": [], "categories": [FACE_CAT, PLATE_CAT]}
    next_img_id = 1
    next_ann_id = 1
    fname_to_id: Dict[str, int] = {}

    def add_images(src: Dict):
        nonlocal next_img_id
        for im in src.get('images', []):
            fn = im.get('file_name', '')
            if fn not in fname_to_id:
                out['images'].append({
                    'id': next_img_id,
                    'file_name': fn,
                    'width': im.get('width', -1),
                    'height': im.get('height', -1),
                })
                fname_to_id[fn] = next_img_id
                next_img_id += 1

    def add_ann_list(src_coco: Dict, id_to_fn: Dict[int, str], cat_id_map: Dict[int, int]):
        nonlocal next_ann_id
        for ann in src_coco.get('annotations', []):
            img_id = ann.get('image_id')
            fn = id_to_fn.get(img_id)
            if fn is None:
                continue
            new_img_id = fname_to_id.get(fn)
            if new_img_id is None:
                continue
            cat = cat_id_map.get(ann.get('category_id'))
            if cat is None:
                continue
            bbox = ann.get('bbox', [0,0,0,0])
            if len(bbox) != 4 or bbox[2] <=0 or bbox[3] <=0:
                continue
            out['annotations'].append({
                'id': next_ann_id,
                'image_id': new_img_id,
                'category_id': cat,
                'bbox': bbox,
                'area': float(bbox[2])*float(bbox[3]),
                'iscrowd': 0,
            })
            next_ann_id += 1

    # Normalize and add images
    normalize_filenames(face_coco)
    add_images(face_coco)
    for pc in plate_cocos:
        normalize_filenames(pc)
        add_images(pc)

    # Build id->filename maps
    face_id_to_fn = build_id_maps(face_coco)
    for pc in plate_cocos:
        # derive plate id map and category id map
        plate_id_to_fn = build_id_maps(pc)
        face_cat_ids = {c['id'] for c in face_coco.get('categories', []) if c.get('name','').lower() in ['face','person','head']}
        if not face_cat_ids:
            face_cat_ids = {1}
        plate_cat_ids = {c['id'] for c in pc.get('categories', []) if 'plate' in c.get('name','').lower()}
        if not plate_cat_ids:
            plate_cat_ids = {2}
        face_map = {cid: 1 for cid in face_cat_ids}
        plate_map = {cid: 2 for cid in plate_cat_ids}

        # Add faces from face_coco once
        # (do this only for the first plate iteration)
        if pc is plate_cocos[0]:
            add_ann_list(face_coco, face_id_to_fn, face_map)
        # Add plates from each plate coco
        add_ann_list(pc, plate_id_to_fn, plate_map)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--faces-ann', required=True, help='COCO JSON for faces (e.g., WiderFace train)')
    ap.add_argument('--faces-root', required=True, help='Images root for faces dataset')
    ap.add_argument('--plates-ann', action='append', default=[], help='COCO JSON for a plate dataset; can repeat')
    ap.add_argument('--plates-root', action='append', default=[], help='Images root for the corresponding plate dataset; order must match --plates-ann')
    ap.add_argument('--common-images-root', required=True, help='A parent directory that contains all images roots; file_name will be relative to this')
    ap.add_argument('--out-train', required=True, help='Output path for merged unified_train.json')
    args = ap.parse_args()

    if len(args.plates_ann) != len(args.plates_root):
        raise SystemExit('plates-ann and plates-root must have the same number of entries')

    faces_ann = Path(args.faces_ann)
    faces_root = Path(args.faces_root)
    common_root = Path(args.common_images_root)
    plates_anns = [Path(p) for p in args.plates_ann]
    plates_roots = [Path(p) for p in args.plates_root]

    face_coco = load_json(faces_ann)
    # Rebase file_name for faces
    rebase_file_names(face_coco, faces_root, common_root)

    plate_cocos: List[Dict] = []
    for ann_p, root_p in zip(plates_anns, plates_roots):
        pc = load_json(ann_p)
        rebase_file_names(pc, root_p, common_root)
        plate_cocos.append(pc)

    merged = merge_faces_plates(face_coco, plate_cocos)

    out_path = Path(args.out_train)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f)
    print('[OK] merged train written to', out_path.as_posix())
    print('[HINT] Set paths.train_images to', common_root.as_posix())
    print('[HINT] Set paths.train_annotations to', out_path.as_posix())


if __name__ == '__main__':
    main()
