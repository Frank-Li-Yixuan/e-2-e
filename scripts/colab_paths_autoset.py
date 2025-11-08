#!/usr/bin/env python
"""
Colab path auto-set helper.
Creates a small YAML overlay (configs/paths_overlay.yaml) filling paths.train_images/annotations etc.
Logic:
  1. Detect mounted Drive root (default /content/drive/MyDrive) or use $DRIVE_ROOT override.
    2. Prefer unified COCO if present under data/unified/ ; else attempt WiderFace + (optional) PP4AV/CCPD merge.
  3. If both WiderFace and PP4AV individual COCO JSON exist, can optionally merge into a temporary unified JSON.

Usage (Colab cell):
    %run scripts/colab_paths_autoset.py --mode auto
    or
    !python scripts/colab_paths_autoset.py --drive-root /content/drive/MyDrive --mode auto

Then train with:
  !python -m src.train_joint --config configs/planB_colab.yaml --mode auto --max_steps 500 \
        --paths-overlay configs/paths_overlay.yaml
(You may also manually edit planB_colab.yaml if you prefer inline.)

The training launcher can be adapted to load overlay and merge before passing cfg to train().
For now this script just writes the overlay; you can manually merge or implement merge logic inside train script.
"""
import os, sys, json, argparse, yaml, hashlib, glob
from pathlib import Path
from typing import Optional

CANDIDATES = [
    ("unified_train", "data/unified/unified_train.json"),
    ("unified_val", "data/unified/unified_val.json"),
    ("widerface_train", "data/widerface_train_coco.json"),
    ("widerface_val", "data/widerface_val_coco.json"),
    ("pp4av", "data/pp4av_coco.json"),
    ("ccpd", "data/ccpd_coco.json"),
]

MERGE_OUTPUT_TRAIN = "data/unified/_auto_merged_train.json"
MERGE_OUTPUT_VAL = "data/unified/_auto_merged_val.json"

LICENSE_PLATE_CAT = {"id": 2, "name": "license_plate"}
FACE_CAT = {"id": 1, "name": "face"}


def sha(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                b = f.read(65536)
                if not b: break
                h.update(b)
        return h.hexdigest()[:12]
    except Exception:
        return "000000000000"


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_filenames(coco: dict) -> None:
    # ensure forward slashes and strip leading ./
    for im in coco.get('images', []):
        fn = im.get('file_name', '')
        fn = fn.replace('\\', '/').lstrip('./')
        im['file_name'] = fn


def merge_faces_plates(face_coco: dict, plate_coco: dict, train: bool) -> dict:
    out = {"images": [], "annotations": [], "categories": [FACE_CAT, LICENSE_PLATE_CAT]}
    # Build map to avoid id collisions
    next_img_id = 1
    next_ann_id = 1
    fname_to_id = {}
    def add_images(src):
        nonlocal next_img_id
        for im in src.get('images', []):
            fn = im['file_name']
            if fn not in fname_to_id:
                new_im = {"id": next_img_id, "file_name": fn, "width": im.get('width', -1), "height": im.get('height', -1)}
                out['images'].append(new_im)
                fname_to_id[fn] = next_img_id
                next_img_id += 1
    add_images(face_coco)
    add_images(plate_coco)
    # Add annotations
    def add_anns(src, cat_remap):
        nonlocal next_ann_id
        for ann in src.get('annotations', []):
            img_id = ann.get('image_id')
            # find file_name by original image id
            # Build original id->file map first
        
    # Build reverse maps for src datasets
    face_id_to_fn = {im['id']: im['file_name'] for im in face_coco.get('images', [])}
    plate_id_to_fn = {im['id']: im['file_name'] for im in plate_coco.get('images', [])}
    def add_ann_list(src_coco, cat_id_map, id_to_fn):
        nonlocal next_ann_id
        for ann in src_coco.get('annotations', []):
            fn = id_to_fn.get(ann.get('image_id'))
            if fn is None: continue
            new_img_id = fname_to_id.get(fn)
            if new_img_id is None: continue
            cat = cat_id_map.get(ann.get('category_id'))
            if cat is None: continue
            bbox = ann.get('bbox', [0,0,0,0])
            if bbox[2] <=0 or bbox[3] <=0: continue
            out['annotations'].append({
                'id': next_ann_id,
                'image_id': new_img_id,
                'category_id': cat,
                'bbox': bbox,
                'area': bbox[2]*bbox[3],
                'iscrowd': 0,
            })
            next_ann_id += 1
    # Determine cat id mapping (face dataset categories ->1, plate dataset categories->2)
    face_cat_ids = {c['id'] for c in face_coco.get('categories', []) if c.get('name','').lower() in ['face','person','head']}
    plate_cat_ids = {c['id'] for c in plate_coco.get('categories', []) if 'plate' in c.get('name','').lower()}
    # fallback if not found
    if not face_cat_ids:
        face_cat_ids = {1}
    if not plate_cat_ids:
        plate_cat_ids = {2}
    face_map = {cid: 1 for cid in face_cat_ids}
    plate_map = {cid: 2 for cid in plate_cat_ids}
    add_ann_list(face_coco, face_map, face_id_to_fn)
    add_ann_list(plate_coco, plate_map, plate_id_to_fn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive-root', default=os.environ.get('DRIVE_ROOT','/content/drive/MyDrive'))
    ap.add_argument('--mode', default='auto', choices=['auto','unified-only','merge'])
    ap.add_argument('--output', default='configs/paths_overlay.yaml')
    ap.add_argument('--images-root', default=None, help='Override images root for both train/val (e.g., /content/drive/MyDrive)')
    ap.add_argument('--val-images-root', default=None, help='Optional override for val images root')
    args = ap.parse_args()
    repo_dir = Path('.')
    data_dir = repo_dir / 'data'
    unified_train = data_dir / 'unified' / 'unified_train.json'
    unified_val = data_dir / 'unified' / 'unified_val.json'
    wider_train = data_dir / 'widerface_train_coco.json'
    wider_val = data_dir / 'widerface_val_coco.json'
    pp4av = data_dir / 'pp4av_coco.json'

    # Helpers: probe Google Drive for known files
    drive_root = Path(args.drive_root)
    drive_datasets = drive_root / 'datasets'
    def find_drive_file(rel_candidates):
        # rel_candidates: list of relative paths under datasets/ or under drive root
        for rel in rel_candidates:
            for base in [drive_datasets, drive_root]:
                p = base / rel
                if p.exists():
                    return p
        # fallback: filename-only recursive search (shallow) to be robust
        names = [Path(c).name for c in rel_candidates]
        for base in [drive_datasets, drive_root]:
            if not base.exists():
                continue
            for nm in names:
                hits = list(base.rglob(nm))
                if hits:
                    return hits[0]
        return None

    # Drive candidates for COCO json
    drive_unified_train = find_drive_file([
        'unified/unified_train.json', 'data/unified/unified_train.json', 'anony-project/data/unified/unified_train.json'
    ])
    drive_unified_val = find_drive_file([
        'unified/unified_val.json', 'data/unified/unified_val.json', 'anony-project/data/unified/unified_val.json'
    ])
    drive_wider_train = find_drive_file([
        'WiderFace/widerface_train_coco.json', 'widerface_train_coco.json', 'data/widerface_train_coco.json'
    ])
    drive_wider_val = find_drive_file([
        'WiderFace/widerface_val_coco.json', 'widerface_val_coco.json', 'data/widerface_val_coco.json'
    ])
    drive_pp4av = find_drive_file([
        'PP4AV/pp4av_coco.json', 'pp4av_coco.json', 'data/pp4av_coco.json'
    ])
    drive_ccpd = find_drive_file([
        'CCPD2020/ccpd_coco.json', 'ccpd_coco.json', 'data/ccpd_coco.json'
    ])

    use_train_ann = None
    use_val_ann = None

    # 1) Prefer repo-local unified
    if args.mode in ['auto','unified-only'] and unified_train.exists() and unified_val.exists():
        use_train_ann = unified_train
        use_val_ann = unified_val
    # 2) Else prefer Drive unified if present
    elif args.mode in ['auto','unified-only'] and drive_unified_train and drive_unified_val:
        use_train_ann = drive_unified_train
        use_val_ann = drive_unified_val
    # 3) Else try merge (repo-local or Drive): WiderFace (faces) + PP4AV/CCPD (plates)
    elif args.mode in ['auto','merge'] and (
        wider_train.exists() or drive_wider_train or wider_val.exists() or drive_wider_val
    ) and (
        pp4av.exists() or drive_pp4av or (data_dir / 'ccpd_coco.json').exists() or drive_ccpd
    ):
        try:
            face_src = wider_train if wider_train.exists() else (drive_wider_train or wider_train)
            plate_src_pp4av = pp4av if pp4av.exists() else drive_pp4av
            plate_src_ccpd = (data_dir / 'ccpd_coco.json') if (data_dir / 'ccpd_coco.json').exists() else drive_ccpd
            face_coco = load_json(face_src)
            normalize_filenames(face_coco)
            merged_train = None
            # sequentially merge multiple plate sources if present
            if plate_src_pp4av is not None:
                plate_coco = load_json(plate_src_pp4av)
                normalize_filenames(plate_coco)
                merged_train = merge_faces_plates(face_coco, plate_coco, train=True)
            if plate_src_ccpd is not None:
                ccpd_coco = load_json(plate_src_ccpd)
                normalize_filenames(ccpd_coco)
                merged_train = merge_faces_plates(merged_train if merged_train is not None else face_coco, ccpd_coco, train=True)
            if merged_train is None:
                raise RuntimeError('No plate dataset found for merge')
            # For val: fallback to wider_val only if no plate val; else reuse some subset
            val_src_path = wider_val if wider_val.exists() else (drive_wider_val or face_src)
            val_src = load_json(val_src_path) if isinstance(val_src_path, (str, Path)) else face_coco
            normalize_filenames(val_src)
            merged_val = None
            if plate_src_pp4av is not None:
                plate_coco = load_json(plate_src_pp4av)
                normalize_filenames(plate_coco)
                merged_val = merge_faces_plates(val_src, plate_coco, train=False)
            if plate_src_ccpd is not None:
                ccpd_coco = load_json(plate_src_ccpd)
                normalize_filenames(ccpd_coco)
                merged_val = merge_faces_plates(merged_val if merged_val is not None else val_src, ccpd_coco, train=False)
            if merged_val is None:
                merged_val = val_src
            out_train = data_dir / 'unified' / '_auto_merged_train.json'
            out_val = data_dir / 'unified' / '_auto_merged_val.json'
            out_train.parent.mkdir(parents=True, exist_ok=True)
            with open(out_train,'w',encoding='utf-8') as f: json.dump(merged_train,f)
            with open(out_val,'w',encoding='utf-8') as f: json.dump(merged_val,f)
            use_train_ann = out_train
            use_val_ann = out_val
            print('[INFO] merged unified train/val written:', out_train, out_val)
        except Exception as e:
            print('[WARN] merge failed, trying plate-only or face-only fallback:', e)
            # Prefer plate-only (PP4AV or CCPD) if face not available
            if (pp4av.exists() or drive_pp4av) or ((data_dir / 'ccpd_coco.json').exists() or drive_ccpd):
                use_train_ann = (pp4av if pp4av.exists() else (drive_pp4av or (data_dir / 'ccpd_coco.json') or drive_ccpd))
                use_val_ann = use_train_ann
            elif wider_train.exists() or drive_wider_train:
                use_train_ann = wider_train if wider_train.exists() else drive_wider_train
                use_val_ann = wider_val if wider_val.exists() else (drive_wider_val or use_train_ann)
    else:
        # Simple fallbacks without merge: prefer unified; else prefer face; else plate
        if wider_train.exists() or drive_wider_train:
            use_train_ann = wider_train if wider_train.exists() else drive_wider_train
        elif pp4av.exists() or drive_pp4av:
            use_train_ann = pp4av if pp4av.exists() else drive_pp4av
        elif (data_dir / 'ccpd_coco.json').exists() or drive_ccpd:
            use_train_ann = (data_dir / 'ccpd_coco.json') if (data_dir / 'ccpd_coco.json').exists() else drive_ccpd
        if wider_val.exists() or drive_wider_val:
            use_val_ann = wider_val if wider_val.exists() else drive_wider_val
        else:
            use_val_ann = use_train_ann

    if use_train_ann is None or use_val_ann is None:
        print('[ERROR] Could not determine annotations. Please upload unified or widerface COCO JSONs.')
        sys.exit(1)

    # Heuristic: choose images root to satisfy COCO file_name join
    # Try DRIVE_ROOT/datasets, then DRIVE_ROOT, else repo root
    def choose_images_root(ann_path: Path) -> str:
        try:
            coco = load_json(ann_path)
            ims = coco.get('images', [])
            if not ims:
                return str(repo_dir)
            fn = ims[0].get('file_name','').replace('\\','/')
            top = fn.split('/')[0] if '/' in fn else ''
            # Prefer drive_datasets if it contains the top-level folder
            if top and (drive_datasets / top).exists():
                return str(drive_datasets)
            # Or drive_root
            if top and (drive_root / top).exists():
                return str(drive_root)
            # Generic fallback
            if drive_datasets.exists():
                return str(drive_datasets)
            if drive_root.exists():
                return str(drive_root)
        except Exception:
            pass
        return str(repo_dir)

    # Allow CLI override for images roots
    train_images_root = args.images_root or choose_images_root(Path(use_train_ann))
    val_images_root = args.val_images_root or args.images_root or choose_images_root(Path(use_val_ann))

    overlay = {
        'paths': {
            'outputs': '/content/outputs/planB',
            'train_images': train_images_root,
            'train_annotations': str(use_train_ann),
            'val_images': val_images_root,
            'val_annotations': str(use_val_ann),
        }
    }
    # Write overlay YAML
    with open(args.output,'w',encoding='utf-8') as f:
        yaml.safe_dump(overlay, f, sort_keys=False, allow_unicode=True)
    print('[OK] paths overlay written to', args.output)
    print('[INFO] train_ann:', use_train_ann, 'val_ann:', use_val_ann)
    print('[INFO] train_images_root:', train_images_root, 'val_images_root:', val_images_root)

if __name__ == '__main__':
    main()
