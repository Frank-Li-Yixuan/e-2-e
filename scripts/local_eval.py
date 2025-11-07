import os, json, gc
from pathlib import Path
from typing import List, Tuple

# Inputs via environment
REPO_DIR = os.environ.get('REPO_DIR', os.getcwd())
GT = os.environ.get('GT_ANNOTATIONS', os.path.join(REPO_DIR, 'data', 'unified', 'unified_val.json'))
ORIG_DIR = os.environ.get('ORIG_DIR', os.path.join(REPO_DIR, 'outputs', 'baselines', 'orig'))
ANON_DIRS_JOINED = os.environ.get('ANON_DIRS_JOINED', '')
ANON_DIRS = [p for p in ANON_DIRS_JOINED.split('|') if p]
OUT_ROOT = os.environ.get('OUT_ROOT', os.path.join(REPO_DIR, 'outputs', 'eval_runs', 'local_quick'))
DET = os.environ.get('DETECTOR_NAME', 'yolov11_combo')
MAX_IMAGES = int(os.environ.get('MAX_IMAGES', '200'))
BS = int(os.environ.get('BATCH_SIZE_DET', '16'))
PLATE_MODEL_PATH = os.environ.get('PLATE_MODEL_PATH', os.path.join(REPO_DIR, 'models', 'yolov8n-license-plate.pt'))

os.makedirs(os.path.join(REPO_DIR, 'outputs', 'preds', DET), exist_ok=True)
os.makedirs(os.path.join(REPO_DIR, 'outputs'), exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

from PIL import Image
import numpy as np
import torch

def load_coco_gt(gt_path: str):
    with open(gt_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    id_map = {os.path.basename(im['file_name']): im['id'] for im in coco.get('images', [])}
    return coco, id_map

coco_gt, id_map = load_coco_gt(GT)

# Auto-detect category ids from GT categories
FACE_CAT_ID, PLATE_CAT_ID = None, None
try:
    cats = coco_gt.get('categories', []) or []
    # Prefer names containing these tokens
    def find_cat(tokens):
        for c in cats:
            name = (c.get('name') or '').lower()
            if any(t in name for t in tokens):
                return int(c.get('id'))
        return None
    f_id = find_cat(['face'])
    p_id = find_cat(['plate', 'license', 'licence'])
    if f_id is not None:
        FACE_CAT_ID = f_id
    if p_id is not None:
        PLATE_CAT_ID = p_id
    # If no explicit plate category name found but only one category exists, assume it's plate
    if PLATE_CAT_ID is None and len(cats) == 1:
        try:
            PLATE_CAT_ID = int(cats[0].get('id'))
        except Exception:
            pass
except Exception:
    pass


def paths(d: str) -> List[str]:
    ps = sorted([str(p) for p in Path(d).glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    return ps[:MAX_IMAGES] if MAX_IMAGES and len(ps) > MAX_IMAGES else ps


def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_retinaface(imgs: List[str], batch_size: int = 8) -> List[Tuple[str, list]]:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    out = []
    for p in imgs:
        try:
            arr = np.array(Image.open(p).convert('RGB'))
        except Exception:
            continue
        faces = app.get(arr)
        dets = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(float).tolist()
            score = float(getattr(f, 'det_score', getattr(f, 'score', 0.0)))
            dets.append({'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score, 'category_id': FACE_CAT_ID})
        out.append((p, dets))
    return out


def run_yolo(imgs: List[str], kind: str = 'face', bs: int = 16, impl: str = 'v8') -> List[Tuple[str, list]]:
    from ultralytics import YOLO
    try:
        if kind == 'plate':
            if PLATE_MODEL_PATH and os.path.isfile(PLATE_MODEL_PATH):
                model = YOLO(PLATE_MODEL_PATH)
            else:
                # For YOLOv11 mode, require a custom local weight; skip otherwise to avoid meaningless results
                if impl == 'v11':
                    raise RuntimeError('YOLOv11 plate requires a local PLATE_MODEL_PATH; none found')
                # YOLOv8 fallback (legacy): try known online refs as last resort
                model = None
                tried = []
                for ref in ['keremberke/yolov8n-license-plate.pt', 'keremberke/yolov8m-license-plate.pt', 'keremberke/yolov8n-license-plate']:
                    try:
                        model = YOLO(ref)
                        break
                    except Exception as e:
                        tried.append((ref, str(e)))
                if model is None:
                    raise RuntimeError(f'No plate model available; tried: {tried}')
            cat = PLATE_CAT_ID
        else:
            # Not used in our pipeline; default tiny backbone if invoked
            model = YOLO('yolo11n.pt' if impl == 'v11' else 'yolov8n.pt')
            cat = 1
    except Exception as e:
        if kind == 'plate':
            print('[WARN] Plate model unavailable, skipping plates. You can place a local model at:', PLATE_MODEL_PATH, 'Error:', e)
            return [(p, []) for p in imgs]
        else:
            raise
    dev = 0 if torch.cuda.is_available() else 'cpu'
    res = []
    while True:
        try:
            for g in chunked(imgs, bs):
                r = model.predict(g, imgsz=640, device=dev, conf=0.25, iou=0.5, verbose=False)
                for p, pr in zip(g, r):
                    dets = []
                    if pr and pr.boxes is not None:
                        for b in pr.boxes:
                            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                            score = float(b.conf[0].item()) if hasattr(b, 'conf') else 0.0
                            dets.append({'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score, 'category_id': cat})
                    res.append((p, dets))
            break
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e) and bs > 1:
                bs = max(1, bs // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect(); continue
            else:
                raise
    return res


def run_and_dump(img_dir: str, out_json: str):
    imgs = paths(img_dir)
    print(f'{DET} on {len(imgs)} images: {img_dir}')
    preds = []
    if DET == 'yolov8_combo':
        df = {} if FACE_CAT_ID is None else dict(run_retinaface(imgs, batch_size=max(1, BS // 2)))
        dp = dict(run_yolo(imgs, 'plate', bs=max(1, BS // 2), impl='v8'))
        for p in imgs:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in df.get(p, []): preds.append({'image_id': im_id, **d})
            for d in dp.get(p, []): preds.append({'image_id': im_id, **d})
    elif DET == 'yolov8_face':
        if FACE_CAT_ID is not None:
            for p, ds in run_retinaface(imgs, batch_size=max(1, BS // 2)):
                im_id = id_map.get(os.path.basename(p));
                if im_id is None: continue
                for d in ds: preds.append({'image_id': im_id, **d})
    elif DET == 'yolov8_plate':
        for p, ds in run_yolo(imgs, 'plate', bs=BS, impl='v8'):
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET == 'yolov11_plate':
        for p, ds in run_yolo(imgs, 'plate', bs=BS, impl='v11'):
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET == 'yolov11_combo':
        df = {} if FACE_CAT_ID is None else dict(run_retinaface(imgs, batch_size=max(1, BS // 2)))
        dp = dict(run_yolo(imgs, 'plate', bs=max(1, BS // 2), impl='v11'))
        for p in imgs:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in df.get(p, []): preds.append({'image_id': im_id, **d})
            for d in dp.get(p, []): preds.append({'image_id': im_id, **d})
    else:
        raise SystemExit('Unsupported DETECTOR_NAME')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(preds, f)
    print('Wrote', out_json, 'num', len(preds))

pred_dir = os.path.join(REPO_DIR, 'outputs', 'preds', DET)
os.makedirs(pred_dir, exist_ok=True)
run_and_dump(ORIG_DIR, os.path.join(pred_dir, os.path.basename(os.path.normpath(ORIG_DIR)) + '.json'))
for d in ANON_DIRS:
    if os.path.isdir(d):
        run_and_dump(d, os.path.join(pred_dir, os.path.basename(os.path.normpath(d)) + '.json'))

# Minimal mAP evaluation (optional if pycocotools available)
do_eval = True
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    # Ensure COCO GT has optional keys to appease some tools
    try:
        with open(GT, 'r', encoding='utf-8') as _f:
            _gt_json = json.load(_f)
        changed = False
        if 'info' not in _gt_json:
            _gt_json['info'] = {}
            changed = True
        if 'licenses' not in _gt_json:
            _gt_json['licenses'] = []
            changed = True
        _gt_path = GT
        if changed:
            import tempfile
            td = tempfile.mkdtemp(prefix='coco_gt_')
            _gt_path = os.path.join(td, 'gt.json')
            with open(_gt_path, 'w', encoding='utf-8') as _wf:
                json.dump(_gt_json, _wf)
    except Exception:
        _gt_path = GT
    cocoGt = COCO(_gt_path)
    def eval_pred(pred_path: str):
        if os.path.getsize(pred_path) > 0:
            cocoDt = cocoGt.loadRes(pred_path)
        else:
            from pycocotools.coco import COCO as _COCO
            cocoDt = _COCO()
        E = COCOeval(cocoGt, cocoDt, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
        all_m = float(E.stats[0])
        # Evaluate a specific category if present; otherwise return -1.0
        existing_cat_ids = set(int(k) for k in getattr(cocoGt, 'cats', {}).keys())
        def cm_maybe(cat_id):
            try:
                if cat_id is None or int(cat_id) not in existing_cat_ids:
                    # No such category in GT
                    return float(-1.0)
                e = COCOeval(cocoGt, cocoDt, 'bbox')
                e.params.catIds = [int(cat_id)]
                e.evaluate(); e.accumulate(); e.summarize()
                return float(e.stats[0])
            except Exception:
                return float(-1.0)
        face_m = cm_maybe(FACE_CAT_ID)
        plate_m = cm_maybe(PLATE_CAT_ID)
        return all_m, face_m, plate_m
except Exception as e:
    print('[WARN] pycocotools not available, skipping mAP evaluation:', e)
    do_eval = False

metrics_csv = os.path.join(REPO_DIR, 'outputs', 'metrics_all.csv')
rows = []

def name_of(p: str) -> str:
    return os.path.basename(os.path.normpath(p))

preds = [(name_of(ORIG_DIR), os.path.join(pred_dir, name_of(ORIG_DIR) + '.json'))]
for d in ANON_DIRS:
    if os.path.isdir(d):
        preds.append((name_of(d), os.path.join(pred_dir, name_of(d) + '.json')))

for n, pj in preds:
    if do_eval:
        try:
            a, f, p = eval_pred(pj)
            rows.append({'name': n, 'map_50_95_all': a, 'map_50_95_face': f, 'map_50_95_plate': p})
        except Exception as e:
            print('[WARN] mAP failed for', n, e)
    else:
        rows.append({'name': n})

import pandas as pd
m = pd.DataFrame(rows)
if os.path.exists(metrics_csv):
    try:
        df = pd.read_csv(metrics_csv)
        df = df.drop(columns=[c for c in ['map_50_95_all','map_50_95_face','map_50_95_plate'] if c in df.columns], errors='ignore')
        out = df.merge(m, on='name', how='outer') if 'name' in df.columns else m
    except Exception:
        out = m
else:
    out = m
out.to_csv(metrics_csv, index=False)
print('Saved', metrics_csv)
