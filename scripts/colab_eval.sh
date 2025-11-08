#!/usr/bin/env bash
set -euo pipefail
set -a

# ===== User Config =====
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive}"
REPO_DIR="${REPO_DIR:-/content/drive/MyDrive/anony-project}"
GT_ANNOTATIONS="${GT_ANNOTATIONS:-$REPO_DIR/data/unified/unified_val.json}"
ORIG_DIR="${ORIG_DIR:-$REPO_DIR/outputs/baselines/orig}"
ANON_DIRS_ARR=(
    "$REPO_DIR/outputs/baselines/pixelation_8x8"
    "$REPO_DIR/outputs/baselines/mask_out"
    "$REPO_DIR/outputs/baselines/gaussian_k7"
    "$REPO_DIR/outputs/baselines/pixelation_16x16"
    "$REPO_DIR/outputs/baselines/gaussian_k15"
)
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/eval_runs/colab_all}"
THRESHOLDS="${THRESHOLDS:-$REPO_DIR/configs/eval_thresholds.yaml}"
MAX_IMAGES=${MAX_IMAGES:-2000}
BATCH_SIZE_DET=${BATCH_SIZE_DET:-16}
# Options: retinaface | yolov8_face | yolov8_plate | yolov8_combo | yolov11_plate | yolov11_combo
DETECTOR_NAME="${DETECTOR_NAME:-yolov11_combo}"
# Default关闭第三方基线，避免网络受限时失败
ENABLE_DEEPPRIVACY2=${ENABLE_DEEPPRIVACY2:-0}
ENABLE_LDFA=${ENABLE_LDFA:-0}
# 可选：本地车牌模型路径（优先使用本地，避免在线下载）
PLATE_MODEL_PATH="${PLATE_MODEL_PATH:-$REPO_DIR/models/yolov8n-license-plate.pt}"

mkdir -p "$OUT_ROOT"

# ===== Mount Drive =====
python - <<'PY'
import os
try:
    from google.colab import drive
    drive.mount(os.environ.get('DRIVE_ROOT','/content/drive'))
    print('Drive mounted')
except Exception as e:
    print('Not in Colab or mount failed:', e)
PY

# ===== Install Deps =====
# Make pip quieter and deterministic; avoid cache, speed up in Colab
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# 0) Tooling first
pip -q install -U pip wheel setuptools || true

# 1) Proactively remove conflicting numpy/opencv variants
pip -q uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless || true

# 2) Install pinned numpy/opencv known-good pair (works with ultralytics on Colab)
pip -q install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78" || true

# 3) PyTorch may already exist in Colab; install if missing
python - <<'PY'
try:
    import torch
    print('Torch exists:', torch.__version__)
except Exception:
    import os
    os.system('pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
PY

# 4) Core Python deps
pip -q install ultralytics insightface onnxruntime-gpu onnxruntime pycocotools lpips pytorch-fid easyocr matplotlib seaborn pandas tqdm pyyaml pillow jedi>=0.18.2 || true

# 5) Project requirements (if any)
if [[ -f "$REPO_DIR/requirements.txt" ]]; then
  pip -q install -r "$REPO_DIR/requirements.txt" || true
fi

# 6) Re-pin numpy/opencv after requirements to prevent accidental upgrades
pip -q install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78" || true

# 7) Sanity check: ensure numpy/cv2 can import with pinned versions
python - <<'PY'
import numpy as np
print('NUMPY VERSION:', np.__version__)
try:
    import cv2
    print('CV2 VERSION:', cv2.__version__)
except Exception as e:
    print('[ERROR] cv2 import failed:', e)
PY

# Optional baselines
if [[ "$ENABLE_DEEPPRIVACY2" == "1" ]]; then
  pip -q install deepprivacy2 || {
    git clone https://github.com/hukkelas/DeepPrivacy2 /content/DeepPrivacy2 || true
    pip -q install -e /content/DeepPrivacy2 || true
  }
fi
if [[ "$ENABLE_LDFA" == "1" ]]; then
  git clone https://github.com/yzhang1918/LDFA /content/LDFA || true
  if [[ -d /content/LDFA ]]; then
    pip -q install -e /content/LDFA || true
  fi
fi

# ===== Detection to COCO JSON =====
mkdir -p "$REPO_DIR/outputs/preds/$DETECTOR_NAME"

# Prepare joined anon dirs for Python child process (env vars cannot carry arrays)
if [[ -n "${ANON_DIRS:-}" ]]; then
    export ANON_DIRS_JOINED="$ANON_DIRS"
else
    export ANON_DIRS_JOINED="$(IFS='|'; echo "${ANON_DIRS_ARR[*]}")"
fi
python - <<PY
import os, json, glob, math, gc
from pathlib import Path
from PIL import Image
import numpy as np
import torch

REPO_DIR = os.environ['REPO_DIR']
GT = os.environ['GT_ANNOTATIONS']
ORIG_DIR = os.environ['ORIG_DIR']
_join = os.environ.get('ANON_DIRS_JOINED') or os.environ.get('ANON_DIRS') or ''
ANON_DIRS = [p for p in _join.split('|') if p]
OUT_ROOT = os.environ['OUT_ROOT']
DET = os.environ['DETECTOR_NAME']
PLATE_MODEL_PATH = os.environ.get('PLATE_MODEL_PATH','')
MAX_IMAGES = int(os.environ['MAX_IMAGES'])
BS = int(os.environ['BATCH_SIZE_DET'])

with open(GT,'r') as f:
    coco = json.load(f)
id_map = {os.path.basename(im['file_name']): im['id'] for im in coco.get('images',[])}

# Auto-detect category ids from GT categories
FACE_CAT_ID, PLATE_CAT_ID = None, None
cats = coco.get('categories', []) or []
def _find_cat(tokens):
    for c in cats:
        name = (c.get('name') or '').lower()
        if any(t in name for t in tokens):
            try:
                return int(c.get('id'))
            except Exception:
                pass
    return None
f_id = _find_cat(['face'])
p_id = _find_cat(['plate','license','licence'])
if f_id is not None:
    FACE_CAT_ID = f_id
if p_id is not None:
    PLATE_CAT_ID = p_id
if PLATE_CAT_ID is None and len(cats) == 1:
    try:
        PLATE_CAT_ID = int(cats[0].get('id'))
    except Exception:
        pass

from pathlib import Path

def paths(d):
    ps = sorted([str(p) for p in Path(d).glob('**/*') if p.suffix.lower() in ['.jpg','.jpeg','.png']])
    return ps[:MAX_IMAGES] if MAX_IMAGES and len(ps)>MAX_IMAGES else ps

def run_retinaface(imgs, batch_size=8):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640,640))
    out=[]
    for p in imgs:
        try:
            arr = np.array(Image.open(p).convert('RGB'))
        except Exception:
            continue
        faces = app.get(arr)
        dets=[]
        for f in faces:
            x1,y1,x2,y2 = f.bbox.astype(float).tolist()
            score = float(getattr(f,'det_score', getattr(f,'score',0.0)))
            if FACE_CAT_ID is None:
                continue
            dets.append({'bbox':[x1,y1,x2-x1,y2-y1], 'score':score, 'category_id':int(FACE_CAT_ID)})
        out.append((p,dets))
    return out

def chunked(lst, n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

def run_retinaface(imgs, batch_size=8):
    """Face detection via RetinaFace (insightface) to avoid local YOLO face weight dependency."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640,640))
    out=[]
    for p in imgs:
        try:
            arr = np.array(Image.open(p).convert('RGB'))
        except Exception:
            continue
        faces = app.get(arr)
        dets=[]
        for f in faces:
            x1,y1,x2,y2 = f.bbox.astype(float).tolist()
            score = float(getattr(f,'det_score', getattr(f,'score',0.0)))
            if FACE_CAT_ID is None:
                continue
            dets.append({'bbox':[x1,y1,x2-x1,y2-y1], 'score':score, 'category_id':int(FACE_CAT_ID)})
        out.append((p,dets))
    return out

def run_yolo(imgs, kind='face', bs=16, impl='v8'):
    from ultralytics import YOLO
    try:
        if kind=='plate':
            if PLATE_MODEL_PATH and os.path.isfile(PLATE_MODEL_PATH):
                model = YOLO(PLATE_MODEL_PATH)
            else:
                if impl == 'v11':
                    raise RuntimeError('YOLOv11 plate requires a local PLATE_MODEL_PATH; none found')
                tried = []
                for ref in ['keremberke/yolov8n-license-plate.pt', 'keremberke/yolov8m-license-plate.pt', 'keremberke/yolov8n-license-plate']:
                    try:
                        model = YOLO(ref)
                        break
                    except Exception as e:
                        tried.append((ref, str(e)))
                        model = None
                if model is None:
                    raise RuntimeError(f"No plate model available; tried: {tried}")
            cat = int(PLATE_CAT_ID) if PLATE_CAT_ID is not None else 2
        else:
            # Fallback generic model if ever used for faces (we avoid this path by default)
            model = YOLO('yolo11n.pt' if impl=='v11' else 'yolov8n.pt')
            cat = int(FACE_CAT_ID) if FACE_CAT_ID is not None else 1
    except Exception as e:
        if kind=='plate':
            print('[WARN] Plate model unavailable, skipping plates. You can place a local model at PLATE_MODEL_PATH=', PLATE_MODEL_PATH, 'Error:', e)
            # Return empty detections to continue pipeline
            return [(p, []) for p in imgs]
        else:
            raise
    dev = 0 if torch.cuda.is_available() else 'cpu'
    res=[]
    while True:
        try:
            for g in chunked(imgs, bs):
                r = model.predict(g, imgsz=640, device=dev, conf=0.25, iou=0.5, verbose=False)
                for p, pr in zip(g, r):
                    dets=[]
                    if pr and pr.boxes is not None:
                        for b in pr.boxes:
                            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                            score = float(b.conf[0].item()) if hasattr(b,'conf') else 0.0
                            dets.append({'bbox':[x1,y1,x2-x1,y2-y1], 'score':score, 'category_id':cat})
                    res.append((p,dets))
            break
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e) and bs>1:
                bs = max(1, bs//2)
                print('[OOM] retry bs=', bs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise
    return res

def run_and_dump(img_dir, out_json):
    imgs = paths(img_dir)
    print(f'{DET} on {len(imgs)} images: {img_dir}')
    preds=[]
    if DET=='retinaface':
        dets = run_retinaface(imgs, batch_size=max(1,BS//2))
        for p, ds in dets:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET=='yolov8_face':
        # Use RetinaFace for faces to avoid dependency on local yolov8n-face.pt
        dets = run_retinaface(imgs, batch_size=max(1,BS//2))
        for p, ds in dets:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET=='yolov8_plate':
        dets = run_yolo(imgs, 'plate', bs=BS, impl='v8')
        for p, ds in dets:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET=='yolov8_combo':
        # Faces via RetinaFace, plates via YOLO
        df = dict(run_retinaface(imgs, batch_size=max(1,BS//2)))
        dp = dict(run_yolo(imgs, 'plate', bs=max(1,BS//2), impl='v8'))
        for p in imgs:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in df.get(p,[]): preds.append({'image_id': im_id, **d})
            for d in dp.get(p,[]): preds.append({'image_id': im_id, **d})
    elif DET=='yolov11_plate':
        dets = run_yolo(imgs, 'plate', bs=BS, impl='v11')
        for p, ds in dets:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET=='yolov11_combo':
        df = dict(run_retinaface(imgs, batch_size=max(1,BS//2)))
        dp = dict(run_yolo(imgs, 'plate', bs=max(1,BS//2), impl='v11'))
        for p in imgs:
            im_id = id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in df.get(p,[]): preds.append({'image_id': im_id, **d})
            for d in dp.get(p,[]): preds.append({'image_id': im_id, **d})
    else:
        raise SystemExit('Unknown DETECTOR_NAME')
    with open(out_json,'w') as f: json.dump(preds,f)
    print('Wrote', out_json, 'num', len(preds))

pred_dir = os.path.join(REPO_DIR,'outputs','preds',DET)
os.makedirs(pred_dir, exist_ok=True)
run_and_dump(ORIG_DIR, os.path.join(pred_dir, os.path.basename(os.path.normpath(ORIG_DIR))+'.json'))
for d in ANON_DIRS:
    if os.path.isdir(d):
        run_and_dump(d, os.path.join(pred_dir, os.path.basename(os.path.normpath(d))+'.json'))
PY

# ===== Evaluate (src.evaluate) =====
cd "$REPO_DIR"
if [[ ${#ANON_DIRS_ARR[@]:-0} -gt 0 ]]; then
    ANON_JOINED=$(IFS=, ; echo "${ANON_DIRS_ARR[*]}")
elif [[ -n "${ANON_DIRS_JOINED:-}" ]]; then
    ANON_JOINED="${ANON_DIRS_JOINED//|/,}"
else
    ANON_JOINED=""
fi
python -m src.evaluate --orig_dir "$ORIG_DIR" --anon_dirs "$ANON_JOINED" --gt_annotations "$GT_ANNOTATIONS" --out_root "$OUT_ROOT" --max_images $MAX_IMAGES --thresholds "$THRESHOLDS" || true

# Mirror figs
mkdir -p "$REPO_DIR/outputs/figs" || true
cp -f "$OUT_ROOT/figs"/*.png "$REPO_DIR/outputs/figs/" 2>/dev/null || true

# ===== Patch mAP if missing =====
python - <<'PY'
import os, json, pandas as pd, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
REPO_DIR=os.environ['REPO_DIR']
GT=os.environ['GT_ANNOTATIONS']
DET=os.environ['DETECTOR_NAME']
ORIG_DIR=os.environ['ORIG_DIR']
_join=os.environ.get('ANON_DIRS_JOINED') or os.environ.get('ANON_DIRS') or ''
ANON_DIRS=[p for p in _join.split('|') if p]
metrics_csv=os.path.join(REPO_DIR,'outputs','metrics_all.csv')

def cmap(gt, pred, cats=None):
    cocoGt = COCO(gt)
    cocoDt = cocoGt.loadRes(pred) if os.path.getsize(pred)>0 else COCO()
    E = COCOeval(cocoGt, cocoDt, 'bbox')
    if cats: E.params.catIds=cats
    E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0])

def name_of(p):
    return os.path.basename(os.path.normpath(p))

rows=[]
pred_root=os.path.join(REPO_DIR,'outputs','preds',DET)
preds=[(name_of(ORIG_DIR), os.path.join(pred_root, name_of(ORIG_DIR)+'.json'))]
for d in ANON_DIRS:
    if os.path.isdir(d):
        preds.append((name_of(d), os.path.join(pred_root, name_of(d)+'.json')))

with open(GT,'r') as f:
    gtj=json.load(f)
cats=gtj.get('categories',[]) or []
def find_id(tokens):
    for c in cats:
        nm=(c.get('name') or '').lower()
        if any(t in nm for t in tokens):
            try:
                return int(c.get('id'))
            except Exception:
                pass
    return None
face_id=find_id(['face'])
plate_id=find_id(['plate','license','licence'])
if plate_id is None and len(cats)==1:
    try:
        plate_id=int(cats[0].get('id'))
    except Exception:
        pass

for n,pj in preds:
    try:
        r={'name':n}
        r['map_50_95_all']=cmap(GT,pj,None)
        r['map_50_95_face']=cmap(GT,pj,[face_id]) if face_id is not None else -1.0
        r['map_50_95_plate']=cmap(GT,pj,[plate_id]) if plate_id is not None else -1.0
        rows.append(r)
    except Exception as e:
        print('[WARN] mAP failed for', n, e)

m=pd.DataFrame(rows)
if os.path.exists(metrics_csv):
    df=pd.read_csv(metrics_csv)
    if 'name' in df.columns:
        df=df.merge(m,on='name',how='left')
    else:
        df=m
else:
    df=m
os.makedirs(os.path.join(REPO_DIR,'outputs'), exist_ok=True)
df.to_csv(metrics_csv,index=False)
print('Saved', metrics_csv)
PY

# ===== DeepPrivacy2 (optional) =====
if [[ "$ENABLE_DEEPPRIVACY2" == "1" ]]; then
  mkdir -p "$REPO_DIR/outputs/deepprivacy2"
  deepprivacy2 --input_dir "$ORIG_DIR" --output_dir "$REPO_DIR/outputs/deepprivacy2" --max_images $MAX_IMAGES || true
fi

# ===== LDFA (optional) =====
if [[ "$ENABLE_LDFA" == "1" && -d /content/LDFA ]]; then
  mkdir -p "$REPO_DIR/outputs/ldfa"
  python /content/LDFA/inference.py --input "$ORIG_DIR" --output "$REPO_DIR/outputs/ldfa" --max_images $MAX_IMAGES || true
fi

# ===== Re-evaluate with extra baselines if any =====
EXTRA_DIRS=()
[[ -d "$REPO_DIR/outputs/deepprivacy2" && -n "$(ls -A "$REPO_DIR/outputs/deepprivacy2" 2>/dev/null)" ]] && EXTRA_DIRS+=("$REPO_DIR/outputs/deepprivacy2")
[[ -d "$REPO_DIR/outputs/ldfa" && -n "$(ls -A "$REPO_DIR/outputs/ldfa" 2>/dev/null)" ]] && EXTRA_DIRS+=("$REPO_DIR/outputs/ldfa")
if [[ ${#EXTRA_DIRS[@]} -gt 0 ]]; then
  ALL_DIRS=("${ANON_DIRS[@]}" "${EXTRA_DIRS[@]}")
  ALL_JOINED=$(IFS=, ; echo "${ALL_DIRS[*]}")
  python -m src.evaluate --orig_dir "$ORIG_DIR" --anon_dirs "$ALL_JOINED" --gt_annotations "$GT_ANNOTATIONS" --out_root "$OUT_ROOT" --max_images $MAX_IMAGES --thresholds "$THRESHOLDS" || true
  mkdir -p "$REPO_DIR/outputs/figs" && cp -f "$OUT_ROOT/figs"/*.png "$REPO_DIR/outputs/figs/" 2>/dev/null || true
fi

# ===== Zip artifacts =====
cd "$REPO_DIR"
rm -f "$REPO_DIR/outputs_colab.zip" 2>/dev/null || true
zip -rq "$REPO_DIR/outputs_colab.zip" outputs || true

echo "Artifacts:"
echo "$REPO_DIR/outputs/metrics_all.csv"
echo "$REPO_DIR/outputs/figs/"
echo "$OUT_ROOT/eval_report.json"
echo "$OUT_ROOT/eval_report.md"
echo "$REPO_DIR/outputs/preds/$DETECTOR_NAME/"
echo "$REPO_DIR/outputs_colab.zip"
