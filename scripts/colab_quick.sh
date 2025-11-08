#!/usr/bin/env bash
set -euo pipefail
set -a

# ===== Minimal, robust Colab runner (mAP only) =====
# Config via env, with sane defaults
REPO_DIR="${REPO_DIR:-/content/drive/MyDrive/anony-project}"
GT_ANNOTATIONS="${GT_ANNOTATIONS:-$REPO_DIR/data/unified/unified_val.json}"
ORIG_DIR="${ORIG_DIR:-$REPO_DIR/outputs/baselines/orig}"
# Try a lean anon set; non-existing will be skipped
ANON_DIRS_ARR=(
  "$REPO_DIR/outputs/baselines/pixelation_16x16"
  "$REPO_DIR/outputs/baselines/gaussian_k15"
)
MAX_IMAGES=${MAX_IMAGES:-200}
BATCH_SIZE_DET=${BATCH_SIZE_DET:-16}
DETECTOR_NAME="${DETECTOR_NAME:-yolov8_combo}"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/outputs/eval_runs/colab_quick}"

mkdir -p "$OUT_ROOT" "$REPO_DIR/outputs/preds/$DETECTOR_NAME" "$REPO_DIR/outputs/figs"

# ===== Fix ABI and install only what we need =====
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
pip -q uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless || true
pip -q install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78"
# Torch may exist in Colab
python - <<'PY'
try:
    import torch
    print('Torch exists:', torch.__version__)
except Exception:
    import os
    os.system('pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
PY
pip -q install ultralytics pycocotools pillow pandas tqdm insightface onnxruntime-gpu onnxruntime || true
# Re-pin to avoid accidental upgrades
pip -q install --upgrade --force-reinstall "numpy<2.0" "opencv-python-headless==4.8.1.78"

# Sanity print
python - <<'PY'
import numpy as np
print('NUMPY VERSION:', np.__version__)
try:
    import cv2
    print('CV2 VERSION:', cv2.__version__)
except Exception as e:
    print('[ERROR] cv2 import failed:', e)
PY

# Prepare joined anon dirs string for child python
export ANON_DIRS_JOINED="$(IFS='|'; echo "${ANON_DIRS_ARR[*]}")"

# ===== Detection (YOLOv8 face+plate or single) =====
python - <<PY
import os, json, gc
from pathlib import Path
from PIL import Image
import numpy as np
import torch

REPO_DIR=os.environ['REPO_DIR']
GT=os.environ['GT_ANNOTATIONS']
ORIG_DIR=os.environ['ORIG_DIR']
ANON_DIRS=[p for p in os.environ.get('ANON_DIRS_JOINED','').split('|') if p]
OUT_ROOT=os.environ['OUT_ROOT']
DET=os.environ['DETECTOR_NAME']
MAX_IMAGES=int(os.environ['MAX_IMAGES'])
BS=int(os.environ['BATCH_SIZE_DET'])

with open(GT,'r') as f:
    coco=json.load(f)
id_map={os.path.basename(im['file_name']): im['id'] for im in coco.get('images',[])}

def paths(d):
    ps=sorted([str(p) for p in Path(d).glob('**/*') if p.suffix.lower() in ['.jpg','.jpeg','.png']])
    return ps[:MAX_IMAGES] if MAX_IMAGES and len(ps)>MAX_IMAGES else ps

def chunked(lst, n):
    for i in range(0,len(lst),n):
        yield lst[i:i+n]

# Face detector using RetinaFace (insightface), robust without external weights files
def run_retinaface(imgs, batch_size=8):
    from insightface.app import FaceAnalysis
    app=FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640,640))
    out=[]
    for p in imgs:
        try:
            arr=np.array(Image.open(p).convert('RGB'))
        except Exception:
            continue
        faces=app.get(arr)
        dets=[]
        for f in faces:
            x1,y1,x2,y2 = f.bbox.astype(float).tolist()
            score=float(getattr(f,'det_score', getattr(f,'score',0.0)))
            dets.append({'bbox':[x1,y1,x2-x1,y2-y1], 'score':score, 'category_id':1})
        out.append((p,dets))
    return out

def run_yolo(imgs, kind='face', bs=16):
    from ultralytics import YOLO
    if kind=='plate':
        model=None
        tried=[]
        for ref in ['keremberke/yolov8n-license-plate.pt','keremberke/yolov8m-license-plate.pt','keremberke/yolov8n-license-plate']:
            try:
                model=YOLO(ref); break
            except Exception as e:
                tried.append((ref,str(e)))
        if model is None:
            print('[WARN] Plate weights unavailable; tried:', tried)
            return [(p,[]) for p in imgs]
        cat=2
    else:
        # We avoid relying on local 'yolov8n-face.pt' file to prevent FileNotFoundError in restricted envs
        # Use RetinaFace path for faces instead; this branch will not be used for 'face'
        model=YOLO('yolov8n.pt'); cat=1  # fallback, not expected for faces in quick script
    dev=0 if torch.cuda.is_available() else 'cpu'
    res=[]
    while True:
        try:
            for g in chunked(imgs, bs):
                r=model.predict(g, imgsz=640, device=dev, conf=0.25, iou=0.5, verbose=False)
                for p, pr in zip(g, r):
                    dets=[]
                    if pr and pr.boxes is not None:
                        for b in pr.boxes:
                            x1,y1,x2,y2=map(float, b.xyxy[0].tolist())
                            score=float(b.conf[0].item()) if hasattr(b,'conf') else 0.0
                            dets.append({'bbox':[x1,y1,x2-x1,y2-y1], 'score':score, 'category_id':cat})
                    res.append((p,dets))
            break
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e) and bs>1:
                bs=max(1, bs//2)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect(); continue
            else:
                raise
    return res

def run_and_dump(img_dir, out_json):
    imgs=paths(img_dir)
    print(f'{DET} on {len(imgs)} images: {img_dir}')
    preds=[]
    if DET=='yolov8_combo':
        # Use RetinaFace for faces to avoid missing yolov8n-face.pt; YOLO for plates
        df=dict(run_retinaface(imgs, batch_size=max(1,BS//2)))
        dp=dict(run_yolo(imgs,'plate',bs=max(1,BS//2)))
        for p in imgs:
            im_id=id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in df.get(p,[]): preds.append({'image_id': im_id, **d})
            for d in dp.get(p,[]): preds.append({'image_id': im_id, **d})
    elif DET=='yolov8_face':
        # Map face-only to RetinaFace for robustness
        for p,ds in run_retinaface(imgs, batch_size=max(1,BS//2)):
            im_id=id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    elif DET=='yolov8_plate':
        for p,ds in run_yolo(imgs,'plate',bs=BS):
            im_id=id_map.get(os.path.basename(p));
            if im_id is None: continue
            for d in ds: preds.append({'image_id': im_id, **d})
    else:
        raise SystemExit('Unsupported DETECTOR_NAME for quick runner')
    os.makedirs(os.path.join(REPO_DIR,'outputs','preds',DET), exist_ok=True)
    with open(out_json,'w') as f: json.dump(preds,f)
    print('Wrote', out_json, 'num', len(preds))

pred_dir=os.path.join(REPO_DIR,'outputs','preds',DET)
os.makedirs(pred_dir, exist_ok=True)
run_and_dump(ORIG_DIR, os.path.join(pred_dir, os.path.basename(os.path.normpath(ORIG_DIR))+'.json'))
for d in ANON_DIRS:
    if os.path.isdir(d):
        run_and_dump(d, os.path.join(pred_dir, os.path.basename(os.path.normpath(d))+'.json'))
PY

# ===== Minimal mAP evaluation (COCOeval) =====
python - <<'PY'
import os, json, pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_DIR=os.environ['REPO_DIR']
GT=os.environ['GT_ANNOTATIONS']
DET=os.environ['DETECTOR_NAME']
ORIG_DIR=os.environ['ORIG_DIR']
ANON_DIRS=[p for p in os.environ.get('ANON_DIRS_JOINED','').split('|') if p]
metrics_csv=os.path.join(REPO_DIR,'outputs','metrics_all.csv')

cocoGt=COCO(GT)

def eval_pred(pred):
    cocoDt=cocoGt.loadRes(pred) if os.path.getsize(pred)>0 else COCO()
    E=COCOeval(cocoGt, cocoDt, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    all=float(E.stats[0])
    # per category
    def cm(cat_id):
        e=COCOeval(cocoGt, cocoDt, 'bbox'); e.params.catIds=[cat_id]; e.evaluate(); e.accumulate(); e.summarize(); return float(e.stats[0])
    face=cm(1); plate=cm(2)
    return all, face, plate

rows=[]
pred_root=os.path.join(REPO_DIR,'outputs','preds',DET)

def name_of(p):
    return os.path.basename(os.path.normpath(p))

preds=[(name_of(ORIG_DIR), os.path.join(pred_root, name_of(ORIG_DIR)+'.json'))]
for d in ANON_DIRS:
    if os.path.isdir(d):
        preds.append((name_of(d), os.path.join(pred_root, name_of(d)+'.json')))

for n,pj in preds:
    try:
        a,f,p=eval_pred(pj)
        rows.append({'name':n,'map_50_95_all':a,'map_50_95_face':f,'map_50_95_plate':p})
    except Exception as e:
        print('[WARN] mAP failed for', n, e)

import pandas as pd
import os
os.makedirs(os.path.join(REPO_DIR,'outputs'), exist_ok=True)
if os.path.exists(metrics_csv):
    df=pd.read_csv(metrics_csv)
    df=df.drop(columns=[c for c in ['map_50_95_all','map_50_95_face','map_50_95_plate'] if c in df.columns], errors='ignore')
    m=pd.DataFrame(rows)
    # merge or concat on name
    if 'name' in df.columns:
        out=df.merge(m,on='name',how='outer')
    else:
        out=m
else:
    out=pd.DataFrame(rows)
out.to_csv(metrics_csv,index=False)
print('Saved', metrics_csv)
PY

# Mirror figs placeholder (no figs in quick run)

# Summary
echo "Artifacts:"
echo "$REPO_DIR/outputs/preds/$DETECTOR_NAME/"
echo "$REPO_DIR/outputs/metrics_all.csv"
