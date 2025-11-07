# Anony Project: End-to-End Anonymization (Detector → Diffusion Inpaint)

This project implements an end-to-end anonymization pipeline:

- ViT-based detector (YOLOS) or DETR to localize sensitive regions (e.g., faces)
- Diffusion inpainting (Stable Diffusion inpainting) OR a lightweight trainable U-Net inpainting generator
- Optional PatchGAN discriminator and alternating training of detector / generator / discriminator
- Supervised pretraining on pseudo-targets
- Full evaluation suite: ArcFace identity similarity, EasyOCR, FID, and downstream detection mAP
- Colab-ready (tested on A100), Windows-friendly

## Folder Structure

```
anony-project/
├─ data/                    # (empty placeholders) images/ masks/ pseudotargets/
├─ configs/
│  └─ joint_small.yaml
├─ notebooks/
│  └─ colab_train.ipynb
├─ src/
│  ├─ datasets.py           # COCO Dataset + preprocessing, transforms
│  ├─ detector_wrapper.py   # ViT / DETR wrapper (load/save/predict/compute_loss)
│  ├─ generator_wrapper.py  # Diffusers inpainting wrapper (DiT/SD inpaint) or trainable U-Net
│  ├─ discriminator.py      # optional PatchGAN discriminator
│  ├─ losses.py             # perceptual, id-suppression (ArcFace), adv loss
│  ├─ train_joint.py        # main train loop with alternating updates
│  ├─ infer_pipeline.py     # detector->generator inference helper
│  └─ eval_utils.py         # ArcFace, EasyOCR, FID, mAP evaluation scripts
├─ requirements.txt
└─ README.md
```

## Quickstart

1) Install dependencies

On Colab (recommended): the notebook installs everything for you. Locally (Windows):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Note on COCO API:
- Windows: `pycocotools-windows` is included in requirements.
- Linux/Colab: `pycocotools` is installed.

Note on xformers:
- Only supported and useful on Linux/Colab to accelerate diffusers. Skip on Windows.

2) Prepare data

Place your inputs under `data/`:
- `data/images/`            Raw input images
- `data/masks/`             Optional binary masks (H×W) aligned with images (PNG)
- `data/pseudotargets/`     Optional anonymized targets for supervised pretraining

Optionally set COCO-style annotations JSON in `configs/joint_small.yaml` under `paths.train_annotations/val_annotations`.

Dataset tools:
- Convert PP4AV to COCO:

```powershell
python scripts/pp4av_to_coco.py --images_dir <PP4AV_images_dir> --ann_dir <PP4AV_annotations_dir> --output data/pp4av_coco.json
```

- Ensure unified categories and merge multiple COCOs (face=1, license_plate=2):

```powershell
python scripts/ensure_coco_mapping.py --inputs data/pp4av_coco.json data/wider_coco.json data/ccpd_coco.json --output data/unified_coco.json --min_side 8
# 也支持通配符输入与数据根目录修复（建议将 datasets_root 指向项目父目录，包含 CCPD2020/、WiderFace/ 等）：
python scripts/ensure_coco_mapping.py --inputs data/*_coco.json --output data/unified_coco.json --min_side 8 --datasets_root ..
```

- List and configure available datasets in `data/dataset_registry.yaml`.

3) Configure

 Edit `configs/joint_small.yaml` to set:
 - Detector backend: `yolos` (ViT) or `detr`
 - Generator backend: `diffusers` (默认) 或 `unet`（轻量可训）
 - 开启 LoRA 轻量微调（默认已开启示例）：`generator.finetune.use_lora: true`
 - 设置 Drive 路径（Colab 推荐）：`paths.*`
 - Toggle discriminator, losses, alternating steps, and pretraining steps

4) Train

```powershell
python -m src.train_joint --config configs/joint_small.yaml
```

This runs pretraining to pseudo-targets (if present), then alternates detector/gen/disc updates.

Logging/metrics:
- `metrics.csv` 按步记录：step, epoch, loss_gen, loss_det, arcface_mean_sim, easyocr_plate_acc, fid, val_map。
- 如果缺少依赖（ArcFace/EasyOCR/FID/COCO），对应列为 NaN，并打印 WARN。

5) Inference

```powershell
python -m src.infer_pipeline --config configs/joint_small.yaml --input data/images --output outputs/anonymized
```

6) Evaluate

```powershell
python -m src.eval_utils --config configs/joint_small.yaml --images outputs/anonymized --gt_annotations <your_coco.json>
```

## Notes on Backends

- Detector
  - `yolos` (ViT detector via Hugging Face). Works zero-shot reasonably for generic objects; for faces, consider using face-specific datasets/labels.
  - `detr` (torchvision), robust COCO detector; you can fine-tune on your face dataset (WiderFace etc.).

- Generator
  - `unet`: Lightweight trainable inpainting model with perceptual and adversarial losses (fast to start and fully differentiable)
  - `diffusers`: Uses Stable Diffusion inpainting for high quality inference. Lightweight fine-tuning via LoRA is scaffolded.

## Colab

Open `notebooks/colab_train.ipynb` and run cells top to bottom. The notebook:
- Installs libs (torch/diffusers/transformers/etc.)
- Downloads or links your dataset from Drive
- Kicks off training with `configs/joint_small.yaml`
- Periodically evaluates and saves samples to Drive

Exact sequence and Drive mounting (first cell):

```python
from google.colab import drive
drive.mount('/content/drive')
# then edit configs/joint_small.yaml paths to /content/drive/MyDrive/...
```

## Evaluation

- ArcFace: lower cosine similarity between original crops and anonymized crops is better (identity suppression)
- EasyOCR: lower text detection or confidence is better (privacy)
- FID: lower is better (image realism)
- mAP: evaluate downstream detector performance on anonymized images

## Windows Hints

- If `xformers` fails to install, it’s optional and only used to speed up diffusers on Linux.
- If `pycocotools` fails, ensure the platform-specific package is used (see requirements.txt) or install from source via prebuilt wheels.

## Pseudotargets Generation (DeepPrivacy2 / LDFA)

使用已有匿名化工具先对训练集批量生成伪目标（2k–10k 对）：

- DeepPrivacy2 示例（参考官方文档，以下仅示意）：

```bash
# 假设 deepprivacy2 已安装
deepprivacy2.infer --input data/images --output data/pseudotargets --face-detector retinaface --device cuda
```

- LDFA（Latent Diffusion Face Anonymization）示意：

```bash
# 假设 LDFA 脚本可用
python ldfa_infer.py --input data/images --output data/pseudotargets --device cuda
```

完成后在 `configs/joint_small.yaml` 配置 `paths.pseudotargets`，训练会先进行 supervised pretrain。

## Dataset suggestions and priorities

- 优先：WIDER FACE（face 主来源）、CCPD（plate 大规模）、UFPR-ALPR/SSIG-SegPlate（plate 多样性）、CrowdHuman（遮挡/密集）。
- 使用 `scripts/ensure_coco_mapping.py` 合并为统一 COCO，并统一类别 ID：face=1, license_plate=2，避免 label mismatch。

## License

This repository aggregates third-party models (Hugging Face, diffusers, insightface). Check their licenses before commercial use.
