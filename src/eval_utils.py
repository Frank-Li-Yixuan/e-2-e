import argparse
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
import numpy as np
from PIL import Image

from .detector_wrapper import DetectorWrapper
from .losses import arcface_similarity


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_arcface(images_dir: str, anonym_dir: str, max_images: Optional[int] = None) -> Optional[float]:
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if max_images:
        files = files[:max_images]
    sims = []
    for fn in files:
        p1 = os.path.join(images_dir, fn)
        p2 = os.path.join(anonym_dir, fn)
        if not os.path.exists(p2):
            continue
        a = torch.from_numpy(np.array(Image.open(p1).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        b = torch.from_numpy(np.array(Image.open(p2).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        sim = arcface_similarity(a.unsqueeze(0), b.unsqueeze(0))
        if sim is None:
            return None
        sims.append(float(sim.mean().cpu()))
    if not sims:
        return None
    return float(np.mean(sims))


def eval_easyocr(anonym_dir: str, max_images: Optional[int] = None) -> Optional[float]:
    try:
        import easyocr  # type: ignore
    except Exception:
        return None
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    files = [f for f in os.listdir(anonym_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if max_images:
        files = files[:max_images]
    total_words = 0
    for fn in files:
        p = os.path.join(anonym_dir, fn)
        res = reader.readtext(p)
        total_words += len(res)
    return float(total_words) / max(1, len(files))


def eval_fid(real_dir: str, fake_dir: str, max_images: Optional[int] = None) -> Optional[float]:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths  # type: ignore
    except Exception:
        return None
    # write temp lists limited by max_images by copying subset directories if necessary is heavy.
    # For simplicity, compute on all available images; caller can control quantity.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use batch_size=1 to avoid variable-size batching issues on Windows/PIL
    fid = calculate_fid_given_paths([real_dir, fake_dir], batch_size=1, device=device, dims=2048)
    return float(fid)


def _list_common_files(dir_a: str, dir_b: str, max_images: Optional[int] = None) -> List[str]:
    files = [f for f in os.listdir(dir_a) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    files = [f for f in files if os.path.exists(os.path.join(dir_b, f))]
    if max_images:
        files = files[:max_images]
    return files


@torch.no_grad()
def eval_lpips(orig_dir: str, anon_dir: str, max_images: Optional[int] = None) -> Optional[float]:
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    loss_fn = lpips.LPIPS(net='alex')
    files = _list_common_files(orig_dir, anon_dir, max_images)
    if not files:
        return None
    vals: List[float] = []
    for fn in files:
        p1 = os.path.join(orig_dir, fn)
        p2 = os.path.join(anon_dir, fn)
        a = torch.from_numpy(np.array(Image.open(p1).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        b = torch.from_numpy(np.array(Image.open(p2).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        # LPIPS expects [-1,1]
        a = a * 2 - 1
        b = b * 2 - 1
        v = float(loss_fn(a.unsqueeze(0), b.unsqueeze(0)).mean().item())
        vals.append(v)
    return float(np.mean(vals)) if vals else None


def _psnr_ssim_pair(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # type: ignore
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
    # convert to Y channel SSIM to be robust
    if a.ndim == 3:
        a_gray = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]).astype(np.float32)
        b_gray = (0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]).astype(np.float32)
    else:
        a_gray = a.astype(np.float32)
        b_gray = b.astype(np.float32)
    ssim = float(structural_similarity(a_gray, b_gray, data_range=255))
    return psnr, ssim


@torch.no_grad()
def eval_psnr_ssim(orig_dir: str, anon_dir: str, max_images: Optional[int] = None) -> Tuple[Optional[float], Optional[float]]:
    files = _list_common_files(orig_dir, anon_dir, max_images)
    if not files:
        return None, None
    psnrs: List[float] = []
    ssims: List[float] = []
    for fn in files:
        p1 = os.path.join(orig_dir, fn)
        p2 = os.path.join(anon_dir, fn)
        a = np.array(Image.open(p1).convert("RGB"))
        b = np.array(Image.open(p2).convert("RGB"))
        p, s = _psnr_ssim_pair(a, b)
        psnrs.append(p)
        ssims.append(s)
    return (float(np.mean(psnrs)) if psnrs else None, float(np.mean(ssims)) if ssims else None)


def eval_map(cfg: Dict[str, Any], images_dir: str, gt_annotations: str, max_images: Optional[int] = None) -> Optional[float]:
    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception:
        return None
    coco_gt = COCO(gt_annotations)
    # Ensure required keys exist for loadRes
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []
    det = DetectorWrapper(cfg)
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if max_images:
        files = files[:max_images]
    results = []
    img_ids = []
    for fn in files:
        p = os.path.join(images_dir, fn)
        img = Image.open(p).convert("RGB")
        w, h = img.size
        x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        x = x * 2 - 1
        out = det.predict(x.unsqueeze(0))[0]
        boxes = out["boxes"].numpy().tolist()
        scores = out["scores"].numpy().tolist()
        labels = out["labels"].numpy().tolist()
        # find image id in COCO GT by filename
        # match either exact file_name or by basename fallback
        img_id = None
        base = os.path.basename(fn)
        for img_entry in coco_gt.dataset.get("images", []):
            file_name = img_entry.get("file_name")
            if file_name == fn or os.path.basename(file_name) == base:
                img_id = img_entry.get("id")
                break
        if img_id is None:
            continue
        img_ids.append(img_id)
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b
            results.append({
                "image_id": img_id,
                "category_id": int(l),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(s),
            })
    if len(results) == 0:
        return None
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    # Optional: evaluate face-only if configured
    eval_cfg = cfg.get('eval', {}) if isinstance(cfg, dict) else {}
    if isinstance(eval_cfg, dict) and eval_cfg.get('face_only', False):
        coco_eval.params.catIds = [1]
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])  # AP@[.5:.95]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Mode A: config + dirs (backward compatible)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--images", type=str, default=None, help="Real/original images directory")
    parser.add_argument("--anonym", type=str, default=None, help="Anonymized images directory (for FID/ArcFace/EasyOCR)")
    parser.add_argument("--gt_annotations", type=str, default=None, help="COCO JSON for mAP evaluation")
    parser.add_argument("--max_images", type=int, default=None)
    # Mode B: direct dirs + explicit out
    parser.add_argument("--orig_dir", type=str, default=None)
    parser.add_argument("--anon_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Output JSON path; will also write a Markdown next to it")
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "arcface_mean_sim": None,
        "easyocr_avg_words": None,
        "fid": None,
        "lpips": None,
        "psnr": None,
        "ssim": None,
        "map": None,
    }

    if args.orig_dir and args.anon_dir:
        images_dir = args.orig_dir
        anonym_dir = args.anon_dir
        cfg = load_config(args.config) if args.config else {"paths": {}}
        sim = eval_arcface(images_dir, anonym_dir, args.max_images)
        results["arcface_mean_sim"] = float(sim) if sim is not None else None
        ocr = eval_easyocr(anonym_dir, args.max_images)
        results["easyocr_avg_words"] = float(ocr) if ocr is not None else None
        fid_v = eval_fid(images_dir, anonym_dir, args.max_images)
        results["fid"] = float(fid_v) if fid_v is not None else None
        lp = eval_lpips(images_dir, anonym_dir, args.max_images)
        results["lpips"] = float(lp) if lp is not None else None
        p, s = eval_psnr_ssim(images_dir, anonym_dir, args.max_images)
        results["psnr"] = float(p) if p is not None else None
        results["ssim"] = float(s) if s is not None else None
        if args.gt_annotations:
            m = eval_map(cfg, anonym_dir, args.gt_annotations, args.max_images)
            results["map"] = float(m) if m is not None else None
        print("Eval:", results)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(results, f)
            md_path = os.path.splitext(args.out)[0] + ".md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("# Eval Report\n\n")
                for k, v in results.items():
                    f.write(f"- {k}: {v if v is not None else 'NaN'}\n")
        raise SystemExit(0)

    # Backward compatible flow
    if not (args.config and args.images and args.anonym):
        print("[ERROR] Missing arguments. Provide either --orig_dir/--anon_dir/--out or --config/--images/--anonym")
        raise SystemExit(2)
    cfg = load_config(args.config)
    sim = eval_arcface(args.images, args.anonym, args.max_images)
    print("ArcFace cosine similarity (lower=better):", sim)
    ocr = eval_easyocr(args.anonym, args.max_images)
    print("EasyOCR avg words/image (lower=better):", ocr)
    fid = eval_fid(args.images, args.anonym, args.max_images)
    print("FID (lower=better):", fid)
    if args.gt_annotations:
        m = eval_map(cfg, args.anonym or args.images, args.gt_annotations, args.max_images)
        print("Downstream mAP (AP@[.5:.95]) on anonymized images:", m)
