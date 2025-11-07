#!/usr/bin/env python
import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure repo root on sys.path so `src` package resolves when running as a script
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.detector_wrapper import DetectorWrapper
from src.generator_wrapper import GeneratorWrapper
import torch.nn.functional as F

def to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    x = x * 2 - 1
    return x


def to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy()
    x = (x * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def boxes_to_mask_xyxy(size: tuple, boxes_xyxy: List[List[float]], dilate: int = 8) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(0.0, x1 - dilate)
        y1 = max(0.0, y1 - dilate)
        x2 = min(w - 1.0, x2 + dilate)
        y2 = min(h - 1.0, y2 + dilate)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def run_infer_from_coco_e2e(cfg: Dict[str, Any], annotations: str, images_root: str, output_dir: str, max_images: int = 200, checkpoint: str | None = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # index images
    images = coco.get('images', [])[:max_images]

    det = DetectorWrapper(cfg)
    gen = GeneratorWrapper(cfg)
    # optional resume
    if checkpoint and os.path.exists(checkpoint):
        try:
            state = torch.load(checkpoint, map_location="cpu")
            g_state = state.get("generator", state)
            missing, unexpected = gen.load_state_dict(g_state, strict=False)
            print(f"[INFO] Loaded generator weights from {checkpoint}; missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {checkpoint}: {e}")

    for img_entry in images:
        file_name = img_entry['file_name']
        in_path = os.path.normpath(os.path.join(images_root, file_name))
        if not os.path.exists(in_path):
            continue
        img = Image.open(in_path).convert('RGB')
        w, h = img.size
        x = to_tensor(img).unsqueeze(0)

        # detector predicts boxes in resized/normalized space; DetectorWrapper handles internal preprocessing
        preds = det.predict(x)
        boxes = preds[0]['boxes']  # tensor [N,4] xyxy

        # build mask on original size
        boxes_list = boxes.detach().cpu().numpy().tolist()
        mask_pil = boxes_to_mask_xyxy((w, h), boxes_list, dilate=8)
        mask_t = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        # resize to training size for generator if needed
        target_size = int(cfg.get('data', {}).get('image_size', 256))
        x_rs = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        m_rs = F.interpolate(mask_t, size=(target_size, target_size), mode='nearest')
        with torch.no_grad():
            y = gen(x_rs, m_rs)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        out = to_pil(y[0])

        out_path = os.path.join(output_dir, os.path.basename(file_name))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.save(out_path)
        # also save mask and overlay for QA
        mask_pil.save(os.path.join(output_dir, os.path.splitext(os.path.basename(file_name))[0] + '_mask.png'))
        overlay = img.copy()
        overlay_draw = ImageDraw.Draw(overlay, 'RGBA')
        overlay_draw.bitmap((0, 0), mask_pil.convert('1'), fill=(255, 0, 0, 80))
        overlay.save(os.path.join(output_dir, os.path.splitext(os.path.basename(file_name))[0] + '_overlay.png'))


if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=200)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # allow config inference.generator_checkpoint as default
    ckpt = args.checkpoint or cfg.get('inference', {}).get('generator_checkpoint')
    run_infer_from_coco_e2e(cfg, args.annotations, args.images_root, args.output, args.max_images, ckpt)
