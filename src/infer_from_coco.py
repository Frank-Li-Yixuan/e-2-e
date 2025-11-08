import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from .generator_wrapper import GeneratorWrapper
import torch.nn.functional as F


def to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    x = x * 2 - 1
    return x


def to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy()
    x = (x * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def boxes_to_mask(size: tuple, boxes: List[List[float]], dilate: int = 8) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)
    for b in boxes:
        x, y, bw, bh = b
        x1 = max(0.0, x - dilate)
        y1 = max(0.0, y - dilate)
        x2 = min(w - 1.0, x + bw + dilate)
        y2 = min(h - 1.0, y + bh + dilate)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def _resolve_image_path(images_root: str, cfg: Dict[str, Any], file_name: str) -> Optional[str]:
    """Try multiple roots to locate the image file."""
    cand = [
        os.path.normpath(os.path.join(images_root, file_name)),
        os.path.normpath(os.path.join(str(cfg.get('paths', {}).get('images', '')), file_name)),
        os.path.normpath(file_name),
        os.path.normpath(os.path.join(os.getcwd(), file_name)),
    ]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


def run_infer_from_coco(cfg: Dict[str, Any], annotations: str, images_root: str, output_dir: str, max_images: int = 16) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # index annotations by image_id
    ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get('annotations', []):
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    gen = GeneratorWrapper(cfg)
    target_size = int(cfg.get('data', {}).get('image_size', 256))

    images = coco.get('images', [])[:max_images]
    for img_entry in images:
        try:
            file_name = img_entry['file_name']
            in_path = _resolve_image_path(images_root, cfg, file_name)
            if in_path is None:
                # skip if not found
                continue
            img = Image.open(in_path).convert('RGB')
            w, h = img.size
            anns = ann_by_img.get(img_entry['id'], [])
            boxes = [a['bbox'] for a in anns]
            if len(boxes) == 0:
                # no boxes, copy original
                out = img
            else:
                mask_pil = boxes_to_mask((w, h), boxes, dilate=8)
                mask_t = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                x = to_tensor(img).unsqueeze(0)
                # resize to training size for UNet/diffusers, then back to original
                x_rs = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
                m_rs = F.interpolate(mask_t, size=(target_size, target_size), mode='nearest')
                with torch.no_grad():
                    y = gen(x_rs, m_rs)
                    y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
                out = to_pil(y[0])
                # also save mask for reference
                mask_pil.save(os.path.join(output_dir, os.path.splitext(os.path.basename(file_name))[0] + '_mask.png'))

            out_path = os.path.join(output_dir, os.path.basename(file_name))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out.save(out_path)
        except Exception as e:
            # continue on errors to avoid aborting entire job
            err_dir = os.path.join(output_dir, '_errors')
            os.makedirs(err_dir, exist_ok=True)
            with open(os.path.join(err_dir, 'errors.txt'), 'a', encoding='utf-8') as ef:
                ef.write(f"{file_name}: {e}\n")


if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--images_root', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=16)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    run_infer_from_coco(cfg, args.annotations, args.images_root, args.output, args.max_images)
