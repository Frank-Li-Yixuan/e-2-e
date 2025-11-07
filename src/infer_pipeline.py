import argparse
import os
from typing import Any, Dict, List

import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw

from .detector_wrapper import DetectorWrapper
from .generator_wrapper import GeneratorWrapper


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
    x = x * 2 - 1
    return x


def to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).detach().cpu().numpy()
    x = (x * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def boxes_to_mask(size: tuple, boxes: torch.Tensor, dilate: int = 8) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b]
        x1 = max(0, x1 - dilate)
        y1 = max(0, y1 - dilate)
        x2 = min(w - 1, x2 + dilate)
        y2 = min(h - 1, y2 + dilate)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def run_infer(cfg: Dict[str, Any], input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_masks = bool(cfg.get("inference", {}).get("save_masks", True))
    save_overlays = bool(cfg.get("inference", {}).get("save_overlays", True))

    # instantiate
    det = DetectorWrapper(cfg)
    gen = GeneratorWrapper(cfg)

    fnames = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    for fn in fnames:
        in_path = os.path.join(input_dir, fn)
        img = Image.open(in_path).convert("RGB")
        w, h = img.size
        # prepare model input
        x = to_tensor(img).unsqueeze(0)
        # detector on resized space (dataset normalization) — keep size the same for simplicity
        preds = det.predict(x)
        boxes = preds[0]["boxes"]
        # build mask
        mask_pil = boxes_to_mask((w, h), boxes, dilate=8)
        mask_t = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        x_resized = x  # operate at native size; for large images consider resizing
        mask_resized = mask_t
        # generate
        with torch.no_grad():
            y = gen(x_resized, mask_resized)
        out_img = to_pil(y[0])

        # save
        out_path = os.path.join(output_dir, fn)
        out_img.save(out_path)
        if save_masks:
            mask_pil.save(os.path.join(output_dir, os.path.splitext(fn)[0] + "_mask.png"))
        if save_overlays:
            overlay = img.copy()
            overlay_draw = ImageDraw.Draw(overlay, "RGBA")
            overlay_draw.bitmap((0, 0), mask_pil.convert("1"), fill=(255, 0, 0, 80))
            overlay.save(os.path.join(output_dir, os.path.splitext(fn)[0] + "_overlay.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_infer(cfg, args.input, args.output)
