import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageFilter


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_coco(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def boxes_by_image(coco: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for a in coco.get('annotations', []):
        out.setdefault(a['image_id'], []).append(a)
    return out


def apply_pixelate(img: Image.Image, box: Tuple[float, float, float, float], pixel: int) -> None:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    x, y, x2, y2 = map(int, [x, y, x2, y2])
    x = max(0, x); y = max(0, y)
    x2 = min(img.width, x2); y2 = min(img.height, y2)
    if x2 <= x or y2 <= y:
        return
    crop = img.crop((x, y, x2, y2))
    # downsample then upsample with nearest
    dw = max(1, (x2 - x) // max(1, pixel))
    dh = max(1, (y2 - y) // max(1, pixel))
    small = crop.resize((dw, dh), resample=Image.NEAREST)
    big = small.resize((x2 - x, y2 - y), resample=Image.NEAREST)
    img.paste(big, (x, y))


def apply_mask_out(img: Image.Image, box: Tuple[float, float, float, float], color=(0, 0, 0)) -> None:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    x, y, x2, y2 = map(int, [x, y, x2, y2])
    x = max(0, x); y = max(0, y)
    x2 = min(img.width, x2); y2 = min(img.height, y2)
    if x2 <= x or y2 <= y:
        return
    overlay = Image.new('RGB', (x2 - x, y2 - y), color=color)
    img.paste(overlay, (x, y))


def apply_gaussian(img: Image.Image, box: Tuple[float, float, float, float], ksize: int) -> None:
    x, y, w, h = box
    x2, y2 = x + w, y + h
    x, y, x2, y2 = map(int, [x, y, x2, y2])
    x = max(0, x); y = max(0, y)
    x2 = min(img.width, x2); y2 = min(img.height, y2)
    if x2 <= x or y2 <= y:
        return
    crop = img.crop((x, y, x2, y2))
    radius = max(0.1, ksize / 2.0)
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=radius))
    img.paste(blurred, (x, y))


def process(
    input_json: str,
    images_root: str,
    out_dir: str,
    method: str,
    pixel: int = 8,
    ksize: int = 7,
    max_images: int = 0,
    save_orig_dir: str = "",
    log_path: str = "",
) -> None:
    ensure_dir(out_dir)
    if save_orig_dir:
        ensure_dir(save_orig_dir)
    coco = load_coco(input_json)
    by_img = boxes_by_image(coco)
    images = coco.get('images', [])
    start = time.time()
    n_done = 0
    for img_entry in images:
        if max_images and n_done >= max_images:
            break
        file_name = img_entry['file_name']
        in_path = os.path.normpath(os.path.join(images_root, file_name))
        if not os.path.exists(in_path):
            # try basename fallback
            base = os.path.basename(file_name)
            in_path = os.path.normpath(os.path.join(images_root, base))
            if not os.path.exists(in_path):
                continue
        try:
            img = Image.open(in_path).convert('RGB')
        except Exception:
            continue
        boxes = [a['bbox'] for a in by_img.get(img_entry['id'], [])]

        out_img = img.copy()
        for b in boxes:
            if method == 'pixelation':
                apply_pixelate(out_img, tuple(b), pixel)
            elif method == 'mask':
                apply_mask_out(out_img, tuple(b))
            elif method == 'gaussian':
                apply_gaussian(out_img, tuple(b), ksize)
            else:
                raise ValueError(f"Unknown method: {method}")

        base = os.path.basename(file_name)
        out_path = os.path.join(out_dir, base)
        out_img.save(out_path)
        if save_orig_dir:
            orig_path = os.path.join(save_orig_dir, base)
            if not os.path.exists(orig_path):
                img.save(orig_path)
        n_done += 1
    secs = time.time() - start
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"method={method}\n")
            if method == 'pixelation':
                f.write(f"pixel={pixel}\n")
            if method == 'gaussian':
                f.write(f"ksize={ksize}\n")
            f.write(f"images={n_done}\n")
            f.write(f"seconds={secs:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_json', required=True)
    ap.add_argument('--images_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--method', required=True, choices=['pixelation','mask','gaussian'])
    ap.add_argument('--pixel', type=int, default=8)
    ap.add_argument('--ksize', type=int, default=7)
    ap.add_argument('--max_images', type=int, default=0)
    ap.add_argument('--save_orig_dir', type=str, default='')
    ap.add_argument('--log', type=str, default='')
    args = ap.parse_args()
    process(
        input_json=args.input_json,
        images_root=args.images_root,
        out_dir=args.out_dir,
        method=args.method,
        pixel=args.pixel,
        ksize=args.ksize,
        max_images=args.max_images,
        save_orig_dir=args.save_orig_dir,
        log_path=args.log,
    )


if __name__ == '__main__':
    main()
