import argparse
import os
from typing import Any, Dict, List, Optional

import torch
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import torch.nn.functional as F
import cv2
from PIL import ImageFont, ImageDraw

from .detector_wrapper import DetectorWrapper
from .generator_wrapper import GeneratorWrapper
from .backbone import build_backbone
from .conditioning import roi_features_from_boxes, fuse_global_roi

# Lazy OCR (easyocr) to assess plate clarity
_OCR_READER = None

def _lazy_init_ocr() -> Optional[Any]:  # type: ignore[return-any]
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER
    try:
        import easyocr  # type: ignore
        _OCR_READER = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:  # pragma: no cover
        print(f"[WARN] easyocr init failed: {e}")
        _OCR_READER = None
    return _OCR_READER


def _get_diffusers_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dcfg = cfg.get('model', {}).get('generator', {}).get('diffusers', {})
    return {
        'mask_dilate': int(dcfg.get('mask_dilate', 0)),
        'mask_dilate_face': int(dcfg.get('mask_dilate_face', dcfg.get('mask_dilate', 0))),
        'mask_dilate_plate': int(dcfg.get('mask_dilate_plate', dcfg.get('mask_dilate', 0))),
        'min_plate_size': int(dcfg.get('min_plate_size', 320)),  # min shorter side for ROI upscale
        'roi_pad_ratio': float(dcfg.get('roi_pad_ratio', 0.25)),  # context padding around box (fraction of max(hw))
        'plate_steps': int(dcfg.get('plate_steps', dcfg.get('steps', 20))),
        'plate_guidance': float(dcfg.get('plate_guidance', dcfg.get('guidance_scale', 7.5))),
        'plate_pattern': str(dcfg.get('plate_pattern', 'LLLDDDD')),
        'plate_retry_max': int(dcfg.get('plate_retry_max', 0)),
        'plate_min_ocr_chars': int(dcfg.get('plate_min_ocr_chars', 4)),
        'plate_seed_shuffle': bool(dcfg.get('plate_seed_shuffle', True)),
        'plate_styles': dcfg.get('plate_styles', {}),
        'plate_style_whitelist': dcfg.get('plate_style_whitelist', None),
        # geometry heuristic
        'plate_geom_min_chars': int(dcfg.get('plate_geom_min_chars', 4)),
        'plate_geom_max_chars': int(dcfg.get('plate_geom_max_chars', 9)),
        'plate_geom_y_std_max': float(dcfg.get('plate_geom_y_std_max', 0.08)),
        'plate_geom_h_std_max': float(dcfg.get('plate_geom_h_std_max', 0.35)),
        'plate_geom_retry_max': int(dcfg.get('plate_geom_retry_max', 0)),
        'plate_geom_last_resort_enhance': bool(dcfg.get('plate_geom_last_resort_enhance', True)),
        # prompt token limiting
        'prompt_token_limit': int(dcfg.get('prompt_token_limit', 75)),
        # text anchoring extras
        'plate_text_anchor': bool(dcfg.get('plate_text_anchor', False)),
        'plate_text_anchor_strength': float(dcfg.get('plate_text_anchor_strength', 0.18)),
        'plate_text_anchor_blur_sigma': float(dcfg.get('plate_text_anchor_blur_sigma', 0.6)),
        'plate_text_margin_ratio': float(dcfg.get('plate_text_margin_ratio', 0.12)),
        'plate_text_font_ratio': float(dcfg.get('plate_text_font_ratio', 0.5)),
        'plate_text_font_ratio_min': float(dcfg.get('plate_text_font_ratio_min', 0.28)),
        'plate_text_font_ratio_max': float(dcfg.get('plate_text_font_ratio_max', 0.72)),
        'plate_text_anchor_retry_max': int(dcfg.get('plate_text_anchor_retry_max', 2)),
        'plate_text_anchor_font_step': float(dcfg.get('plate_text_anchor_font_step', 0.06)),
        'plate_text_local_sharpen_alpha': float(dcfg.get('plate_text_local_sharpen_alpha', 1.25)),
        'plate_text_local_sharpen_beta': float(dcfg.get('plate_text_local_sharpen_beta', 0.0)),
        # CN/international rendering controls
        'plate_cn_use_province': bool(dcfg.get('plate_cn_use_province', True)),
        'plate_feather_px': int(dcfg.get('plate_feather_px', 2)),
        'plate_perspective_jitter_deg': float(dcfg.get('plate_perspective_jitter_deg', 0.0)),
        # CN aspect tuning
        'plate_aspect_blue': float(dcfg.get('plate_aspect_blue', 3.14)),  # standard CN blue 440x140mm
        'plate_aspect_green': float(dcfg.get('plate_aspect_green', 3.43)),  # NEV green slightly wider
        'plate_aspect_correct_strength': float(dcfg.get('plate_aspect_correct_strength', 0.35)),
    }


def _random_plate_string(pattern: str) -> str:
    import random, string
    letters = string.ascii_uppercase
    digits = string.digits
    out = []
    for ch in pattern:
        if ch == 'L':
            out.append(random.choice(letters))
        elif ch == 'D':
            out.append(random.choice(digits))
        elif ch in ['-', ' ', '_']:
            out.append(ch)
        else:
            out.append(random.choice(letters + digits))
    return ''.join(out)


# ---------------- CN plate helpers -----------------
_CN_PROVINCES = list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
_CN_LETTERS = [ch for ch in "ABCDEFGHJKLMNPQRSTUVWXYZ"]  # exclude I,O


def _random_cn_plate_text(nev: bool = False) -> str:
    import random, string
    prov = random.choice(_CN_PROVINCES)
    letter = random.choice(_CN_LETTERS)
    if nev:
        df = random.choice(["D", "F"])  # NEV marker
        tail = []
        for _ in range(5):
            # use digits+letters excluding I,O for a CN look
            ch = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
            tail.append(ch)
        # Use middle dot for visual authenticity
        return f"{prov}{letter}{df}·{''.join(tail)}"
    else:
        tail = []
        for _ in range(5):
            ch = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
            tail.append(ch)
        return f"{prov}{letter}·{''.join(tail)}"


def _random_generic_plate_text() -> str:
    """Latin-only generic: 'AB 12345' for international use (no province CJK, no dot)."""
    import random, string
    letters = [ch for ch in "ABCDEFGHJKLMNPQRSTUVWXYZ"]
    two = random.choice(letters) + random.choice(letters)
    nums = "".join(random.choice(string.digits) for _ in range(5))
    return f"{two} {nums}"


def _detect_cn_plate_type(roi_rgb: np.ndarray, plate_xyxy: List[int]) -> str:
    """Detect likely CN plate background color: 'blue' or 'green'. Fallback 'blue'."""
    x1, y1, x2, y2 = plate_xyxy
    h, w = roi_rgb.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        crop = roi_rgb
    else:
        crop = roi_rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hch = hsv[..., 0].astype(np.float32) * 2.0  # [0,360)
    sch = hsv[..., 1].astype(np.float32) / 255.0
    vch = hsv[..., 2].astype(np.float32) / 255.0
    mask_color = sch > 0.25
    if mask_color.sum() < 50:
        return 'blue'
    h_vals = hch[mask_color]
    # simple thresholds: green around 60-170, blue around 180-260
    frac_green = np.mean((h_vals >= 60) & (h_vals <= 170))
    frac_blue = np.mean((h_vals >= 180) & (h_vals <= 260))
    if frac_green > 0.35 and frac_green > frac_blue:
        return 'green'
    return 'blue'


def _load_font(size: int, pref: str = 'latin') -> ImageFont.FreeTypeFont:
    """Try multiple fonts for CJK/Latin rendering on Windows; fallback to default."""
    candidates = []
    if pref == 'cjk':
        candidates = [
            r"C:\\Windows\\Fonts\\msyhbd.ttc",  # bold first
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simkai.ttf",
        ]
    else:
        candidates = [
            r"C:\\Windows\\Fonts\\arialbd.ttf",
            r"C:\\Windows\\Fonts\\arial.ttf",
            r"C:\\Windows\\Fonts\\consola.ttf",
        ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _render_cn_plate_anchor(base_roi: torch.Tensor, plate_xyxy: List[int], text_all: str, plate_type: str,
                            font_ratio: float, margin_ratio: float,
                            aspect_targets: tuple = (3.14, 3.43), aspect_correct_strength: float = 0.35,
                            feather_px: int = 2, perspective_deg: float = 0.0,
                            use_province: bool = True) -> torch.Tensor:
    """Render CN-like plate with optional province CJK or international Latin-only text.
    Separate patch composition + optional perspective jitter + feathered blend.
    base_roi: [B,3,H,W] in [-1,1]; plate_xyxy in ROI coords.
    """
    b, c, h, w = base_roi.shape
    out = base_roi.clone()
    font_size = max(10, int((plate_xyxy[3] - plate_xyxy[1]) * font_ratio))
    font_cjk = _load_font(font_size, pref='cjk')
    font_lat = _load_font(max(10, int(font_size * 0.95)), pref='latin')
    for i in range(b):
        img = ((out[i].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(img)
        x1, y1, x2, y2 = [int(v) for v in plate_xyxy]
        plate_w = max(1, x2 - x1)
        plate_h = max(1, y2 - y1)
        # Aspect normalization toward target aspect
        tgt_aspect = aspect_targets[1] if plate_type == 'green' else aspect_targets[0]
        cur_aspect = plate_w / max(1.0, float(plate_h))
        if abs(cur_aspect - tgt_aspect) > 0.05:
            soft_target = cur_aspect + (tgt_aspect - cur_aspect) * max(0.0, min(1.0, aspect_correct_strength))
            if soft_target > cur_aspect:
                new_h = int(round(plate_w / soft_target))
                new_h = min(new_h, plate_h)
                cy = (y1 + y2) // 2
                y1 = max(0, cy - new_h // 2)
                y2 = y1 + new_h
            else:
                new_w = int(round(plate_h * soft_target))
                new_w = min(new_w, plate_w)
                cx = (x1 + x2) // 2
                x1 = max(0, cx - new_w // 2)
                x2 = x1 + new_w
            plate_w = max(1, x2 - x1)
            plate_h = max(1, y2 - y1)
        # Gradient background patch
        if plate_type == 'green':
            base_color = np.array([55, 140, 65], dtype=np.float32)
            text_color = (10, 10, 10)
        else:
            base_color = np.array([30, 90, 185], dtype=np.float32)
            text_color = (240, 240, 245)
        grad = np.zeros((plate_h, plate_w, 3), dtype=np.float32)
        for yy in range(plate_h):
            trow = yy / max(1, plate_h - 1)
            light_factor = 0.15 * math.sin(trow * math.pi) + 0.85
            grad[yy, :, :] = np.clip(base_color * light_factor, 0, 255)
        stripe_y = int(plate_h * 0.45)
        stripe_h = max(1, int(plate_h * 0.12))
        grad[stripe_y:stripe_y + stripe_h] = np.clip(grad[stripe_y:stripe_y + stripe_h] * 1.12, 0, 255)
        noise = (np.random.randn(*grad.shape) * 4.0).astype(np.float32)
        grad = np.clip(grad + noise, 0, 255).astype(np.uint8)
        plate_pil = Image.fromarray(grad)
        plate_draw = ImageDraw.Draw(plate_pil)
        # Border + rivets
        border_col = (235, 235, 238) if plate_type != 'green' else (220, 230, 220)
        plate_draw.rectangle([0, 0, plate_w - 1, plate_h - 1], outline=border_col, width=max(1, int(plate_h * 0.06)))
        hole_r = max(1, int(0.055 * plate_h))
        cx_l, cy = int(0.12 * plate_w), int(0.18 * plate_h)
        cx_r = plate_w - int(0.12 * plate_w)
        plate_draw.ellipse([cx_l - hole_r, cy - hole_r, cx_l + hole_r, cy + hole_r], fill=(200, 200, 200))
        plate_draw.ellipse([cx_r - hole_r, cy - hole_r, cx_r + hole_r, cy + hole_r], fill=(200, 200, 200))
        # Text layout
        margin_x = int(plate_w * margin_ratio)
        ty = max(0, (plate_h - font_size) // 2)
        shadow_off = 1
        stroke_col = (20, 40, 90) if plate_type != 'green' else (0, 0, 0)
        stroke_w = max(1, int(plate_h * 0.016))
        if use_province and len(text_all) > 1:
            prov = text_all[0]
            rest = text_all[1:]
            prov_w = plate_draw.textlength(prov, font=font_cjk)
            spacing_w = int(0.05 * plate_w)
            available_w = plate_w - 2 * margin_x
            rest_w = plate_draw.textlength(rest, font=font_lat)
            total_w = prov_w + spacing_w + rest_w
            if total_w > available_w and rest_w > 0:
                scale = max(0.7, (available_w - prov_w - spacing_w) / max(1.0, rest_w))
                font_lat = _load_font(max(8, int(font_lat.size * scale)), pref='latin')
            tx = margin_x
            plate_draw.text((tx, ty), prov, font=font_cjk, fill=text_color,
                            stroke_width=stroke_w, stroke_fill=(0, 0, 0))
            tx += int(plate_draw.textlength(prov, font=font_cjk) + spacing_w)
            plate_draw.text((tx + shadow_off, ty + shadow_off), rest, font=font_lat, fill=(35, 35, 35))
            plate_draw.text((tx, ty), rest, font=font_lat, fill=text_color, stroke_width=stroke_w, stroke_fill=stroke_col)
        else:
            txt = text_all
            available_w = plate_w - 2 * margin_x
            tw = plate_draw.textlength(txt, font=font_lat)
            if tw > available_w and tw > 0:
                scale = max(0.7, available_w / tw)
                font_lat = _load_font(max(8, int(font_lat.size * scale)), pref='latin')
            tx = margin_x
            plate_draw.text((tx + shadow_off, ty + shadow_off), txt, font=font_lat, fill=(35, 35, 35))
            plate_draw.text((tx, ty), txt, font=font_lat, fill=text_color, stroke_width=stroke_w, stroke_fill=stroke_col)
        # Speckles
        speck_count = max(6, int(plate_w * plate_h * 0.0006))
        for _ in range(speck_count):
            sx = np.random.randint(0, plate_w)
            sy2 = np.random.randint(0, plate_h)
            if np.random.rand() < 0.6:
                plate_draw.point((sx, sy2), fill=(np.random.randint(120, 180),) * 3)
        # Perspective jitter
        if perspective_deg and perspective_deg > 0.0:
            import random as _r, math as _m
            deg = (_r.random() * 2 - 1) * perspective_deg
            dx = int(_m.tan(_m.radians(deg)) * plate_h * 0.25)
            src = np.float32([[0, 0], [plate_w - 1, 0], [plate_w - 1, plate_h - 1], [0, plate_h - 1]])
            dst = np.float32([[max(0, 0 + dx), 0], [min(plate_w - 1, plate_w - 1 - dx), 0],
                              [plate_w - 1, plate_h - 1], [0, plate_h - 1]])
            M = cv2.getPerspectiveTransform(src, dst)
            plate_np = np.array(plate_pil)[:, :, ::-1]
            warped = cv2.warpPerspective(plate_np, M, (plate_w, plate_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            plate_pil = Image.fromarray(warped[:, :, ::-1])
        # Feather mask blend into base
        mask = Image.new('L', (plate_w, plate_h), 0)
        mdraw = ImageDraw.Draw(mask)
        try:
            mdraw.rounded_rectangle([0, 0, plate_w - 1, plate_h - 1], radius=max(2, int(0.08 * plate_h)), fill=255)
        except Exception:
            mdraw.rectangle([0, 0, plate_w - 1, plate_h - 1], fill=255)
        if feather_px and feather_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        pil.paste(plate_pil, (x1, y1), mask)
        out_np = np.array(pil).astype(np.float32) / 255.0
        out_t = torch.from_numpy(out_np).permute(2, 0, 1)
        out[i] = out_t * 2 - 1
    return out


def _choose_plate_style(dcfg: Dict[str, Any]) -> Dict[str, Any]:
    import random
    styles: Dict[str, Dict[str, str]] = dcfg.get('plate_styles', {}) or {}
    if not styles:
        return {"name": "default", "pattern": dcfg.get('plate_pattern', 'LLLDDDD'), "prompt_hint": ""}
    whitelist = dcfg.get('plate_style_whitelist')
    names = list(styles.keys())
    if isinstance(whitelist, list) and len(whitelist) > 0:
        names = [n for n in names if n in whitelist] or names
    name = random.choice(names)
    st = styles.get(name, {})
    return {
        "name": name,
        "pattern": st.get('pattern', dcfg.get('plate_pattern', 'LLLDDDD')),
        "prompt_hint": st.get('prompt_hint', ''),
        "neg_hint": st.get('neg_hint', ''),
    }


def _ocr_extract_alnum(img: Image.Image) -> str:
    reader = _lazy_init_ocr()
    if reader is None:
        return ""
    try:
        res = reader.readtext(np.array(img))  # type: ignore[arg-type]
        texts = []
        for _, txt, conf in res:
            if conf < 0.2:
                continue
            filtered = ''.join([c for c in txt.upper() if c.isalnum()])
            if filtered:
                texts.append(filtered)
        return ''.join(texts)
    except Exception:
        return ""


def _measure_geom(img: Image.Image):
    # Convert to grayscale and binarize to detect character-like components
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Adaptive threshold or OTSU fallback
    try:
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 10)
    except Exception:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove small noise
    thr = cv2.medianBlur(thr, 3)
    # Find contours
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], 0.0, 0.0, (h, w)
    boxes = []
    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 0.002 * h * w or area > 0.15 * h * w:
            continue
        if hh < 0.15 * h or hh > 0.9 * h:
            continue
        boxes.append((x, y, ww, hh))
    if len(boxes) == 0:
        return [], 0.0, 0.0, (h, w)
    # Sort by x to approximate left-to-right ordering
    boxes.sort(key=lambda b: b[0])
    ys = np.array([y + hh / 2 for (_, y, _, hh) in boxes], dtype=np.float32)
    hs = np.array([hh for (_, _, _, hh) in boxes], dtype=np.float32)
    y_std = float(np.std(ys) / max(h, 1))
    h_std = float(np.std(hs) / (np.mean(hs) + 1e-6))
    return boxes, y_std, h_std, (h, w)


def _geom_quality_ok(img: Image.Image, min_chars: int, max_chars: int, y_std_max: float, h_std_max: float) -> bool:
    boxes, y_std, h_std, _ = _measure_geom(img)
    count_ok = (min_chars <= len(boxes) <= max_chars)
    return count_ok and (y_std <= y_std_max) and (h_std <= h_std_max)


def _mild_affine_enhance(t: torch.Tensor) -> torch.Tensor:
    # t: [B,3,H,W] in [-1,1]; apply light sharpen/contrast in place and return same size
    b, c, h, w = t.shape
    out_list = []
    for i in range(b):
        img = ((t[i].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
        # Unsharp mask
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        sharp = cv2.addWeighted(img, 1.2, blur, -0.2, 0)
        # Gentle contrast stretch
        alpha, beta = 1.08, 0
        enhanced = cv2.convertScaleAbs(sharp, alpha=alpha, beta=beta)
        out = torch.from_numpy(enhanced.astype(np.float32) / 255.0).permute(2, 0, 1)
        out = out * 2 - 1
        out_list.append(out)
    return torch.stack(out_list, dim=0)


def _render_anchor_text(base_roi: torch.Tensor, text: str, font_ratio: float, margin_ratio: float) -> torch.Tensor:
    # base_roi: [B,3,H,W] in [-1,1]; returns same shape with synthetic text overlay (white chars on dark mask) for guidance
    b, c, h, w = base_roi.shape
    out = base_roi.clone()
    try:
        # Choose a common font; fallback to default if unavailable
        font_size = max(10, int(h * font_ratio))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        margin_x = int(w * margin_ratio)
        # Build blank RGB canvas from base (we just overlay text region)
        for i in range(b):
            img = ((out[i].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            # Center vertically
            text_w = draw.textlength(text, font=font)
            x = max(margin_x, (w - text_w) // 2)
            y = max(0, (h - font_size) // 2)
            # Draw a slight dark plate base to enhance contrast
            plate_bg_color = (max(0, int(np.mean(img[:,:,0]) - 30)),) * 3
            draw.rectangle([0,0,w,h], fill=None)
            # White text with slight shadow for clarity
            shadow_offset = 1
            draw.text((x+shadow_offset,y+shadow_offset), text, font=font, fill=(40,40,40))
            draw.text((x,y), text, font=font, fill=(240,240,240))
            arr = np.array(pil).astype(np.float32)/255.0
            arr = torch.from_numpy(arr).permute(2,0,1)
            out[i] = arr*2 -1
    except Exception:
        return base_roi
    return out


def _gaussian_blur_tensor(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: [B,3,H,W] in [-1,1]
    if sigma <= 0:
        return x
    out = []
    for i in range(x.size(0)):
        img = ((x[i].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        t = torch.from_numpy(blurred.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = t * 2 - 1
        out.append(t)
    return torch.stack(out, dim=0)


def _local_sharpen_on_boxes(t: torch.Tensor, boxes: List[tuple], alpha: float = 1.2, beta: float = 0.0) -> torch.Tensor:
    # t: [B,3,H,W] in [-1,1], boxes in (x,y,w,h) on HxW canvas, apply unsharp/contrast only inside boxes
    b, c, h, w = t.shape
    out = t.clone()
    for i in range(b):
        img = ((out[i].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).cpu().numpy() * 255).astype(np.uint8)
        for (x, y, ww, hh) in boxes:
            x0 = int(max(0, x))
            y0 = int(max(0, y))
            x1 = int(min(w, x + ww))
            y1 = int(min(h, y + hh))
            if x1 <= x0 or y1 <= y0:
                continue
            patch = img[y0:y1, x0:x1, :]
            blur = cv2.GaussianBlur(patch, (0, 0), 0.8)
            sharp = cv2.addWeighted(patch, alpha, blur, -(alpha - 1.0), 0)
            enhanced = cv2.convertScaleAbs(sharp, alpha=1.06, beta=beta)
            img[y0:y1, x0:x1, :] = enhanced
        t_i = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t_i = t_i * 2 - 1
        out[i] = t_i
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def boxes_to_mask(boxes: torch.Tensor, h: int, w: int, dilate_px: int = 0) -> torch.Tensor:
    """
    boxes: [N,4] xyxy in image coords (CPU tensor)
    returns [1,H,W] float mask in {0,1}
    """
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for b in boxes.tolist():
        x1, y1, x2, y2 = b
        if dilate_px > 0:
            x1 = max(0, x1 - dilate_px)
            y1 = max(0, y1 - dilate_px)
            x2 = min(w - 1, x2 + dilate_px)
            y2 = min(h - 1, y2 + dilate_px)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    m = torch.from_numpy(np.array(mask, dtype=np.uint8)).float() / 255.0
    return m.unsqueeze(0)


@torch.no_grad()
def run_infer(cfg: Dict[str, Any], images_dir: str, out_dir: str, max_images: int = 100, recursive: bool = False,
              face_prompt: str = None, plate_prompt: str = None, neg_prompt: str = None,
              use_backbone: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device','auto') != 'cpu' else 'cpu')
    detector = DetectorWrapper(cfg).to(device)
    generator = GeneratorWrapper(cfg)
    backbone = build_backbone(cfg) if use_backbone else None

    # Collect images
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if recursive:
        all_imgs: List[str] = []
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(exts):
                    all_imgs.append(os.path.join(root, f))
    else:
        all_imgs = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(exts)]
    all_imgs = sorted(all_imgs)[:max_images]
    if len(all_imgs) == 0:
        print(f"[WARN] No images found under {images_dir} (recursive={recursive}).")
        return

    for p in all_imgs:
        img = Image.open(p).convert('RGB')
        w, h = img.size
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        arr = arr * 2 - 1  # [-1,1]
        inp = arr.unsqueeze(0)
        preds = detector.predict(inp)
        # Only keep face (1) and license plate (2) detections
        if len(preds) > 0:
            det = preds[0]
            boxes_all: torch.Tensor = det.get('boxes', torch.zeros((0, 4)))
            labels_all: torch.Tensor = det.get('labels', torch.zeros((0,), dtype=torch.long))
            scores_all: torch.Tensor = det.get('scores', torch.zeros((0,)))
            allowed = (labels_all == 1) | (labels_all == 2)
            # If labels are missing or empty, fall back to no mask
            if labels_all.numel() > 0:
                boxes = boxes_all[allowed]
                labels = labels_all[allowed]
            else:
                boxes = torch.zeros((0, 4))
                labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4))
        # 默认提示词（可被 CLI 覆盖）
        default_face_prompt = face_prompt or (
            "a photorealistic human face, neutral expression, natural skin tone, realistic lighting, high detail"
        )
        default_plate_prompt = plate_prompt or (
            "a photorealistic vehicle license plate, rectangular metal plate with rounded corners, high-contrast embossed alphanumeric characters, black digits and letters on reflective sheeting, mounting bolts, plausible but non-identifiable random characters, realistic lighting"
        )
        # 车牌增强的负面提示：避免纯色贴片、无字符
        default_neg = neg_prompt or (
            "blank, solid color, no characters, smudged, melted, deformed, extra arms, extra fingers, watermark, low contrast, blurry, distorted, low quality"
        )

        # 若无框，原样保存；若有框，逐框依次修复，按类别使用不同提示
        out = inp.clone()
        if boxes.numel() > 0:
            # 按面积从大到小，减少覆盖伪影
            areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            order = torch.argsort(areas, descending=True)
            boxes = boxes[order]
            labels = labels[order]
            dcfg = _get_diffusers_cfg(cfg)
            for j in range(boxes.shape[0]):
                box = boxes[j].unsqueeze(0)
                lab = int(labels[j].item()) if labels.numel() > 0 else -1
                # 类别自适应的掩膜膨胀
                dilate_px = dcfg['mask_dilate_face'] if lab == 1 else (dcfg['mask_dilate_plate'] if lab == 2 else dcfg['mask_dilate'])
                m = boxes_to_mask(box.cpu(), h, w, dilate_px=int(dilate_px))  # [1, H, W]
                if m.dim() == 3:
                    m = m.unsqueeze(0)
                if m.shape[0] != out.shape[0]:
                    m = m.expand(out.shape[0], -1, -1, -1)
                pad_h = int(math.ceil(h / 32) * 32 - h)
                pad_w = int(math.ceil(w / 32) * 32 - w)
                # 选择提示
                prompt = default_face_prompt if lab == 1 else (default_plate_prompt if lab == 2 else "")
                # 可选：提取条件特征（当前为占位，未注入 UNet，只传递给接口）
                cond = None
                if backbone is not None:
                    bb_out = backbone(out.to(device))  # tokens/maps from current canvas
                    if bb_out.maps:
                        # 简单用第一层 map 提取 ROI 特征
                        fmap = bb_out.maps[0]
                        roi = roi_features_from_boxes(fmap, box.to(fmap.device), output_size=(8, 8))
                        cond_vec = fuse_global_roi(bb_out.tokens, roi)
                        cond = {"cond_vec": cond_vec.detach()}

                # 车牌走ROI高分辨率路径，其它类别走全幅就地重绘
                if lab == 2:
                    x1, y1, x2, y2 = [int(v) for v in box[0].tolist()]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    pad = int(max(bw, bh) * dcfg['roi_pad_ratio'])
                    rx1 = max(0, x1 - pad)
                    ry1 = max(0, y1 - pad)
                    rx2 = min(w, x2 + pad)
                    ry2 = min(h, y2 + pad)
                    # 裁剪 ROI
                    roi_img = out[:, :, ry1:ry2, rx1:rx2].clone()
                    # 在ROI坐标系下构建单框掩膜
                    rel_box = torch.tensor([[x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]], dtype=torch.float32)
                    roi_mask = boxes_to_mask(rel_box, ry2 - ry1, rx2 - rx1, dilate_px=int(dilate_px)).unsqueeze(0)
                    # pad到32倍数
                    rh, rw = ry2 - ry1, rx2 - rx1
                    padh = int(math.ceil(rh / 32) * 32 - rh)
                    padw = int(math.ceil(rw / 32) * 32 - rw)
                    if padh > 0 or padw > 0:
                        pad_tuple = (0, padw, 0, padh)
                        roi_img = F.pad(roi_img, pad_tuple, mode='reflect')
                        roi_mask = F.pad(roi_mask, pad_tuple, mode='constant', value=0.0)
                    # 放大到最小边 min_plate_size
                    rhp, rwp = roi_img.shape[-2:]
                    short_side = min(rhp, rwp)
                    scale = max(1.0, dcfg['min_plate_size'] / float(short_side))
                    new_h = int(math.ceil(rhp * scale / 32) * 32)
                    new_w = int(math.ceil(rwp * scale / 32) * 32)
                    if scale > 1.0:
                        roi_img_up = F.interpolate(roi_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                        roi_mask_up = F.interpolate(roi_mask, size=(new_h, new_w), mode='nearest')
                    else:
                        roi_img_up, roi_mask_up = roi_img, roi_mask
                    # 适度增大步数/引导，提升细节与字符感
                    plate_steps = int(dcfg['plate_steps'])
                    plate_guidance = float(dcfg['plate_guidance'])
                    # 随机选择国家/地区风格并生成匿名字符；注入风格提示
                    style = _choose_plate_style(dcfg)
                    style_prompt_hint = style.get('prompt_hint', '')
                    style_neg_hint = style.get('neg_hint', '')
                    pattern = style.get('pattern', dcfg['plate_pattern'])
                    base_prompt = prompt
                    if style_prompt_hint:
                        base_prompt = f"{base_prompt}, {style_prompt_hint}"
                    # 随机生成匿名车牌字符并嵌入 prompt；OCR 质量检测不足则重试
                    plate_text = _random_plate_string(pattern)
                    prompt_plate = f"{base_prompt}, showing the text {plate_text}, clear high-contrast alphanumeric characters"
                    neg_all = default_neg + (f", {style_neg_hint}" if style_neg_hint else "")
                    retries = 0
                    max_retries = int(dcfg['plate_retry_max'])
                    seed_base = int(torch.randint(0, 10_000_000, (1,)).item())
                    while True:
                        seed_val = (seed_base + retries) if dcfg['plate_seed_shuffle'] else None
                        out_roi = generator(
                            roi_img_up.to(device), roi_mask_up.to(device),
                            prompt=prompt_plate, negative_prompt=neg_all, cond=cond,
                            steps=plate_steps, guidance_scale=plate_guidance,
                            seed=seed_val,
                        )
                        # Downscale to original ROI for OCR
                        final_roi = out_roi
                        if final_roi.shape[-2:] != (rhp, rwp):
                            final_roi = F.interpolate(final_roi, size=(rhp, rwp), mode='bilinear', align_corners=False)
                        roi_pil = Image.fromarray(((final_roi[0].permute(1, 2, 0).clamp(-1,1)*0.5+0.5).detach().cpu().numpy()*255).astype('uint8'))
                        ocr_text = _ocr_extract_alnum(roi_pil)
                        geom_ok = _geom_quality_ok(roi_pil,
                                                   dcfg['plate_geom_min_chars'],
                                                   dcfg['plate_geom_max_chars'],
                                                   dcfg['plate_geom_y_std_max'],
                                                   dcfg['plate_geom_h_std_max'])
                        ocr_ok = isinstance(ocr_text, str) and len(ocr_text) >= int(dcfg['plate_min_ocr_chars'])
                        if ocr_ok and geom_ok:
                            break
                        # Geometry retry allowance separate from OCR retries
                        total_allowed = max_retries + dcfg['plate_geom_retry_max']
                        if retries >= total_allowed:
                            # Last resort enhancement if enabled
                            if dcfg['plate_geom_last_resort_enhance']:
                                out_roi = _mild_affine_enhance(out_roi)
                            break
                        retries += 1
                        plate_text = _random_plate_string(pattern)
                        prompt_plate = f"{base_prompt}, showing the text {plate_text}, clear high-contrast alphanumeric characters"
                    # 使用最终 out_roi，必要时缩放回原尺寸
                    if out_roi.shape[-2:] != (rhp, rwp):
                        out_roi = F.interpolate(out_roi, size=(rhp, rwp), mode='bilinear', align_corners=False)
                    # 双阶段文本锚定：合成文本 -> (可选)模糊降噪 -> 低strength重绘 -> 局部字符框锐化
                    if dcfg.get('plate_text_anchor', False):
                        # 中国车牌专用：若风格是 CN 列表或 whitelist 仅 CN，则启用 CN 模板模拟
                        is_cn_style = style.get('name','').upper() in ['CN','CHINA','ZH'] or (dcfg.get('plate_style_whitelist') == ['CN'])
                        # 检测原始 ROI 颜色类型（绿色新能源或蓝色普通）用于背景渲染
                        roi_rgb_for_color = ((roi_img_up[0].permute(1,2,0).clamp(-1,1)*0.5+0.5).cpu().numpy()*255).astype(np.uint8)
                        plate_type = _detect_cn_plate_type(roi_rgb_for_color, [0,0,roi_rgb_for_color.shape[1], roi_rgb_for_color.shape[0]]) if is_cn_style else 'blue'
                        if is_cn_style:
                            # 生成符合 CN 结构的文本（新能源概率 30%）并根据配置是否保留省份汉字
                            import random
                            anchor_text = _random_cn_plate_text(nev=(plate_type=='green'))
                            if not dcfg.get('plate_cn_use_province', True):
                                anchor_text = anchor_text[1:].replace('·', ' ')
                        else:
                            anchor_text = plate_text.replace('-', '').replace('·','').replace(' ','')
                        # 动态字体比例回路
                        font_ratio = float(dcfg['plate_text_font_ratio'])
                        margin_ratio_cfg = float(dcfg['plate_text_margin_ratio'])
                        blur_sigma = float(dcfg['plate_text_anchor_blur_sigma'])
                        anchor_strength = float(dcfg['plate_text_anchor_strength'])
                        min_chars = int(dcfg['plate_geom_min_chars'])
                        max_chars = int(dcfg['plate_geom_max_chars'])
                        y_std_max = float(dcfg['plate_geom_y_std_max'])
                        h_std_max = float(dcfg['plate_geom_h_std_max'])
                        font_step = float(dcfg['plate_text_anchor_font_step'])
                        fr_min = float(dcfg['plate_text_font_ratio_min'])
                        fr_max = float(dcfg['plate_text_font_ratio_max'])
                        anchor_retries = int(dcfg['plate_text_anchor_retry_max'])
                        last_boxes = []
                        for atry in range(anchor_retries + 1):
                            if is_cn_style:
                                # 将原图坐标的车牌框映射到当前 out_roi 尺寸
                                sy = out_roi.shape[-2] / float(rhp)
                                sx = out_roi.shape[-1] / float(rwp)
                                px1 = int(max(0, (x1 - rx1) * sx))
                                py1 = int(max(0, (y1 - ry1) * sy))
                                px2 = int(min(out_roi.shape[-1], (x2 - rx1) * sx))
                                py2 = int(min(out_roi.shape[-2], (y2 - ry1) * sy))
                                plate_xyxy = [px1, py1, px2, py2]
                                anchor_canvas_stage1 = _render_cn_plate_anchor(
                                    out_roi,
                                    plate_xyxy,
                                    anchor_text,
                                    plate_type=plate_type,
                                    font_ratio=font_ratio,
                                    margin_ratio=margin_ratio_cfg,
                                    aspect_targets=(float(dcfg['plate_aspect_blue']), float(dcfg['plate_aspect_green'])),
                                    aspect_correct_strength=float(dcfg['plate_aspect_correct_strength']),
                                    feather_px=int(dcfg.get('plate_feather_px', 2)),
                                    perspective_deg=float(dcfg.get('plate_perspective_jitter_deg', 0.0)),
                                    use_province=bool(dcfg.get('plate_cn_use_province', True)),
                                )
                            else:
                                anchor_canvas_stage1 = _render_anchor_text(out_roi, anchor_text,
                                                                           font_ratio=font_ratio,
                                                                           margin_ratio=margin_ratio_cfg)
                            anchor_canvas_blurred = _gaussian_blur_tensor(anchor_canvas_stage1, sigma=blur_sigma)
                            refine_mask = F.interpolate(roi_mask, size=anchor_canvas_blurred.shape[-2:], mode='nearest')
                            out_refined_stage2 = generator(anchor_canvas_blurred.to(device), refine_mask.to(device),
                                                           prompt=prompt_plate,
                                                           negative_prompt=neg_all,
                                                           cond=cond,
                                                           steps=max(8, plate_steps//2),
                                                           guidance_scale=plate_guidance,
                                                           seed=seed_val,
                                                           strength=anchor_strength)
                            roi_eval = out_refined_stage2
                            if roi_eval.shape[-2:] != (rhp, rwp):
                                roi_eval = F.interpolate(roi_eval, size=(rhp, rwp), mode='bilinear', align_corners=False)
                            roi_eval_pil = Image.fromarray(((roi_eval[0].permute(1,2,0).clamp(-1,1)*0.5+0.5).cpu().numpy()*255).astype('uint8'))
                            boxes_geom, y_std_val, h_std_val, _ = _measure_geom(roi_eval_pil)
                            last_boxes = boxes_geom
                            geom_ok2 = (min_chars <= len(boxes_geom) <= max_chars) and (y_std_val <= y_std_max) and (h_std_val <= h_std_max)
                            if geom_ok2 or atry == anchor_retries:
                                # 局部字符框检测并增强对比/锐度
                                if boxes_geom:
                                    out_refined_stage2 = _local_sharpen_on_boxes(out_refined_stage2, boxes_geom,
                                                                                alpha=float(dcfg['plate_text_local_sharpen_alpha']),
                                                                                beta=float(dcfg['plate_text_local_sharpen_beta']))
                                out_roi = out_refined_stage2
                                break
                            # 微调字体尺寸：字符少则增大，字符多则减小；若偏差仅体现在几何std过大，则轻微减小
                            if len(boxes_geom) < min_chars:
                                font_ratio = min(fr_max, font_ratio * (1.0 + font_step))
                            elif len(boxes_geom) > max_chars:
                                font_ratio = max(fr_min, font_ratio * (1.0 - font_step))
                            else:
                                font_ratio = max(fr_min, min(fr_max, font_ratio * (1.0 - 0.5 * font_step)))
                    # 还原到原ROI尺寸
                    if out_roi.shape[-2:] != (rhp, rwp):
                        out_roi = F.interpolate(out_roi, size=(rhp, rwp), mode='bilinear', align_corners=False)
                    # 覆盖回画布
                    out[:, :, ry1:ry2, rx1:rx2] = out_roi[:, :, : (ry2 - ry1), : (rx2 - rx1)].detach()
                else:
                    # 全图路径（脸等），pad到32再生成
                    out_pad = out
                    m_pad = m
                    if pad_h > 0 or pad_w > 0:
                        pad_tuple = (0, pad_w, 0, pad_h)
                        out_pad = F.pad(out_pad, pad_tuple, mode='reflect')
                        m_pad = F.pad(m_pad, pad_tuple, mode='constant', value=0.0)
                    out_gen = generator(out_pad.to(device), m_pad.to(device), prompt=prompt, negative_prompt=default_neg, cond=cond)
                    if pad_h > 0 or pad_w > 0:
                        out_gen = out_gen[:, :, :h, :w]
                    out = out_gen.detach()  # 作为下一轮输入
        out_img = (out[0].detach().cpu().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
        out_img = Image.fromarray((out_img * 255).astype('uint8'))
        out_path = os.path.join(out_dir, os.path.basename(p))
        out_img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--images', required=True, help='Folder of images for anonymization')
    ap.add_argument('--output', required=True, help='Output folder')
    ap.add_argument('--max_images', type=int, default=100)
    ap.add_argument('--gen_backend', type=str, default=None, choices=['unet','diffusers'], help='Override generator backend for smoke/test')
    ap.add_argument('--det_backend', type=str, default=None, choices=['yolos','detr','face_insight','yolo_plate','face_plate_hybrid'], help='Override detector backend for smoke/test')
    ap.add_argument('--recursive', action='store_true', help='Recursively scan image directory')
    ap.add_argument('--det_conf', type=float, default=None, help='Override detector confidence threshold')
    ap.add_argument('--face_prompt', type=str, default=None, help='Prompt used for face regions')
    ap.add_argument('--plate_prompt', type=str, default=None, help='Prompt used for license plate regions')
    ap.add_argument('--neg_prompt', type=str, default=None, help='Negative prompt for diffusion inpainting')
    ap.add_argument('--use_backbone', action='store_true', help='Enable shared backbone features as generator condition (experimental)')
    ap.add_argument('--mask_dilate_px', type=int, default=None, help='Dilate each detected box by N pixels when building mask')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.gen_backend is not None:
        cfg.setdefault('model', {}).setdefault('generator', {})
        cfg['model']['generator']['backend'] = args.gen_backend
    if args.det_backend is not None:
        cfg.setdefault('model', {}).setdefault('detector', {})
        cfg['model']['detector']['name'] = args.det_backend
    if args.det_conf is not None:
        cfg.setdefault('model', {}).setdefault('detector', {})
        cfg['model']['detector']['conf_threshold'] = float(args.det_conf)
    if args.mask_dilate_px is not None:
        cfg.setdefault('model', {}).setdefault('generator', {}).setdefault('diffusers', {})
        cfg['model']['generator']['diffusers']['mask_dilate'] = int(args.mask_dilate_px)
    run_infer(cfg, args.images, args.output, max_images=args.max_images, recursive=args.recursive,
              face_prompt=args.face_prompt, plate_prompt=args.plate_prompt, neg_prompt=args.neg_prompt,
              use_backbone=bool(args.use_backbone))


if __name__ == '__main__':
    main()
