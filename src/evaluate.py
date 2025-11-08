import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, List, Tuple

import yaml
import torch
import math

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from . import eval_utils


def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = sys.version
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_mb"] = props.total_memory // (1024 * 1024)
    except Exception as e:
        info["gpu_error"] = str(e)
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True, timeout=5)
        info["nvidia_smi"] = out
    except Exception as e:
        info["nvidia_smi"] = None
        info["nvidia_smi_error"] = str(e)
    return info


def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return e


def measure_generator_runtime(cfg: Dict[str, Any], images_dir: str, gt_annotations: str, max_images: int = 16) -> Dict[str, Any]:
    # Build masks from GT, run generator forward to estimate seconds per image
    from .generator_wrapper import GeneratorWrapper
    from PIL import Image, ImageDraw
    import numpy as np
    with open(gt_annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    # index by image_id
    ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in coco.get('annotations', []):
        ann_by_img.setdefault(a['image_id'], []).append(a)
    gen = GeneratorWrapper(cfg)
    # iterate first N images that exist in images_dir
    images = coco.get('images', [])
    count = 0
    t0 = time.time()
    for img_entry in images:
        if count >= max_images:
            break
        p = os.path.join(images_dir, os.path.basename(img_entry['file_name']))
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert('RGB')
        w, h = img.size
        boxes = [a['bbox'] for a in ann_by_img.get(img_entry['id'], [])]
        # build mask
        mask = Image.new('L', (w, h), color=0)
        draw = ImageDraw.Draw(mask)
        for x, y, bw, bh in boxes:
            draw.rectangle([x, y, x + bw, y + bh], fill=255)
        # to tensors
        x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        m = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)
        x = x.unsqueeze(0)
        m = m.unsqueeze(0)
        with torch.no_grad():
            _ = gen(x, m)
        count += 1
    t1 = time.time()
    return {"images": count, "seconds": t1 - t0, "sec_per_image": (t1 - t0) / max(1, count)}


def compute_lpips(orig_dir: str, anon_dir: str, max_images: Optional[int] = None) -> Optional[float]:
    try:
        import lpips  # type: ignore
        from PIL import Image
        import numpy as np
    except Exception:
        return None
    files = [f for f in os.listdir(orig_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if max_images:
        files = files[:max_images]
    loss_fn = lpips.LPIPS(net='alex')
    vals: List[float] = []
    for fn in files:
        p1 = os.path.join(orig_dir, fn)
        p2 = os.path.join(anon_dir, fn)
        if not os.path.exists(p2):
            continue
        a = torch.from_numpy(np.array(Image.open(p1).convert('RGB')).astype(np.float32) / 255.0).permute(2, 0, 1)
        b = torch.from_numpy(np.array(Image.open(p2).convert('RGB')).astype(np.float32) / 255.0).permute(2, 0, 1)
        # to [-1,1]
        a = a * 2 - 1
        b = b * 2 - 1
        v = float(loss_fn(a.unsqueeze(0), b.unsqueeze(0)).item())
        vals.append(v)
    if len(vals) == 0:
        return None
    return float(sum(vals) / len(vals))


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_md(path: str, report: Dict[str, Any], warns: List[str]) -> None:
    lines: List[str] = []
    lines.append("# Eval Report\n")
    env = report.get("env", {})
    lines.append("## Environment\n")
    lines.append(f"- Python: {env.get('python')}\n")
    lines.append(f"- Torch: {env.get('torch')} (CUDA avail: {env.get('cuda_available')}, CUDA: {env.get('torch_cuda_version')})\n")
    if env.get('gpu_name'):
        lines.append(f"- GPU: {env.get('gpu_name')} ({env.get('gpu_vram_mb')} MB)\n")
    if env.get('nvidia_smi'):
        lines.append("<details><summary>nvidia-smi</summary>\n\n")
        lines.append("```\n" + env.get('nvidia_smi') + "```\n\n</details>\n")

    lines.append("## Metrics (orig vs anon)\n")
    for k, v in report.get("metrics", {}).items():
        lines.append(f"- {k}: {v}\n")

    if report.get("per_size"):
        lines.append("## Size-based summary\n")
        for sz, row in report["per_size"].items():
            lines.append(f"- {sz}: {row}\n")

    if warns:
        lines.append("\n## WARN/Notes\n")
        for w in warns:
            lines.append(f"- {w}\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def append_metrics_csv(path: str, rows: List[List[Any]]) -> None:
    import csv
    header = ["metric", "value", "notes"]
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def classify_status(val: Optional[float], warn: float, fail: float, higher_is_better: bool = False) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 'WARN'
    if higher_is_better:
        # PASS >= warn; FAIL < fail
        if val < fail:
            return 'FAIL'
        if val < warn:
            return 'WARN'
        return 'PASS'
    else:
        # lower is better: PASS <= warn; FAIL > fail
        if val > fail:
            return 'FAIL'
        if val > warn:
            return 'WARN'
        return 'PASS'


def load_thresholds(path: str) -> Dict[str, Any]:
    defaults = {
        'detection': {'ap50_drop_warn': 0.10, 'ap50_drop_fail': 0.20},
        'perceptual': {'fid_warn': 50.0, 'fid_fail': 100.0, 'lpips_warn': 0.20, 'lpips_fail': 0.35},
        'deid': {'arcface_mean_warn': 0.45, 'arcface_mean_fail': 0.6, 'easyocr_drop_warn': 0.5, 'easyocr_drop_fail': 0.8},
        'runtime': {'fps_warn': 5.0, 'fps_fail': 1.0},
        'size_bins': {'small': 16, 'medium': 64}
    }
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                cfg = yaml.safe_load(f)
                for k, v in defaults.items():
                    if k not in cfg:
                        cfg[k] = v
                return cfg
            except Exception:
                return defaults
    return defaults


def plot_figures(out_dir: str, rows: List[Dict[str, Any]]) -> None:
    figs_dir = os.path.join(out_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    # FID/LPIPS bar
    labels = [r['baseline'] for r in rows]
    fid_vals = [r.get('FID', float('nan')) for r in rows]
    lpips_vals = [r.get('LPIPS', float('nan')) for r in rows]
    plt.figure(figsize=(8,4))
    x = range(len(labels))
    plt.bar([i-0.2 for i in x], fid_vals, width=0.4, label='FID')
    plt.bar([i+0.2 for i in x], lpips_vals, width=0.4, label='LPIPS')
    plt.xticks(list(x), labels, rotation=30, ha='right')
    plt.ylabel('score')
    plt.title('FID / LPIPS by baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'fid_lpips.png'))
    plt.close()

    # ArcFace mean box/points
    arc_vals = [r.get('ArcFace_mean_cosine', float('nan')) for r in rows]
    plt.figure(figsize=(8,4))
    plt.bar(list(x), arc_vals)
    plt.xticks(list(x), labels, rotation=30, ha='right')
    plt.ylabel('mean cosine (lower=better)')
    plt.title('ArcFace cosine mean by baseline')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'arcface_bar.png'))
    plt.close()

    # mAP bar placeholder
    map_vals = [r.get('mAP_AP@[.5:.95]', float('nan')) for r in rows]
    plt.figure(figsize=(8,4))
    plt.bar(list(x), map_vals)
    plt.xticks(list(x), labels, rotation=30, ha='right')
    plt.ylabel('mAP AP@[.5:.95]')
    plt.title('mAP (detector baseline needed)')
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'map_bar.png'))
    plt.close()


def eval_one(orig_dir: str, anon_dir: str, gt_annotations: str, cfg: Dict[str, Any], max_images: int) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    warns: List[str] = []
    # FID
    fid = safe_call(eval_utils.eval_fid, orig_dir, anon_dir, max_images)
    metrics['FID'] = float(fid) if not isinstance(fid, Exception) and fid is not None else float('nan')
    if isinstance(fid, Exception) or fid is None:
        warns.append(f"FID unavailable: {fid}")
    # LPIPS
    lp = compute_lpips(orig_dir, anon_dir, max_images)
    metrics['LPIPS'] = float(lp) if lp is not None else float('nan')
    if lp is None:
        warns.append('LPIPS unavailable (lpips not installed)')
    # ArcFace
    arc = safe_call(eval_utils.eval_arcface, orig_dir, anon_dir, max_images)
    metrics['ArcFace_mean_cosine'] = float(arc) if not isinstance(arc, Exception) and arc is not None else float('nan')
    if isinstance(arc, Exception) or arc is None:
        warns.append(f"ArcFace unavailable: {arc}")
    # EasyOCR
    ocr = safe_call(eval_utils.eval_easyocr, anon_dir, max_images)
    metrics['EasyOCR_avg_words_per_image'] = float(ocr) if not isinstance(ocr, Exception) and ocr is not None else float('nan')
    if isinstance(ocr, Exception) or ocr is None:
        warns.append(f"EasyOCR unavailable or no results: {ocr}")
    # mAP
    m = safe_call(eval_utils.eval_map, cfg if cfg else {"paths": {}}, anon_dir, gt_annotations, max_images)
    metrics['mAP_AP@[.5:.95]'] = float(m) if not isinstance(m, Exception) and m is not None else float('nan')
    if isinstance(m, Exception) or m is None:
        warns.append("mAP unavailable (detector baseline required; run on Colab)")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default='configs/joint_small.yaml')
    parser.add_argument('--orig_dir', type=str, required=True)
    parser.add_argument('--anon_dir', type=str, required=False)
    parser.add_argument('--anon_dirs', type=str, required=False, help='Comma-separated list of anonymized/baseline dirs')
    parser.add_argument('--gt_annotations', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=False, default='outputs')
    parser.add_argument('--out_root', type=str, required=False, default=None)
    parser.add_argument('--max_images', type=int, default=100)
    parser.add_argument('--thresholds', type=str, default='configs/eval_thresholds.yaml')
    args = parser.parse_args()

    out_dir = args.out_root if args.out_root else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # load cfg if exists
    cfg: Dict[str, Any] = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    env = get_env_info()
    thresholds = load_thresholds(args.thresholds)

    # determine baseline dirs
    baselines: List[Tuple[str, str]] = []  # (name, path)
    if args.anon_dirs:
        for p in args.anon_dirs.split(','):
            p = p.strip()
            if p:
                baselines.append((os.path.basename(p.rstrip('/\\')), p))
    elif args.anon_dir:
        baselines.append((os.path.basename(args.anon_dir.rstrip('/\\')), args.anon_dir))
    else:
        baselines.append(('anon', args.orig_dir))

    # also include 'orig' self-metrics row
    rows: List[Dict[str, Any]] = []
    # compute orig vs orig as a reference
    orig_metrics = {
        'FID': 0.0,
        'LPIPS': 0.0,
        'ArcFace_mean_cosine': 1.0,
        'EasyOCR_avg_words_per_image': safe_call(eval_utils.eval_easyocr, args.orig_dir, min(args.max_images, 100)),
        'mAP_AP@[.5:.95]': float('nan'),
    }
    if isinstance(orig_metrics['EasyOCR_avg_words_per_image'], Exception) or orig_metrics['EasyOCR_avg_words_per_image'] is None:
        orig_metrics['EasyOCR_avg_words_per_image'] = float('nan')
    rows.append({'baseline': 'orig', **{k: (float(v) if not isinstance(v, Exception) and v is not None else float('nan')) for k, v in orig_metrics.items()}})

    # evaluate each baseline
    for name, path in baselines:
        metrics = eval_one(args.orig_dir, path, args.gt_annotations, cfg, args.max_images)
        # save per-baseline eval.json
        subdir = os.path.join(out_dir, name)
        os.makedirs(subdir, exist_ok=True)
        save_json(os.path.join(subdir, 'eval.json'), {'baseline': name, 'metrics': metrics})
        rows.append({'baseline': name, **metrics})

    # write metrics_all.csv
    import csv
    metrics_all_csv = os.path.join('outputs', 'metrics_all.csv')
    os.makedirs(os.path.dirname(metrics_all_csv), exist_ok=True)
    with open(metrics_all_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        cols = ['baseline', 'FID', 'LPIPS', 'ArcFace_mean_cosine', 'EasyOCR_avg_words_per_image', 'mAP_AP@[.5:.95]']
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, '') for c in cols])

    # figures
    plot_figures(out_dir, rows)

    # build combined report with statuses
    def get_val(name: str) -> Optional[float]:
        for r in rows:
            if r['baseline'] == name:
                return r.get('EasyOCR_avg_words_per_image', None)  # used for OCR drop baseline
        return None

    report: Dict[str, Any] = {
        'env': env,
        'inputs': {
            'orig_dir': args.orig_dir,
            'anon_dirs': [p for _, p in baselines],
            'gt_annotations': args.gt_annotations,
            'max_images': args.max_images,
        },
        'metrics': {},
        'warns': [],
    }

    # statuses per baseline
    statuses: Dict[str, Any] = {}
    ocr_orig = rows[0].get('EasyOCR_avg_words_per_image', float('nan'))
    for r in rows:
        name = r['baseline']
        if name == 'orig':
            continue
        st: Dict[str, Any] = {}
        # perceptual
        fid = r.get('FID', float('nan'))
        lp = r.get('LPIPS', float('nan'))
        st['FID_status'] = classify_status(fid, thresholds['perceptual']['fid_warn'], thresholds['perceptual']['fid_fail'], higher_is_better=False)
        st['LPIPS_status'] = classify_status(lp, thresholds['perceptual']['lpips_warn'], thresholds['perceptual']['lpips_fail'], higher_is_better=False)
        # deid
        arc = r.get('ArcFace_mean_cosine', float('nan'))
        st['ArcFace_status'] = classify_status(arc, thresholds['deid']['arcface_mean_warn'], thresholds['deid']['arcface_mean_fail'], higher_is_better=False)
        ocr = r.get('EasyOCR_avg_words_per_image', float('nan'))
        if not (isinstance(ocr, float) and not math.isnan(ocr)) or not (isinstance(ocr_orig, float) and not math.isnan(ocr_orig)) or ocr_orig <= 0:
            st['EasyOCR_status'] = 'WARN'
        else:
            drop = max(0.0, 1.0 - (ocr / ocr_orig))
            # higher drop is better for de-id; so higher_is_better=True with warn/fail thresholds
            st['EasyOCR_status'] = classify_status(drop, thresholds['deid']['easyocr_drop_warn'], thresholds['deid']['easyocr_drop_fail'], higher_is_better=True)
        # detection mAP drop (needs baseline)
        st['mAP_status'] = 'WARN'  # local placeholder
        statuses[name] = st

    report['metrics'] = {'rows': rows, 'statuses': statuses}

    # Save combined
    save_json(os.path.join('outputs', 'eval_report.json'), report)
    save_md(os.path.join('outputs', 'eval_report.md'), report, report.get('warns', []))

    print('[EVAL DONE] Combined:')
    print(' -', os.path.join('outputs', 'metrics_all.csv'))
    print(' -', os.path.join('outputs', 'eval_report.json'))
    print(' -', os.path.join('outputs', 'eval_report.md'))


if __name__ == '__main__':
    main()
