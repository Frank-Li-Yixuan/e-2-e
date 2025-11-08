#!/usr/bin/env python
"""
Training report generator.

Reads a metrics.csv written by src/train_joint.py and produces:
- A combined PNG with curves for: loss_gen, det_loss, arcface_mean_sim, easyocr_plate_acc, fid, val_map
- Separate PNGs per metric
- A lightweight HTML summary embedding the figures

Usage examples:
  python scripts/report_training.py --metrics /content/outputs/planB/metrics.csv
  python scripts/report_training.py --config configs/planB_colab.yaml
  python scripts/report_training.py --out-dir /content/outputs/planB

Auto-detection order if --metrics not provided:
  1) If --out-dir given: use out-dir/metrics.csv
  2) If --config given: read paths.outputs then use outputs/metrics.csv
  3) Fallback: search for the newest metrics.csv under ./outputs and ./runs
"""
import argparse
import csv
import glob
import os
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


def load_cfg_outputs(config_path: str) -> str:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        out_dir = cfg.get('paths', {}).get('outputs', 'runs/joint_small')
        return out_dir
    except Exception:
        return 'runs/joint_small'


def find_latest_metrics() -> str:
    cands = []
    for root in ['outputs', 'runs', '.']:
        cands.extend(glob.glob(os.path.join(root, '**', 'metrics.csv'), recursive=True))
    if not cands:
        return ''
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def read_metrics(csv_path: str) -> Tuple[List[int], Dict[str, List[float]]]:
    xs: List[int] = []
    cols: Dict[str, List[float]] = {}
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return xs, cols
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row.get('step', len(xs)))
            except Exception:
                step = len(xs)
            xs.append(step)
            for k, v in row.items():
                if k == 'step':
                    continue
                if k not in cols:
                    cols[k] = []
                try:
                    if v is None or v == '' or str(v).lower() == 'nan':
                        cols[k].append(math.nan)
                    else:
                        cols[k].append(float(v))
                except Exception:
                    cols[k].append(math.nan)
    return xs, cols


def plot_series(xs: List[int], ys: List[float], title: str, out_path: str, ylabel: str = '', smooth: int = 1):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.xlabel('step')
    if ylabel:
        plt.ylabel(ylabel)
    # optional moving average smoothing
    if smooth and smooth > 1 and len(ys) >= smooth:
        sm = []
        w = smooth
        s = 0.0
        for i, v in enumerate(ys):
            s += 0 if math.isnan(v) else v
            if i >= w:
                s -= 0 if math.isnan(ys[i-w]) else ys[i-w]
            if i >= w - 1:
                sm.append(s / float(w))
            else:
                sm.append(v)
        plt.plot(xs, sm, label=f'{title} (MA{w})')
    else:
        plt.plot(xs, ys, label=title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grid(xs: List[int], cols: Dict[str, List[float]], out_dir: str) -> str:
    # Choose main metrics
    metrics = [
        ('loss_gen', 'Generator Loss'),
        ('loss_det', 'Detector Loss'),
        ('arcface_mean_sim', 'ArcFace mean sim (↓)'),
        ('easyocr_plate_acc', 'EasyOCR plate score (↑)'),
        ('fid', 'FID (↓)'),
        ('val_map', 'mAP AP@[.5:.95] (↑)'),
    ]
    # 2x3 grid
    grid_path = os.path.join(out_dir, 'report_grid.png')
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14, 8))
    for i, (k, title) in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        ys = cols.get(k, [])
        if ys:
            plt.plot(xs, ys, label=title)
        plt.title(title)
        plt.xlabel('step')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(grid_path)
    plt.close()
    return grid_path


def write_html(out_dir: str, figures: List[str]) -> str:
    html_path = os.path.join(out_dir, 'report.html')
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<html><head><meta charset="utf-8"><title>Training Report</title></head><body>\n')
        f.write(f'<h1>Training Report</h1><p>Generated: {ts}</p>\n')
        for fig in figures:
            rel = os.path.basename(fig)
            f.write(f'<div><img src="{rel}" style="max-width:100%"/></div>\n')
        f.write('</body></html>\n')
    return html_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', type=str, default=None, help='Path to metrics.csv')
    ap.add_argument('--out-dir', type=str, default=None, help='Output directory (defaults to dirname(metrics.csv))')
    ap.add_argument('--config', type=str, default=None, help='Optional config to locate outputs path')
    ap.add_argument('--smooth', type=int, default=5, help='Moving average window for per-metric plots')
    args = ap.parse_args()

    metrics = args.metrics
    out_dir = args.out_dir
    if not metrics:
        if out_dir:
            metrics = os.path.join(out_dir, 'metrics.csv')
        elif args.config:
            out_dir = load_cfg_outputs(args.config)
            metrics = os.path.join(out_dir, 'metrics.csv')
        else:
            metrics = find_latest_metrics()
            out_dir = os.path.dirname(metrics) if metrics else 'outputs'

    if not metrics or not os.path.exists(metrics):
        print('[ERROR] metrics.csv not found. Provide --metrics or --config/--out-dir.')
        sys.exit(1)
    if not out_dir:
        out_dir = os.path.dirname(metrics)

    xs, cols = read_metrics(metrics)
    if not xs:
        print('[WARN] No rows in metrics.csv')

    # Per-metric figures
    fig_paths = []
    for key, title in [
        ('loss_gen', 'Generator Loss'),
        ('loss_det', 'Detector Loss'),
        ('arcface_mean_sim', 'ArcFace mean sim (lower is better)'),
        ('easyocr_plate_acc', 'EasyOCR plate score (higher is better)'),
        ('fid', 'FID (lower is better)'),
        ('val_map', 'mAP AP@[.5:.95] (higher is better)'),
    ]:
        ys = cols.get(key)
        if ys:
            p = os.path.join(out_dir, f'{key}.png')
            plot_series(xs, ys, title, p, ylabel=key, smooth=args.smooth)
            fig_paths.append(p)

    # Grid figure
    grid = plot_grid(xs, cols, out_dir)
    fig_paths.insert(0, grid)

    html = write_html(out_dir, fig_paths)
    print('[OK] Wrote figures and HTML to', out_dir)
    print(' -', grid)
    for p in fig_paths[1:]:
        print(' -', p)
    print(' -', html)


if __name__ == '__main__':
    main()
