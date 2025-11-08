#!/usr/bin/env python
import argparse
import csv
import os
import json
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

from src.eval_utils import arcface_similarity  # uses insightface under the hood


def load_metrics(csv_path: str):
    rows = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
    return rows


def plot_bars(rows, metric: str, out_path: str, title: str):
    methods = [r['method'] for r in rows]
    vals = []
    for r in rows:
        v = r.get(metric, '')
        try:
            vals.append(float(v) if v != '' and v.lower() != 'none' else np.nan)
        except Exception:
            vals.append(np.nan)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.bar(methods, vals)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def compute_arcface_scores(orig_dir: str, anon_dir: str, max_images: int = 500) -> List[Tuple[str, float]]:
    files = [f for f in os.listdir(orig_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    files = [f for f in files if os.path.exists(os.path.join(anon_dir, f))]
    files = files[:max_images]
    scores: List[Tuple[str, float]] = []
    for fn in files:
        p1 = os.path.join(orig_dir, fn)
        p2 = os.path.join(anon_dir, fn)
        a = torch.from_numpy(np.array(Image.open(p1).convert('RGB')).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        b = torch.from_numpy(np.array(Image.open(p2).convert('RGB')).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        sim = arcface_similarity(a.unsqueeze(0), b.unsqueeze(0))
        if sim is None:
            continue
        scores.append((fn, float(sim.mean().item())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def export_worst_cases(orig_dir: str, anon_dir: str, out_zip: str, top_k: int = 50) -> None:
    import zipfile
    scores = compute_arcface_scores(orig_dir, anon_dir, max_images=1000)
    top = scores[:top_k]
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for fn, score in top:
            z.write(os.path.join(orig_dir, fn), arcname=os.path.join('orig', fn))
            z.write(os.path.join(anon_dir, fn), arcname=os.path.join('anon', fn))
    # also write a JSON manifest with scores
    with open(os.path.splitext(out_zip)[0] + '_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'scores': top}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='outputs/metrics_all.csv')
    parser.add_argument('--figs_out', default='outputs/figs')
    parser.add_argument('--worst_orig', default=None)
    parser.add_argument('--worst_anon', default=None)
    parser.add_argument('--worst_out_zip', default='outputs/worst_cases_top50.zip')
    args = parser.parse_args()

    rows = load_metrics(args.csv)
    if rows:
        plot_bars(rows, 'fid', os.path.join(args.figs_out, 'fid_bar.png'), 'FID (lower is better)')
        plot_bars(rows, 'arcface_mean_sim', os.path.join(args.figs_out, 'arcface_bar.png'), 'ArcFace mean sim (lower is better)')
        plot_bars(rows, 'map', os.path.join(args.figs_out, 'map_bar.png'), 'mAP AP@[.5:.95] (higher is better)')
        if any('lpips' in r for r in rows[0].keys()):
            plot_bars(rows, 'lpips', os.path.join(args.figs_out, 'lpips_bar.png'), 'LPIPS (lower is better)')
    if args.worst_orig and args.worst_anon:
        export_worst_cases(args.worst_orig, args.worst_anon, args.worst_out_zip, top_k=50)
