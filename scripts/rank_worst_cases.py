#!/usr/bin/env python
"""
Rank worst-case pairs by ArcFace similarity (higher = more identity leakage),
and copy top-K originals/anonymized images into an output folder. Optionally write a CSV and a zip.

Usage:
  python scripts/rank_worst_cases.py \
    --orig_dir outputs/aligned/e2e_val/orig \
    --anon_dir outputs/aligned/e2e_val/anon \
    --top_k 50 \
    --out_dir outputs/worst_cases/e2e_val \
    --csv outputs/worst_cases/e2e_val_scores.csv \
    --zip outputs/worst_cases/e2e_val_top50.zip
"""
import argparse
import csv
import os
import shutil
import zipfile
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# Ensure repo root on sys.path so `src` package resolves when running as a script
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.losses import arcface_similarity  # type: ignore


@torch.no_grad()
def compute_arcface_per_image(orig_dir: str, anon_dir: str, limit: int = 0) -> List[Tuple[str, float]]:
    files = [f for f in os.listdir(orig_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    files = [f for f in files if os.path.exists(os.path.join(anon_dir, f))]
    if limit:
        files = files[:limit]
    scores: List[Tuple[str, float]] = []
    for fn in files:
        p1 = os.path.join(orig_dir, fn)
        p2 = os.path.join(anon_dir, fn)
        a = torch.from_numpy(np.array(Image.open(p1).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        b = torch.from_numpy(np.array(Image.open(p2).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1) * 2 - 1
        sim = arcface_similarity(a.unsqueeze(0), b.unsqueeze(0))
        val = float(sim.mean().item()) if sim is not None else 0.0
        scores.append((fn, val))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig_dir', required=True)
    ap.add_argument('--anon_dir', required=True)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--zip', dest='zip_path', default=None)
    ap.add_argument('--limit', type=int, default=0, help='Optional cap for how many files to score before ranking')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scores = compute_arcface_per_image(args.orig_dir, args.anon_dir, args.limit)
    # Sort descending: higher similarity is worse (more identity leakage)
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[: args.top_k]

    # Write CSV if requested
    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['filename', 'arcface_sim'])
            for fn, val in scores:
                w.writerow([fn, f"{val:.6f}"])

    # Copy top-K into out_dir preserving two subfolders
    out_orig = os.path.join(args.out_dir, 'orig')
    out_anon = os.path.join(args.out_dir, 'anon')
    os.makedirs(out_orig, exist_ok=True)
    os.makedirs(out_anon, exist_ok=True)
    for fn, _ in top:
        shutil.copyfile(os.path.join(args.orig_dir, fn), os.path.join(out_orig, fn))
        shutil.copyfile(os.path.join(args.anon_dir, fn), os.path.join(out_anon, fn))

    # Optional zip
    if args.zip_path:
        os.makedirs(os.path.dirname(args.zip_path), exist_ok=True)
        with zipfile.ZipFile(args.zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for root in [out_orig, out_anon]:
                for fn in os.listdir(root):
                    zf.write(os.path.join(root, fn), arcname=os.path.join(os.path.basename(root), fn))
        print('Wrote zip:', args.zip_path)

    print(f'Done. Ranked {len(scores)} files, exported top-{len(top)} to {args.out_dir}')


if __name__ == '__main__':
    main()
