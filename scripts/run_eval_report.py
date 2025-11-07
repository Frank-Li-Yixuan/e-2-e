#!/usr/bin/env python
"""
Scan latest val_samples (or given dirs) and compute metrics via eval_utils,
writing outputs/eval_report.json and outputs/eval_report.md.
"""
import argparse
import glob
import os
import json
from pathlib import Path

from src import eval_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig_dir', default=None)
    ap.add_argument('--anon_dir', default=None)
    ap.add_argument('--out_json', default='outputs/eval_report.json')
    ap.add_argument('--gt_annotations', default=None)
    ap.add_argument('--max_images', type=int, default=50)
    args = ap.parse_args()

    if not args.orig_dir or not args.anon_dir:
        # try locate latest under outputs/val_samples/*
        cand = sorted(Path('outputs/val_samples').glob('step*/'), key=os.path.getmtime, reverse=True)
        if cand:
            latest = cand[0]
            args.orig_dir = str(latest / 'orig')
            args.anon_dir = str(latest / 'anon')

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out_md = os.path.splitext(args.out_json)[0] + '.md'

    # call eval_utils (using its CLI-like API)
    # Here we import functions and call them directly
    res = {
        'arcface_mean_sim': None,
        'easyocr_avg_words': None,
        'fid': None,
        'map': None,
    }
    sim = eval_utils.eval_arcface(args.orig_dir, args.anon_dir, args.max_images)
    if sim is not None:
        res['arcface_mean_sim'] = float(sim)
    ocr = eval_utils.eval_easyocr(args.anon_dir, args.max_images)
    if ocr is not None:
        res['easyocr_avg_words'] = float(ocr)
    fid = eval_utils.eval_fid(args.orig_dir, args.anon_dir, args.max_images)
    if fid is not None:
        res['fid'] = float(fid)
    if args.gt_annotations:
        # map requires a config; we call via CLI wrapper instead (not provided here), so leave None
        pass

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Eval Report\n\n')
        for k, v in res.items():
            f.write(f'- {k}: {v if v is not None else "NaN"}\n')
    print('Wrote', args.out_json, 'and', out_md)


if __name__ == '__main__':
    main()
