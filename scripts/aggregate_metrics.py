#!/usr/bin/env python
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='outputs/eval_report_*.json')
    ap.add_argument('--out', default='outputs/metrics_all.csv')
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    cols = ['method', 'map', 'fid', 'lpips', 'psnr', 'ssim', 'arcface_mean_sim', 'easyocr_avg_words']
    rows = [','.join(cols)]
    for fp in files:
        base = os.path.basename(fp)
        method = base.replace('eval_report_', '').replace('.json', '')
        try:
            data = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            data = {}
        vals = [
            method,
            str(data.get('map', '')),
            str(data.get('fid', '')),
            str(data.get('lpips', '')),
            str(data.get('psnr', '')),
            str(data.get('ssim', '')),
            str(data.get('arcface_mean_sim', '')),
            str(data.get('easyocr_avg_words', '')),
        ]
        rows.append(','.join(vals))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))
    print('Wrote', args.out, 'from', len(files), 'files')


if __name__ == '__main__':
    main()
