#!/usr/bin/env python
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='outputs/metrics_all.csv')
    ap.add_argument('--out', default='outputs/eval_report.md')
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.csv) and os.path.getsize(args.csv) > 0:
        with open(args.csv, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('# Eval Report (Baselines)\n\n')
        if rows:
            # Determine available columns
            cols = ['method']
            metric_cols = ['map', 'fid', 'arcface_mean_sim', 'easyocr_avg_words']
            cols.extend([c for c in metric_cols if c in rows[0]])

            # Header
            header_names = {
                'method': 'method',
                'map': 'mAP(AP@.5:.95)',
                'fid': 'FID',
                'arcface_mean_sim': 'ArcFace mean sim (↓)',
                'easyocr_avg_words': 'EasyOCR words (↓)',
            }
            f.write('| ' + ' | '.join(header_names.get(c, c) for c in cols) + ' |\n')
            f.write('|'+ '|'.join(['---']*len(cols)) + '|\n')

            # Rows
            for r in rows:
                values = [r.get(c, '') for c in cols]
                f.write('| ' + ' | '.join(values) + ' |\n')
        else:
            f.write('No metrics yet.')
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
