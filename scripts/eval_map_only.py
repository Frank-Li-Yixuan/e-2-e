#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml

# Ensure repo root on sys.path so `src` package resolves when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval_utils import eval_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/joint_small.yaml')
    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--gt_annotations', type=str, required=True)
    ap.add_argument('--max_images', type=int, default=200)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    m = eval_map(cfg, args.images_dir, args.gt_annotations, args.max_images)
    res = {'map': float(m) if m is not None else None}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    md = os.path.splitext(args.out)[0] + '.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Eval (mAP only)\n\n')
        f.write(f"- mAP (AP@.5:.95): {res['map'] if res['map'] is not None else 'NaN'}\n")
    print('Wrote', args.out, 'and', md)


if __name__ == '__main__':
    main()
