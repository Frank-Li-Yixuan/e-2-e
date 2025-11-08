#!/usr/bin/env python
"""
Colab one-click runner:
  1) Optionally mount Google Drive
  2) Auto-detect dataset paths and write configs/paths_overlay.yaml
  3) Launch training (src.train_joint) with planB_colab.yaml and the overlay
  4) Generate a training report from metrics.csv

Designed to also work outside Colab (mount step is skipped).

Example:
  python scripts/colab_oneclick.py --max_steps 500
"""
import argparse
import os
import sys
import subprocess


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def maybe_mount_drive(force: bool = False) -> None:
    if not (force or in_colab()):
        print('[INFO] Not in Colab; skip drive mount')
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        print('[OK] Drive mounted at /content/drive')
    except Exception as e:
        print('[WARN] Failed to mount drive:', e)


def run(cmd: list, env=None) -> int:
    print('[RUN]', ' '.join(cmd))
    return subprocess.call(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/planB_colab.yaml')
    ap.add_argument('--paths-overlay', default='configs/paths_overlay.yaml')
    ap.add_argument('--mode', default='auto', choices=['auto','pretrain','joint'])
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--mount', action='store_true', help='Force mount Drive even if env check fails')
    ap.add_argument('--skip-mount', action='store_true', help='Skip Drive mount explicitly')
    args = ap.parse_args()

    if not args.skip_mount:
        maybe_mount_drive(force=args.mount)

    # 1) Auto-detect dataset paths
    rc = run([sys.executable, 'scripts/colab_paths_autoset.py', '--mode', 'auto', '--output', args.paths_overlay])
    if rc != 0:
        print('[ERROR] colab_paths_autoset failed')
        sys.exit(rc)

    # 2) Launch training
    train_cmd = [
        sys.executable, '-m', 'src.train_joint',
        '--config', args.config,
        '--paths-overlay', args.paths_overlay,
        '--mode', args.mode,
    ]
    if args.max_steps is not None:
        train_cmd += ['--max_steps', str(args.max_steps)]
    rc = run(train_cmd)
    if rc != 0:
        print('[ERROR] Training failed with code', rc)
        sys.exit(rc)

    # 3) Generate training report
    # Try paths.outputs from config via report_training.py --config
    rc = run([sys.executable, 'scripts/report_training.py', '--config', args.config])
    if rc != 0:
        print('[WARN] Report generation via --config failed; trying default outputs path')
        # Fallback to default known Colab overlay outputs
        run([sys.executable, 'scripts/report_training.py', '--out-dir', '/content/outputs/planB'])

    print('[DONE] One-click pipeline finished')


if __name__ == '__main__':
    main()
