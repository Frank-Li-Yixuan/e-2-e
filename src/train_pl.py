import argparse
import os
from typing import Any, Dict

import yaml
import pytorch_lightning as pl
import torch

from .pl_module import AnonyLightningModule
from .data_module import AnonyDataModule


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--paths-overlay", type=str, default=None)
    # Allow overriding max_steps from CLI (supports both --max_steps and --max-steps)
    parser.add_argument("--max_steps", "--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.paths_overlay and os.path.exists(args.paths_overlay):
        try:
            with open(args.paths_overlay, 'r', encoding='utf-8') as f:
                overlay = yaml.safe_load(f) or {}
            if isinstance(overlay, dict) and 'paths' in overlay and isinstance(overlay['paths'], dict):
                base = cfg.get('paths', {}) or {}
                base.update(overlay['paths'])
                cfg['paths'] = base
                print(f"[INFO] Applied paths overlay from {args.paths_overlay}")
        except Exception as e:
            print(f"[WARN] Failed to apply paths overlay: {e}")

    # Override max_steps if provided via CLI
    if args.max_steps is not None:
        cfg.setdefault("train", {})
        cfg["train"]["max_steps"] = int(args.max_steps)
        print(f"[INFO] Overriding train.max_steps to {int(args.max_steps)} via CLI")

    # Auto-apply default paths overlay if not provided and file exists
    if not args.paths_overlay:
        default_overlay = os.path.join(os.path.dirname(args.config), 'paths_overlay.yaml')
        if os.path.exists(default_overlay):
            try:
                with open(default_overlay, 'r', encoding='utf-8') as f:
                    overlay = yaml.safe_load(f) or {}
                if isinstance(overlay, dict) and 'paths' in overlay and isinstance(overlay['paths'], dict):
                    base = cfg.get('paths', {}) or {}
                    base.update(overlay['paths'])
                    cfg['paths'] = base
                    print(f"[INFO] Auto-applied default paths overlay: {default_overlay}")
            except Exception as e:
                print(f"[WARN] Failed to auto-apply default overlay: {e}")

    # If still missing essential paths, print a helpful hint
    p = cfg.get('paths', {}) or {}
    if not p.get('train_images') or not p.get('val_images'):
        print("[HINT] paths.train_images/val_images not set. You can generate configs/paths_overlay.yaml by running:\n"
              "  %run scripts/colab_paths_autoset.py --mode auto\n"
              "and then pass: --paths-overlay configs/paths_overlay.yaml")

    # Reproducibility (optional deterministic)
    try:
        seed = int(cfg.get("seed", 42))
        pl.seed_everything(seed, workers=True)
        if bool(cfg.get("repro", {}).get("deterministic", False)):
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # Enable Tensor Cores matmul for Ampere+ GPUs (A100) for speed
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
            print("[INFO] Set torch.set_float32_matmul_precision('high') for Tensor Cores.")
        except Exception:
            pass

    model = AnonyLightningModule(cfg)
    dm = AnonyDataModule(cfg)

    # Precision mapping
    prec = str(cfg.get("precision", "fp32")).lower()
    if prec in ("fp16", "16", "16-mixed"):
        precision = "16-mixed"
    elif prec in ("bf16", "bf16-mixed"):
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    max_steps = cfg.get("train", {}).get("max_steps", None)
    grad_clip = float(cfg.get("grad", {}).get("clip_norm", 0.0) or 0.0)
    accum = int(cfg.get("train", {}).get("grad_accum_steps", 1) or 1)

    # Eval cadence controls (optional, with safe defaults)
    eval_cfg = cfg.get("eval", {})
    val_check_interval = eval_cfg.get("val_check_interval", None)
    limit_val_batches = int(eval_cfg.get("limit_val_batches", 0) or 0)

    # Manual optimization is used in AnonyLightningModule, so we must NOT pass gradient_clip_val to Trainer.
    # Gradient clipping is performed inside training_step manually.
    # Allow configuring or disabling sanity validation steps to avoid noisy failures on empty val
    num_sanity_val_steps = int(eval_cfg.get("num_sanity_val_steps", 0) or 0)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision=precision,
        max_steps=max_steps,
        # gradient_clip_val removed due to manual optimization
        accumulate_grad_batches=accum,
        log_every_n_steps=50,
        enable_checkpointing=False,
        enable_progress_bar=True,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches if limit_val_batches > 0 else None,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
