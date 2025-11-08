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

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
        precision=precision,
        max_steps=max_steps,
        gradient_clip_val=grad_clip if grad_clip > 0 else None,
        accumulate_grad_batches=accum,
        log_every_n_steps=50,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
