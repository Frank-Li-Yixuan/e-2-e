from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import build_dataloaders


class AnonyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build dataloaders lazily and provide sensible fallbacks with clear errors.

        Behavior:
        - If train_images are missing but val_images exist, use val for training with a WARN log.
        - If neither split can be built, raise a RuntimeError with path hints.
        """
        if self._train_loader is None or self._val_loader is None:
            tr, va = build_dataloaders(self.cfg)
            # build_dataloaders already returns DataLoader objects or None
            if tr is None and va is not None:
                print("[WARN] train_images/annotations not found; falling back to val split for training.")
                tr = va
            if tr is None and va is None:
                paths = self.cfg.get("paths", {}) or {}
                raise RuntimeError(
                    "Failed to build any dataloader. Please set cfg.paths: train_images/train_annotations and val_images/val_annotations. "
                    f"Current paths: {paths}"
                )
            self._train_loader = tr
            self._val_loader = va

    def train_dataloader(self) -> DataLoader:
        # Be robust to callers that access train_dataloader before setup()
        if self._train_loader is None:
            try:
                self.setup("fit")
            except Exception:
                self.setup(None)
        if self._train_loader is None:
            raise RuntimeError(
                "Train dataloader is not available. Ensure cfg.paths.train_images exists and points to a valid images root, "
                "and cfg.paths.train_annotations to a valid COCO JSON. You can also run scripts/colab_paths_autoset.py to generate configs/paths_overlay.yaml "
                "and launch with --paths-overlay configs/paths_overlay.yaml."
            )
        return self._train_loader

    def val_dataloader(self) -> Optional[DataLoader]:
        # Likewise, lazily build validation dataloader if needed
        if self._val_loader is None:
            try:
                self.setup("validate")
            except Exception:
                self.setup(None)
        return self._val_loader
