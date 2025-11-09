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
        if self._train_loader is None or self._val_loader is None:
            tr, va = build_dataloaders(self.cfg)
            # build_dataloaders already returns DataLoader objects
            self._train_loader = tr
            self._val_loader = va

    def train_dataloader(self) -> DataLoader:
        # Be robust to callers that access train_dataloader before setup()
        if self._train_loader is None:
            try:
                self.setup("fit")
            except Exception:
                self.setup(None)
        assert self._train_loader is not None, "DataModule not set up"
        return self._train_loader

    def val_dataloader(self) -> Optional[DataLoader]:
        # Likewise, lazily build validation dataloader if needed
        if self._val_loader is None:
            try:
                self.setup("validate")
            except Exception:
                self.setup(None)
        return self._val_loader
