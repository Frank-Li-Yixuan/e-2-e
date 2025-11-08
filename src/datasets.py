import os
import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.ops import box_convert
from torchvision import transforms as T

try:
    from pycocotools.coco import COCO  # type: ignore
except Exception:  # pragma: no cover
    COCO = None  # type: ignore


def pil_load_rgb(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def pil_load_mask(path: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    if not os.path.exists(path):
        return Image.new("L", size if size else (512, 512), color=0)
    m = Image.open(path).convert("L")
    if size is not None and (m.size[0] != size[0] or m.size[1] != size[1]):
        m = m.resize(size, Image.NEAREST)
    return m


class CocoFaces(Dataset):
    """
    Generic COCO-style dataset for faces or other regions. If annotations are missing,
    dataset acts as image-only with optional mask/pseudotarget lookup by filename.
    """

    def __init__(
        self,
        images_dir: str,
        annotations: Optional[str] = None,
        masks_dir: Optional[str] = None,
        pseudotargets_dir: Optional[str] = None,
        image_size: int = 512,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.annotations = annotations
        self.masks_dir = masks_dir
        self.pseudotargets_dir = pseudotargets_dir
        self.image_size = image_size
        self.normalize = normalize

        # populate file list: prefer COCO image list (supports subdirectories in file_name)
        self.fnames: List[str] = []

        self.coco = None
        self.id_to_filename: Dict[int, str] = {}
        self.filename_to_id: Dict[str, int] = {}
        self.catid_to_name: Dict[int, str] = {}

        if annotations and COCO is not None and os.path.exists(annotations):
            self.coco = COCO(annotations)
            for img in self.coco.dataset.get("images", []):
                fn = str(img.get("file_name", ""))
                # normalize to forward slashes for consistency; join will handle on Windows
                fn = fn.replace("\\", "/")
                self.id_to_filename[img["id"]] = fn
                self.filename_to_id[fn] = img["id"]
                self.fnames.append(fn)
            for cat in self.coco.dataset.get("categories", []):
                self.catid_to_name[cat["id"]] = cat["name"]
        else:
            # Fallback: list files directly from directory (flat, non-recursive)
            self.fnames = sorted([
                f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST)
        self.normalize_tf = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if normalize else None

    def __len__(self) -> int:
        return len(self.fnames)

    def _load_coco_targets(self, filename: str) -> Dict[str, Any]:
        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        if self.coco is None:
            return {"boxes": boxes_xyxy, "labels": labels}
        img_id = self.filename_to_id.get(filename)
        if img_id is None:
            return {"boxes": boxes_xyxy, "labels": labels}
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        for a in anns:
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a.get("category_id", 1))
        return {"boxes": boxes_xyxy, "labels": labels}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = self.fnames[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = pil_load_rgb(img_path)
        w, h = img.size

        mask = None
        if self.masks_dir:
            mask_path = os.path.join(self.masks_dir, os.path.splitext(fname)[0] + ".png")
            mask = pil_load_mask(mask_path, size=(w, h))
        else:
            mask = Image.new("L", (w, h), color=0)

        pseudo = None
        if self.pseudotargets_dir:
            p_path = os.path.join(self.pseudotargets_dir, fname)
            if os.path.exists(p_path):
                pseudo = pil_load_rgb(p_path)

        targets = self._load_coco_targets(fname)

        # basic resizing
        img_t = self.to_tensor(self.resize(img))  # [3,H,W] in 0..1
        mask_t = T.functional.pil_to_tensor(self.resize_mask(mask)).float() / 255.0  # [1,H,W]
        if self.normalize_tf:
            img_t = self.normalize_tf(img_t)

        pseudo_t = None
        if pseudo is not None:
            pseudo_t = self.to_tensor(self.resize(pseudo))
            if self.normalize_tf:
                pseudo_t = self.normalize_tf(pseudo_t)

        # scale boxes to resized size if any
        boxes_xyxy = targets.get("boxes", [])
        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32) if len(boxes_xyxy) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        if boxes.numel() > 0:
            # scale from original W,H to resized image_size
            scale_x = self.image_size / float(w)
            scale_y = self.image_size / float(h)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        labels = torch.tensor(targets.get("labels", []), dtype=torch.int64) if len(targets.get("labels", [])) > 0 else torch.zeros((0,), dtype=torch.int64)

        return {
            "image": img_t,            # normalized [-1,1] if normalize=True
            "mask": mask_t,             # [1,H,W] in [0,1]
            "pseudo": pseudo_t,         # optional
            "boxes": boxes,             # [N,4] xyxy
            "labels": labels,           # [N]
            "filename": fname,
            "orig_size": (h, w),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # stack images/masks; keep lists for variable-size targets
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    pseudos = [b["pseudo"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    filenames = [b["filename"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    return {
        "images": images,
        "masks": masks,
        "pseudos": pseudos,
        "boxes": boxes,
        "labels": labels,
        "filenames": filenames,
        "orig_sizes": orig_sizes,
    }


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    paths = cfg.get("paths", {})
    image_size = cfg.get("data", {}).get("image_size", 512)
    normalize = cfg.get("data", {}).get("normalize", True)
    sample_strategy = cfg.get("data", {}).get("sample_strategy", "none")  # none | oversample_face | oversample_plate | balanced

    def make_dl(split: str) -> Optional[DataLoader]:
        img_dir = paths.get(f"{split}_images")
        if not img_dir or not os.path.exists(img_dir):
            return None
        ds = CocoFaces(
            images_dir=img_dir,
            annotations=paths.get(f"{split}_annotations"),
            masks_dir=paths.get(f"{split}_masks"),
            pseudotargets_dir=paths.get("pseudotargets"),
            image_size=image_size,
            normalize=normalize,
        )
        sampler = None
        shuffle = (split == "train")
        if split == "train" and sample_strategy != "none":
            # build per-image weights based on presence of classes
            weights = []
            for i in range(len(ds)):
                # read targets via __getitem__ lightweight path
                item = ds[i]
                labels = set(item["labels"].tolist()) if item["labels"].numel() > 0 else set()
                w = 1.0
                if sample_strategy == "oversample_face":
                    if 1 in labels:
                        w = 3.0
                elif sample_strategy == "oversample_plate":
                    if 2 in labels:
                        w = 3.0
                elif sample_strategy == "balanced":
                    # boost images that contain fewer represented classes (face/plate)
                    contains = (1 in labels) + (2 in labels)
                    w = {0: 0.5, 1: 1.0, 2: 2.0}.get(contains, 1.0)
                weights.append(w)
            sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
            shuffle = False
        dl = DataLoader(
            ds,
            batch_size=cfg.get("train", {}).get("batch_size", 2),
            num_workers=cfg.get("train", {}).get("num_workers", 4),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )
        return dl

    return make_dl("train"), make_dl("val")
