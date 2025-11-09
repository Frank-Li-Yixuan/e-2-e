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
import torch.nn.functional as F

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
        auto_box_mask: bool = False,
        box_mask_dilate: int = 0,
        box_mask_soft_edges: int = 0,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.annotations = annotations
        self.masks_dir = masks_dir
        self.pseudotargets_dir = pseudotargets_dir
        self.image_size = image_size
        self.normalize = normalize
        self.auto_box_mask = auto_box_mask
        self.box_mask_dilate = int(box_mask_dilate)
        self.box_mask_soft_edges = int(box_mask_soft_edges)

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

        # If no explicit mask provided and auto_box_mask enabled, synthesize mask from boxes
        if self.auto_box_mask and mask_t.sum().item() == 0 and boxes.numel() > 0:
            # boxes are in resized coordinate space already
            H, W = mask_t.shape[1], mask_t.shape[2]
            synth = torch.zeros((H, W), dtype=torch.float32)
            for b in boxes:
                x1, y1, x2, y2 = b.tolist()
                # clamp & int conversion
                x1i = max(0, min(W - 1, int(x1)))
                y1i = max(0, min(H - 1, int(y1)))
                x2i = max(0, min(W, int(x2)))
                y2i = max(0, min(H, int(y2)))
                synth[y1i:y2i, x1i:x2i] = 1.0
            # Optional dilation using max-pooling like expansion
            if self.box_mask_dilate > 0:
                k = int(self.box_mask_dilate)
                if k > 0:
                    # create pooling window (approximate dilation)
                    pad = k
                    synth_pad = F.pad(synth.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), value=0)
                    # use max-pooling with stride 1 to dilate
                    synth_d = F.max_pool2d(synth_pad, kernel_size=1 + 2 * k, stride=1)
                    synth = synth_d.squeeze(0).squeeze(0)[:, :W][:H, :]
                    synth = synth[:H, :W]
            # Optional soft edges via average pooling (feathering)
            if self.box_mask_soft_edges > 0:
                r = int(self.box_mask_soft_edges)
                if r > 0:
                    # simple blur then renormalize to [0,1]
                    synth_blur = F.avg_pool2d(synth.unsqueeze(0).unsqueeze(0), kernel_size=1 + 2 * r, stride=1, padding=r)
                    synth = synth_blur.squeeze(0).squeeze(0).clamp(0, 1)
            mask_t = synth.unsqueeze(0)

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
    auto_box_mask = cfg.get("data", {}).get("auto_box_mask", False)
    box_mask_dilate = cfg.get("data", {}).get("box_mask_dilate", 0)
    box_mask_soft_edges = cfg.get("data", {}).get("box_mask_soft_edges", 0)

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
            auto_box_mask=auto_box_mask,
            box_mask_dilate=box_mask_dilate,
            box_mask_soft_edges=box_mask_soft_edges,
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
        # DataLoader stability knobs
        # Prefer data.num_workers (dataset specific) then train.num_workers, fallback to 4
        num_workers = int(
            cfg.get("data", {}).get("num_workers",
                    cfg.get("train", {}).get("num_workers", 4))
        )
        persistent_workers = bool(cfg.get("train", {}).get("persistent_workers", False)) and num_workers > 0
        prefetch_factor = int(cfg.get("train", {}).get("prefetch_factor", 2)) if num_workers > 0 else None

        dl_kwargs: Dict[str, Any] = dict(
            dataset=ds,
            batch_size=cfg.get("train", {}).get("batch_size", 2),
            num_workers=num_workers,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = max(2, prefetch_factor)
        dl = DataLoader(**dl_kwargs)
        return dl

    return make_dl("train"), make_dl("val")
