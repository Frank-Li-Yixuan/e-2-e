from typing import Any, Dict, List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import nms

try:
    from transformers import AutoImageProcessor, YolosForObjectDetection  # type: ignore
except Exception:  # pragma: no cover
    AutoImageProcessor = None  # type: ignore
    YolosForObjectDetection = None  # type: ignore


class DetectorWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        dcfg = cfg.get("model", {}).get("detector", {})
        name = dcfg.get("name", "yolos")
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
        self.backend = name  # actual backend in use; may become 'dummy'

        # thresholds
        self.conf_thres = float(dcfg.get("conf_threshold", 0.5))
        self.nms_iou = float(dcfg.get("nms_iou", 0.5))

        if name == "detr":
            self.processor = None
            try:
                self.model = torchvision.models.detection.detr_resnet50(pretrained=dcfg.get("pretrained", True))
                self.model.to(self.device)
            except Exception:
                # Fallback: dummy detector to keep smoke tests offline and avoid API issues
                self.model = None
                self.backend = "dummy"
        elif name == "face_insight":
            # Lightweight face detector using insightface FaceAnalysis; returns label=1 (face)
            self.processor = None
            try:
                import insightface  # type: ignore
                self.insight_app = insightface.app.FaceAnalysis(name="buffalo_l")  # type: ignore[attr-defined]
                self.insight_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
                self.backend = "face_insight"
            except Exception:
                # Fallback to dummy if insightface unavailable
                self.insight_app = None  # type: ignore[attr-defined]
                self.backend = "dummy"
        elif name == "yolos":
            if YolosForObjectDetection is None:
                raise ImportError("Transformers not installed for YOLOS backend")
            hf_id = dcfg.get("hf_model_id", "hustvl/yolos-tiny")
            self.processor = AutoImageProcessor.from_pretrained(hf_id)
            # Our tensors are already in [0,1] after inverse-normalization; avoid double rescale (1/255)
            try:
                if hasattr(self.processor, 'do_rescale'):
                    self.processor.do_rescale = False  # type: ignore[attr-defined]
            except Exception:
                pass
            self.model = YolosForObjectDetection.from_pretrained(hf_id)
            self.model.to(self.device)
        elif name == "yolo_plate":
            # Ultralytics YOLO-based license plate detector; maps all detections to label=2
            try:
                from ultralytics import YOLO  # type: ignore
                model_path = dcfg.get("model_path", "models/plate_yolo11_best.pt")
                if not os.path.exists(model_path):
                    # fallback to project root models path
                    alt = os.path.join(os.getcwd(), model_path)
                    model_path = alt if os.path.exists(alt) else model_path
                self.yolo = YOLO(model_path)
                self.backend = "yolo_plate"
            except Exception:
                self.yolo = None  # type: ignore[attr-defined]
                self.backend = "dummy"
        elif name == "face_plate_hybrid":
            # Combine insightface (faces) + Ultralytics YOLO plates
            self.processor = None
            # face
            try:
                import insightface  # type: ignore
                self.insight_app = insightface.app.FaceAnalysis(name="buffalo_l")  # type: ignore[attr-defined]
                self.insight_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            except Exception:
                self.insight_app = None  # type: ignore[attr-defined]
            # plate
            try:
                from ultralytics import YOLO  # type: ignore
                model_path = dcfg.get("plate_model_path", "models/plate_yolo11_best.pt")
                if not os.path.exists(model_path):
                    alt = os.path.join(os.getcwd(), model_path)
                    model_path = alt if os.path.exists(alt) else model_path
                self.yolo_plate = YOLO(model_path)
            except Exception:
                self.yolo_plate = None  # type: ignore[attr-defined]
            self.backend = "face_plate_hybrid"
        else:
            raise ValueError(f"Unknown detector backend: {name}")

        # Only call nn.Module.eval() for torch nn.Module backends
        if name in ("yolos", "detr"):
            self.eval()

    @torch.no_grad()
    def predict(self, images: torch.Tensor, orig_sizes: Optional[List[Tuple[int, int]]] = None) -> List[Dict[str, Any]]:
        """
        images: [B,3,H,W] normalized to [-1,1] by dataset.
        Returns per-image dict with boxes (xyxy), scores, labels.
        """
        # Avoid calling eval() which may inadvertently trigger non-Module backends' train()
        if self.backend in ("yolos", "detr"):
            self.eval()
        imgs = (images * 0.5 + 0.5).clamp(0, 1)  # back to [0,1]
        outputs: List[Dict[str, Any]] = []
        if self.backend == "dummy":
            # Return empty predictions
            for _ in range(imgs.size(0)):
                outputs.append({
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                })
        elif self.backend == "face_insight":
            # Per-image run with insightface; label=1 for faces
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
            for i in range(imgs.size(0)):
                arr = (imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")
                # insightface expects BGR np.ndarray
                bgr = arr[:, :, ::-1].copy()
                faces = [] if self.insight_app is None else self.insight_app.get(bgr)  # type: ignore[attr-defined]
                boxes_list: List[List[float]] = []
                scores_list: List[float] = []
                labels_list: List[int] = []
                for f in faces:
                    # f.bbox: [x1,y1,x2,y2]
                    x1, y1, x2, y2 = [float(v) for v in f.bbox]
                    score = float(getattr(f, 'det_score', 1.0))
                    if score < self.conf_thres:
                        continue
                    boxes_list.append([x1, y1, x2, y2])
                    scores_list.append(score)
                    labels_list.append(1)  # face class id in our COCO
                if len(boxes_list) > 0:
                    b = torch.tensor(boxes_list, dtype=torch.float32)
                    s = torch.tensor(scores_list, dtype=torch.float32)
                    l = torch.tensor(labels_list, dtype=torch.long)
                    keep_idx = nms(b, s, self.nms_iou)
                    b = b[keep_idx]
                    s = s[keep_idx]
                    l = l[keep_idx]
                else:
                    b = torch.zeros((0, 4), dtype=torch.float32)
                    s = torch.zeros((0,), dtype=torch.float32)
                    l = torch.zeros((0,), dtype=torch.long)
                outputs.append({
                    "boxes": b,
                    "scores": s,
                    "labels": l,
                })
        elif self.name == "detr":
            # DETR expects list of tensors in 0..1, normalized internally
            preds = self.model([img.to(self.device) for img in imgs])  # type: ignore[arg-type]
            for p in preds:
                boxes = p["boxes"]
                scores = p["scores"]
                labels = p["labels"]
                keep = scores > self.conf_thres
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                # nms
                if boxes.numel() > 0:
                    keep_idx = nms(boxes, scores, self.nms_iou)
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]
                    labels = labels[keep_idx]
                outputs.append({
                    "boxes": boxes.detach().cpu(),
                    "scores": scores.detach().cpu(),
                    "labels": labels.detach().cpu(),
                })
        elif self.name == "yolos":
            assert self.processor is not None
            inputs = self.processor(list(imgs.detach().cpu()), return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pred = self.model(**inputs)
            target_sizes = inputs.get("pixel_values").shape[-2:]  # type: ignore
            r = self.processor.post_process_object_detection(pred, threshold=self.conf_thres, target_sizes=[target_sizes for _ in range(imgs.size(0))])  # type: ignore
            for rr in r:
                boxes = rr.get("boxes", torch.zeros((0, 4), device=self.device))
                scores = rr.get("scores", torch.zeros((0,), device=self.device))
                labels = rr.get("labels", torch.zeros((0,), device=self.device, dtype=torch.long))
                boxes = boxes.to(self.device)
                scores = scores.to(self.device)
                labels = labels.to(self.device)
                if boxes.numel() > 0:
                    keep_idx = nms(boxes, scores, self.nms_iou)
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]
                    labels = labels[keep_idx]
                outputs.append({
                    "boxes": boxes.detach().cpu(),
                    "scores": scores.detach().cpu(),
                    "labels": labels.detach().cpu(),
                })
        # YOLO plate backend
        elif self.backend == "yolo_plate":
            try:
                # run per image
                from ultralytics.engine.results import Results  # type: ignore
                for i in range(imgs.size(0)):
                    arr = (imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")
                    res = self.yolo.predict(arr, verbose=False, conf=self.conf_thres)  # type: ignore[attr-defined]
                    if len(res) > 0 and getattr(res[0], 'boxes', None) is not None:
                        bxyxy = res[0].boxes.xyxy.detach().cpu()  # type: ignore[attr-defined]
                        scores = res[0].boxes.conf.detach().cpu()  # type: ignore[attr-defined]
                        labels = torch.full((bxyxy.shape[0],), 2, dtype=torch.long)  # license_plate=2
                        if bxyxy.numel() > 0:
                            keep_idx = nms(bxyxy, scores, self.nms_iou)
                            bxyxy = bxyxy[keep_idx]
                            scores = scores[keep_idx]
                            labels = labels[keep_idx]
                    else:
                        bxyxy = torch.zeros((0, 4), dtype=torch.float32)
                        scores = torch.zeros((0,), dtype=torch.float32)
                        labels = torch.zeros((0,), dtype=torch.long)
                    outputs.append({
                        "boxes": bxyxy,
                        "scores": scores,
                        "labels": labels,
                    })
            except Exception:
                # fallback: no detections
                for _ in range(imgs.size(0)):
                    outputs.append({
                        "boxes": torch.zeros((0, 4)),
                        "scores": torch.zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.long),
                    })
        # Hybrid backend merge face + plate
        elif self.backend == "face_plate_hybrid":
            import numpy as np  # type: ignore
            merged_outputs: List[Dict[str, torch.Tensor]] = []
            for i in range(imgs.size(0)):
                # faces
                arr = (imgs[i].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")
                b_faces = torch.zeros((0, 4), dtype=torch.float32)
                s_faces = torch.zeros((0,), dtype=torch.float32)
                l_faces = torch.zeros((0,), dtype=torch.long)
                if self.insight_app is not None:
                    try:
                        bgr = arr[:, :, ::-1].copy()
                        faces = self.insight_app.get(bgr)  # type: ignore[attr-defined]
                        if faces:
                            fb, fs = [], []
                            for f in faces:
                                x1, y1, x2, y2 = [float(v) for v in f.bbox]
                                score = float(getattr(f, 'det_score', 1.0))
                                if score >= self.conf_thres:
                                    fb.append([x1, y1, x2, y2])
                                    fs.append(score)
                            if len(fb) > 0:
                                b_faces = torch.tensor(fb, dtype=torch.float32)
                                s_faces = torch.tensor(fs, dtype=torch.float32)
                                l_faces = torch.full((b_faces.shape[0],), 1, dtype=torch.long)  # face=1
                    except Exception:
                        pass
                # plates
                b_plates = torch.zeros((0, 4), dtype=torch.float32)
                s_plates = torch.zeros((0,), dtype=torch.float32)
                l_plates = torch.zeros((0,), dtype=torch.long)
                if getattr(self, 'yolo_plate', None) is not None:
                    try:
                        res = self.yolo_plate.predict(arr, verbose=False, conf=self.conf_thres)  # type: ignore[attr-defined]
                        if len(res) > 0 and getattr(res[0], 'boxes', None) is not None:
                            bxyxy = res[0].boxes.xyxy.detach().cpu()  # type: ignore[attr-defined]
                            scores = res[0].boxes.conf.detach().cpu()  # type: ignore[attr-defined]
                            if bxyxy.numel() > 0:
                                keep_idx = nms(bxyxy, scores, self.nms_iou)
                                bxyxy = bxyxy[keep_idx]
                                scores = scores[keep_idx]
                            b_plates = bxyxy
                            s_plates = scores
                            l_plates = torch.full((bxyxy.shape[0],), 2, dtype=torch.long)
                    except Exception:
                        pass
                # merge
                boxes = torch.cat([b_faces, b_plates], dim=0) if (b_faces.numel() + b_plates.numel()) > 0 else torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.cat([s_faces, s_plates], dim=0) if (s_faces.numel() + s_plates.numel()) > 0 else torch.zeros((0,), dtype=torch.float32)
                labels = torch.cat([l_faces, l_plates], dim=0) if (l_faces.numel() + l_plates.numel()) > 0 else torch.zeros((0,), dtype=torch.long)
                merged_outputs.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })
            outputs = merged_outputs
        else:
            # Unknown/unsupported backend: return empty
            for _ in range(imgs.size(0)):
                outputs.append({
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                })
        return outputs

    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute detector loss given annotated targets. Returns (loss, scalars)."""
        self.train()
        imgs = (batch["images"].to(self.device) * 0.5 + 0.5).clamp(0, 1)
        targets: List[Dict[str, torch.Tensor]] = []
        for i in range(len(batch["boxes"])):
            targets.append({
                "boxes": batch["boxes"][i].to(self.device),
                "labels": (batch["labels"][i].to(self.device) if batch["labels"][i].numel() > 0 else torch.zeros((0,), dtype=torch.long, device=self.device)),
            })
        if self.backend == "dummy":
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            scalars = {}
            return loss, scalars
        if self.name == "detr":
            loss_dict = self.model([img for img in imgs], targets)  # type: ignore[arg-type]
            loss = sum(v for v in loss_dict.values())
            scalars = {f"det/{k}": float(v.detach().cpu()) for k, v in loss_dict.items()}
            return loss, scalars
        else:
            assert self.processor is not None and YolosForObjectDetection is not None
            # Hugging Face YOLOS supervised loss path
            inputs = self.processor(list(imgs.detach().cpu()), return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Convert boxes to relative cxcywh as required by transformers
            h, w = inputs["pixel_values"].shape[-2:]  # type: ignore
            labels = []
            for t in targets:
                boxes_xyxy = t["boxes"].detach().cpu()
                if boxes_xyxy.numel() == 0:
                    labels.append({"class_labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                                   "boxes": torch.zeros((0, 4), dtype=torch.float, device=self.device)})
                    continue
                cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / w
                cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / h
                bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / w
                bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / h
                boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=-1).to(self.device)
                class_labels = t["labels"].to(self.device)
                labels.append({"class_labels": class_labels, "boxes": boxes_cxcywh})
            out = self.model(**inputs, labels=labels)  # type: ignore
            loss = out.loss
            scalars = {"det/yolos_loss": float(loss.detach().cpu())}
            return loss, scalars

    def save(self, path: str) -> None:
        state = {
            "name": self.name,
            "state_dict": self.model.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("name") != self.name:
            raise ValueError("Detector name mismatch in checkpoint")
        self.model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
