from typing import Dict, Any, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import vgg19, VGG19_Weights  # type: ignore
except Exception:  # pragma: no cover
    vgg19 = None  # type: ignore
    VGG19_Weights = None  # type: ignore

_arcface_model = None


def get_device(cfg: Dict[str, Any]) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")


class PerceptualLoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        # Avoid heavy downloads by default; enable with USE_VGG_PERCEPTUAL=1
        self.vgg = None
        if vgg19 is not None and os.environ.get("USE_VGG_PERCEPTUAL", "0") == "1":
            try:
                self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].to(device)  # up to relu3_1
                for p in self.vgg.parameters():
                    p.requires_grad = False
            except Exception:
                self.vgg = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return F.l1_loss(x, y)
        # x,y in [-1,1] -> to [0,1] -> normalize imagenet
        def prep(t: torch.Tensor) -> torch.Tensor:
            t = (t * 0.5 + 0.5).clamp(0, 1)
            return (t - self.mean) / self.std
        fx = self.vgg(prep(x))
        fy = self.vgg(prep(y))
        return F.l1_loss(fx, fy)


def adversarial_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))


def adversarial_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
    loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) * 0.5


def _get_arcface() -> Optional[Any]:
    global _arcface_model
    if _arcface_model is not None:
        return _arcface_model
    try:
        import insightface  # type: ignore
        model = insightface.app.FaceAnalysis(name="buffalo_l")
        model.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        _arcface_model = model
        return _arcface_model
    except Exception:
        return None


@torch.no_grad()
def arcface_similarity(orig: torch.Tensor, anonym: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Compute cosine similarity of ArcFace embeddings between original and anonymized images.
    Inputs: [B,3,H,W] in [-1,1]. Returns tensor [B] or None if unavailable.
    """
    model = _get_arcface()
    if model is None:
        return None
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    def to_uint8(t: torch.Tensor) -> Image.Image:
        arr = ((t.clamp(-1, 1) * 0.5 + 0.5) * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        return Image.fromarray(arr)

    sims = []
    for i in range(orig.size(0)):
        img1 = np.asarray(to_uint8(orig[i]))
        img2 = np.asarray(to_uint8(anonym[i]))
        # get embeddings from detected face crops; if none, skip
        feat1 = None
        faces1 = model.get(img1)
        if len(faces1) > 0:
            feat1 = faces1[0].normed_embedding
        feat2 = None
        faces2 = model.get(img2)
        if len(faces2) > 0:
            feat2 = faces2[0].normed_embedding
        if feat1 is None or feat2 is None:
            sims.append(torch.tensor(0.0))
        else:
            f1 = torch.tensor(feat1)
            f2 = torch.tensor(feat2)
            sims.append(F.cosine_similarity(f1, f2, dim=0).clamp(-1, 1))
    return torch.stack(sims)
