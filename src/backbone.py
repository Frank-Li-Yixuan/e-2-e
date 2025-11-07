from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover
    AutoImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore


class BackboneOutputs:
    def __init__(self, tokens: Optional[torch.Tensor] = None, maps: Optional[List[torch.Tensor]] = None) -> None:
        # tokens: [B, N, C]
        # maps: list of [B, C, H, W] feature maps
        self.tokens = tokens
        self.maps = maps or []


class ViTBackbone(nn.Module):
    """
    Minimal ViT/DETR/YOLOS-like backbone wrapper that exposes token embeddings and a coarse feature map.
    Intended as a shared encoder for detection and generator conditioning.
    """

    def __init__(self, model_id: str = "hustvl/yolos-tiny", device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.available = (AutoModel is not None and AutoImageProcessor is not None)
        if self.available:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                # avoid double 1/255 rescale
                if hasattr(self.processor, "do_rescale"):
                    try:
                        self.processor.do_rescale = False  # type: ignore[attr-defined]
                    except Exception:
                        pass
                self.backbone = AutoModel.from_pretrained(model_id, output_hidden_states=True)
                self.backbone.to(self.device)
                self.backbone.eval()
            except Exception:
                self.available = False
                self.processor = None  # type: ignore
                self.backbone = None  # type: ignore
        else:
            self.processor = None  # type: ignore
            self.backbone = None  # type: ignore

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> BackboneOutputs:
        """
        images: [B,3,H,W] in [-1,1]
        returns tokens [B,N,C] and a coarse map [B,C,Hp,Wp] if possible
        """
        if not self.available:
            return BackboneOutputs(tokens=None, maps=[])
        imgs_01 = (images * 0.5 + 0.5).clamp(0, 1)
        inputs = self.processor(list(imgs_01.detach().cpu()), return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.backbone(**inputs)
        # hidden_states: tuple of [B,N,C]
        hs = out.hidden_states if hasattr(out, "hidden_states") else None
        tokens = hs[-1] if hs is not None else None
        maps: List[torch.Tensor] = []
        if tokens is not None:
            # Try to reshape tokens (excluding class token if present) into spatial map
            b, n, c = tokens.shape
            # assume square grid size from processor (patch embeddings)
            # simple heuristic: try closest square for (n or n-1)
            def to_hw(num: int) -> Optional[Tuple[int,int]]:
                r = int(num ** 0.5)
                for k in range(max(1, r-2), r+3):
                    if k * k == num:
                        return k, k
                return None
            # drop cls token if shapes mismatch
            grid = to_hw(n)
            tok = tokens
            if grid is None and n > 1:
                grid = to_hw(n - 1)
                if grid is not None:
                    tok = tokens[:, 1:, :]
            if grid is not None:
                hh, ww = grid
                fmap = tok.transpose(1, 2).reshape(b, c, hh, ww)
                maps.append(fmap)
        return BackboneOutputs(tokens=tokens, maps=maps)


def build_backbone(cfg: Dict[str, Any]) -> ViTBackbone:
    bcfg = cfg.get("model", {}).get("backbone", {})
    model_id = bcfg.get("hf_model_id", "hustvl/yolos-tiny")
    return ViTBackbone(model_id=model_id)
