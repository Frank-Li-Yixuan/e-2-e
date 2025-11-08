from typing import Any, Dict, Optional, List

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


def denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W] in [-1,1] -> return HxWx3 uint8
    y = (x.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    y = y.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    return y


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class SimpleUNetInpaint(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 64, use_skip: bool = True) -> None:
        super().__init__()
        self.use_skip = use_skip
        c = base
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, padding=3), nn.LeakyReLU(0.2, inplace=True), ResBlock(c)
        )
        self.enc2 = nn.Sequential(nn.Conv2d(c, c * 2, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(c * 2))
        self.enc3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(c * 4))

        self.mid = nn.Sequential(ResBlock(c * 4), ResBlock(c * 4))

        self.dec3 = nn.Sequential(nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(c * 2))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(c))
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Inputs are normalized [-1,1]; concatenated with mask
        x = torch.cat([img, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.mid(e3)
        d3 = self.dec3(m)
        if self.use_skip:
            d3 = d3 + e2
        d2 = self.dec2(d3)
        if self.use_skip:
            d2 = d2 + e1
        out = torch.tanh(self.out(d2))
        # compose with original using mask: fill masked (1) with prediction, keep unmasked
        return img * (1 - mask) + out * mask


class DiffusersInpaintBackend(nn.Module):
    def __init__(self, model_id: str, device: torch.device, enable_xformers: bool = False, torch_compile: bool = False, finetune_cfg: Optional[Dict[str, Any]] = None, steps: int = 20, guidance_scale: float = 7.5, prompt_token_limit: int = 75) -> None:
        super().__init__()
        self.device = device
        self.model_id = model_id
        self.enable_xformers = enable_xformers
        self.torch_compile = torch_compile
        self.finetune_cfg = finetune_cfg or {"use_lora": False}
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.prompt_token_limit = int(prompt_token_limit)
        self.pipe = None  # lazy

    def _maybe_init(self) -> None:
        if self.pipe is not None:
            return
        from diffusers import StableDiffusionInpaintPipeline  # type: ignore
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(self.model_id)
        # Reduce VRAM footprint on 4GB GPUs
        try:
            self.pipe.enable_attention_slicing("max")
        except Exception:
            pass
        if self.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        try:
            # use fp16 on CUDA if available
            if self.device.type == 'cuda':
                self.pipe = self.pipe.to(self.device, torch.float16)
            else:
                self.pipe = self.pipe.to(self.device)
        except Exception:
            self.pipe = self.pipe.to(self.device)
        # optional LoRA lightweight finetune
        self._trainable_params = []
        if self.finetune_cfg.get("use_lora", False):
            try:
                # Freeze everything by default (we'll only train LoRA)
                self.pipe.unet.requires_grad_(False)
                if hasattr(self.pipe, "text_encoder"):
                    try:
                        self.pipe.text_encoder.requires_grad_(False)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                # Build per-attention processor LoRA modules
                from diffusers.models.attention_processor import LoRAAttnProcessor  # type: ignore
                from diffusers.training_utils import AttnProcsLayers  # type: ignore

                r = int(self.finetune_cfg.get("lora_rank", 8))
                learnable: List[str] = list(self.finetune_cfg.get("learnable_layers", []))

                # Map each attention processor name to an appropriate LoRA processor
                lora_attn_procs = {}
                for name in list(self.pipe.unet.attn_processors.keys()):
                    # name examples:
                    #  'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'
                    #  'mid_block.attentions.0.transformer_blocks.0.attn2.processor'
                    unet_scoped_name = f"unet.{name}"
                    if len(learnable) > 0 and not any(patt in unet_scoped_name for patt in learnable):
                        # skip layers not selected by user patterns
                        continue

                    # Determine cross_attention_dim and hidden_size per block
                    cross_attention_dim = None if name.endswith("attn1.processor") else self.pipe.unet.config.cross_attention_dim
                    if name.startswith("mid_block"):
                        hidden_size = self.pipe.unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks."):].split(".")[0])
                        hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks."):].split(".")[0])
                        hidden_size = self.pipe.unet.config.block_out_channels[block_id]
                    else:
                        # default fallback
                        hidden_size = self.pipe.unet.config.block_out_channels[0]

                    lora_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=r,
                    )

                if len(lora_attn_procs) == 0:
                    # If user patterns filter everything out, fallback to enabling all
                    for name in list(self.pipe.unet.attn_processors.keys()):
                        cross_attention_dim = None if name.endswith("attn1.processor") else self.pipe.unet.config.cross_attention_dim
                        if name.startswith("mid_block"):
                            hidden_size = self.pipe.unet.config.block_out_channels[-1]
                        elif name.startswith("up_blocks"):
                            block_id = int(name[len("up_blocks."):].split(".")[0])
                            hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
                        elif name.startswith("down_blocks"):
                            block_id = int(name[len("down_blocks."):].split(".")[0])
                            hidden_size = self.pipe.unet.config.block_out_channels[block_id]
                        else:
                            hidden_size = self.pipe.unet.config.block_out_channels[0]
                        lora_attn_procs[name] = LoRAAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            rank=r,
                        )

                # Set LoRA processors into the UNet
                self.pipe.unet.set_attn_processor(lora_attn_procs)

                # Collect LoRA parameters only
                lora_layers = AttnProcsLayers(self.pipe.unet.attn_processors)
                # Ensure all LoRA params are trainable
                for p in lora_layers.parameters():
                    p.requires_grad_(True)
                # Move LoRA layers to the same device/dtype as UNet
                try:
                    dtype_to_use = torch.float16 if (self.device.type == 'cuda') else torch.float32
                    lora_layers.to(self.device, dtype=dtype_to_use)
                except Exception:
                    try:
                        lora_layers.to(self.device)
                    except Exception:
                        pass
                self._trainable_params = list(lora_layers.parameters())

                # Optional info
                try:
                    n_train = sum(int(p.numel()) for p in self._trainable_params)
                    print(f"[LoRA] Enabled LoRA on UNet with rank={r}, trainable params={n_train:,}")
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] LoRA setup failed; continuing without LoRA: {e}")
                self._trainable_params = []

    def _trim_prompt(self, text: Optional[str]) -> Optional[str]:
        if text is None or text == "":
            return text
        try:
            tok = getattr(self.pipe, "tokenizer", None)  # type: ignore[attr-defined]
            if tok is None:
                return text
            # Encode without truncation to measure
            ids = tok(text, add_special_tokens=True, return_attention_mask=False, return_tensors=None).get("input_ids", [])
            # Handle both list[int] and list[list[int]] (batch) returns
            if isinstance(ids, list) and len(ids) > 0:
                if isinstance(ids[0], list):  # batched
                    flat_ids = ids[0]
                else:
                    flat_ids = ids
                tok_count = len(flat_ids)
            else:
                flat_ids = []
                tok_count = 0
            # Reserve space for special tokens if tokenizer adds them internally (CLIP max 77)
            effective_limit = max(4, min(self.prompt_token_limit, 77))
            if tok_count <= self.prompt_token_limit:
                return text
            # Heuristic: try trimming by comma-separated segments first, then by words
            parts = [p.strip() for p in text.split(",") if p.strip()]
            while len(parts) > 1:
                test_text = ", ".join(parts)
                test_ids = tok(test_text, add_special_tokens=True, return_attention_mask=False, return_tensors=None).get("input_ids", [])
                if isinstance(test_ids, list) and len(test_ids) > 0 and isinstance(test_ids[0], list):
                    test_ids = test_ids[0]
                if len(test_ids) <= effective_limit:
                    return test_text
                # drop the last segment
                parts = parts[:-1]
            # Fallback to word-wise trim
            words = text.split()
            lo, hi = 1, len(words)
            best = words
            while lo <= hi:
                mid = (lo + hi) // 2
                test = " ".join(words[:mid])
                test_ids = tok(test, add_special_tokens=True, return_attention_mask=False, return_tensors=None).get("input_ids", [])
                if isinstance(test_ids, list) and len(test_ids) > 0 and isinstance(test_ids[0], list):
                    test_ids = test_ids[0]
                if len(test_ids) <= effective_limit:
                    best = words[:mid]
                    lo = mid + 1
                else:
                    hi = mid - 1
            trimmed = " ".join(best)
            # Strict final enforcement: if still above limit, slice token IDs and decode
            final_ids = tok(trimmed, add_special_tokens=True, return_attention_mask=False, return_tensors=None).get("input_ids", [])
            if isinstance(final_ids, list) and len(final_ids) > 0 and isinstance(final_ids[0], list):
                final_ids = final_ids[0]
            if len(final_ids) > effective_limit:
                # Keep first effective_limit tokens; attempt decode
                sub_ids = final_ids[:effective_limit]
                try:
                    # Some tokenizers expose decode; if not, fallback to joining tokens
                    trimmed2 = tok.decode(sub_ids, skip_special_tokens=True)
                    if trimmed2.strip():
                        return trimmed2.strip()
                except Exception:
                    pass
            return trimmed.strip()
        except Exception:
            return text

    @torch.no_grad()
    def generate(self, img: torch.Tensor, mask: torch.Tensor, prompt: str = "", negative_prompt: Optional[str] = None,
                 cond: Optional[Dict[str, Any]] = None, steps: Optional[int] = None, guidance_scale: Optional[float] = None,
                 seed: Optional[int] = None, strength: Optional[float] = None) -> torch.Tensor:
        self._maybe_init()
        assert self.pipe is not None
        # Convert to PIL HWC
        b = img.size(0)
        outs: List[torch.Tensor] = []
        for i in range(b):
            pil_img = Image.fromarray(denorm_to_uint8(img[i]))
            pil_mask = Image.fromarray((mask[i].squeeze(0).clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8))
            # Optional deterministic seed
            gen = None
            if seed is not None:
                try:
                    gen = torch.Generator(device=self.device).manual_seed(int(seed))
                except Exception:
                    gen = torch.Generator().manual_seed(int(seed))
            # Trim prompts to avoid CLIP truncation warnings (77 token max incl. specials)
            prompt_use = self._trim_prompt(prompt)
            neg_use = self._trim_prompt(negative_prompt if negative_prompt else None)
            kwargs = dict(
                prompt=prompt_use,
                negative_prompt=neg_use if neg_use else None,
                image=pil_img,
                mask_image=pil_mask,
                height=pil_img.height,
                width=pil_img.width,
                num_inference_steps=int(steps) if steps is not None else self.steps,
                guidance_scale=float(guidance_scale) if guidance_scale is not None else self.guidance_scale,
                generator=gen,
            )
            # Some inpaint pipelines support strength; pass if provided
            if strength is not None:
                try:
                    kwargs["strength"] = float(strength)
                except Exception:
                    pass
            out = self.pipe(**kwargs).images[0]
            out_t = torch.from_numpy(np.array(out)).permute(2, 0, 1).float() / 255.0
            out_t = out_t * 2 - 1  # to [-1,1]
            out_t = out_t.to(img.device)
            # compose with original to ensure background invariance
            composed = img[i] * (1 - mask[i]) + out_t * mask[i]
            outs.append(composed)
        return torch.stack(outs, dim=0)


class GeneratorWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        gcfg = cfg.get("model", {}).get("generator", {})
        self.backend = gcfg.get("backend", "unet")
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
        if self.backend == "unet":
            self.net = SimpleUNetInpaint(
                in_channels=int(gcfg.get("in_channels", 4)),
                base=int(gcfg.get("base_channels", 64)),
                use_skip=bool(gcfg.get("use_skip", True)),
            ).to(self.device)
        elif self.backend == "diffusers":
            dcfg = gcfg.get("diffusers", {})
            self.net = DiffusersInpaintBackend(
                model_id=dcfg.get("model_id", "runwayml/stable-diffusion-inpainting"),
                device=self.device,
                enable_xformers=bool(dcfg.get("enable_xformers", False)),
                torch_compile=bool(dcfg.get("torch_compile", False)),
                finetune_cfg=gcfg.get("finetune", {}),
                steps=int(dcfg.get("steps", 20)),
                guidance_scale=float(dcfg.get("guidance_scale", 7.5)),
                prompt_token_limit=int(dcfg.get("prompt_token_limit", 75)),
            )
            # Eagerly initialize Diffusers pipeline so LoRA params are materialized
            # before configure_optimizers collects trainable parameters.
            try:
                self.net._maybe_init()  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown generator backend: {self.backend}")

    def forward(self, img: torch.Tensor, mask: torch.Tensor, prompt: str = "", negative_prompt: Optional[str] = None,
                cond: Optional[Dict[str, Any]] = None, steps: Optional[int] = None, guidance_scale: Optional[float] = None,
                seed: Optional[int] = None, strength: Optional[float] = None) -> torch.Tensor:
        if self.backend == "unet":
            return self.net(img.to(self.device), mask.to(self.device))
        else:
            return self.net.generate(
                img.to(self.device),
                mask.to(self.device),
                prompt=prompt,
                negative_prompt=negative_prompt,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                strength=strength,
            )

    def trainable_parameters(self):
        if self.backend == "unet":
            return self.net.parameters()
        else:
            # Diffusers backend: return trainable parameters if LoRA/attn procs enabled
            return getattr(self.net, "_trainable_params", [])
