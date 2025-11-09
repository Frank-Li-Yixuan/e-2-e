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
                import diffusers as _df
                print(f"[LoRA/Debug] diffusers.__version__={getattr(_df, '__version__', 'unknown')}")

                # Freeze base UNet and text encoder (we only train adapters)
                self.pipe.unet.requires_grad_(False)
                if hasattr(self.pipe, "text_encoder"):
                    try:
                        self.pipe.text_encoder.requires_grad_(False)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                # Normalize layer patterns from config (support aliases like up_blocks[-1])
                def _normalize_patterns(pats: List[str]) -> List[str]:
                    norm: List[str] = []
                    try:
                        last_up = len(self.pipe.unet.up_blocks) - 1  # type: ignore
                    except Exception:
                        last_up = 3
                    for s in pats:
                        s2 = s.replace("unet.", "")
                        s2 = s2.replace("[0]", ".0").replace("[1]", ".1").replace("[2]", ".2").replace("[3]", ".3")
                        s2 = s2.replace(".attentions[", ".attentions.")
                        if "up_blocks[-1]" in s2:
                            s2 = s2.replace("up_blocks[-1]", f"up_blocks.{last_up}")
                        norm.append(s2)
                    return norm

                lora_rank = int(self.finetune_cfg.get("lora_rank", 8))
                learnable_cfg = self.finetune_cfg.get("learnable_layers", ["unet.up_blocks[-1]"])
                patterns = _normalize_patterns(learnable_cfg)

                # Preferred modern path A: PEFT + pipeline-level adapters
                try:
                    from peft import LoraConfig as PeftLoraConfig  # type: ignore
                    adapter_name = str(self.finetune_cfg.get("adapter_name", "anon"))
                    peft_cfg = PeftLoraConfig(
                        r=lora_rank,
                        lora_alpha=lora_rank,
                        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                        bias="none",
                    )
                    # Freeze base UNet before injecting
                    for p in self.pipe.unet.parameters():
                        p.requires_grad_(False)
                    added = False
                    # Try pipeline-level first (diffusers>=0.31)
                    try:
                        add_ppl = getattr(self.pipe, "add_lora_adapters", None)
                        if callable(add_ppl):
                            add_ppl(adapter_name, peft_cfg)
                            added = True
                            # Activate this adapter (if API exists)
                            try:
                                set_adapters = getattr(self.pipe, "set_adapters", None)
                                if callable(set_adapters):
                                    set_adapters([adapter_name])
                            except Exception:
                                pass
                    except Exception as e_add:
                        added = False
                        last_err = e_add
                    # Fallback to UNet-level add_adapter if pipeline-level unavailable
                    if not added:
                        try:
                            add_unet = getattr(self.pipe.unet, "add_adapter", None)
                            if callable(add_unet):
                                add_unet(peft_cfg)
                                added = True
                        except Exception as e_add2:
                            added = False
                            last_err = e_add2
                    if not added:
                        raise RuntimeError(f"Failed to add PEFT LoRA adapters ({last_err})")

                    # Enable grads only for LoRA params in selected blocks
                    params: List[torch.nn.Parameter] = []
                    for n, p in self.pipe.unet.named_parameters():
                        is_lora = ("lora_" in n) or (".lora" in n) or ("lora_A" in n) or ("lora_B" in n)
                        if not is_lora:
                            p.requires_grad_(False)
                            continue
                        if patterns and not any(pat in n for pat in patterns):
                            p.requires_grad_(False)
                        else:
                            p.requires_grad_(True)
                            params.append(p)
                    self._trainable_params = params
                    print(f"[LoRA/PEFT] adapter={adapter_name} rank={lora_rank} selected_params={len(params)}")
                    return  # PEFT path succeeded; skip older paths
                except Exception as e_peft:
                    print(f"[LoRA] PEFT adapters path failed: {e_peft}")

                # Preferred modern path B: Diffusers LoRAConfig + add_attn_procs (no-PEFT)
                try:
                    from diffusers.models.lora import LoRAConfig  # type: ignore
                    unet = self.pipe.unet
                    lora_config = LoRAConfig(
                        r=lora_rank,
                        lora_alpha=lora_rank,
                        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                        init_lora_weights="gaussian",
                    )
                    unet.add_attn_procs(lora_config)
                    # Enable grads only for selected blocks
                    params: List[torch.nn.Parameter] = []
                    for name, module in unet.attn_processors.items():  # type: ignore[attr-defined]
                        scope_name = f"unet.{name}"
                        if patterns and not any(pat in scope_name for pat in patterns):
                            for p in module.parameters():
                                p.requires_grad_(False)
                            continue
                        for p in module.parameters():
                            p.requires_grad_(True)
                            params.append(p)
                    self._trainable_params = params
                    print(f"[LoRA] Applied LoRAConfig rank={lora_rank} selected_layers={len(params)}")
                except Exception as e_conf:
                    # Fallback: use AttnProcessor2_0 and selectively unfreeze parameters by name
                    print(f"[LoRA] LoRAConfig path failed ({e_conf}); falling back to selective AttnProcessor2_0.")
                    # Import AttnProcessor2_0 lazily with robust import paths
                    AttnProcessor2_0 = None
                    try:
                        from diffusers.models.attention_processor import AttnProcessor2_0  # type: ignore
                    except Exception:
                        try:
                            from diffusers import AttnProcessor2_0  # type: ignore
                        except Exception:
                            AttnProcessor2_0 = None  # type: ignore
                    if AttnProcessor2_0 is None:
                        raise RuntimeError("AttnProcessor2_0 not available in this diffusers version")
                    for p in self.pipe.unet.parameters():
                        p.requires_grad_(False)
                    self.pipe.unet.set_attn_processor(AttnProcessor2_0())
                    params: List[torch.nn.Parameter] = []
                    for n, p in self.pipe.unet.named_parameters():
                        if any(pat in n for pat in patterns):
                            p.requires_grad_(True)
                            params.append(p)
                    if len(params) == 0:
                        # last resort: allow last up_block attention weights
                        for n, p in self.pipe.unet.named_parameters():
                            if "up_blocks." in n and ("attn" in n or "attention" in n):
                                p.requires_grad_(True)
                                params.append(p)
                    self._trainable_params = params
                    print(f"[LoRA/Fallback] selected_params={len(params)}")
            except Exception as e:
                # As a last resort, fallback to enabling a small subset of UNet attention linear layers for finetuning
                print(f"[WARN] LoRA setup failed; attempting partial UNet finetune fallback: {e}")
                try:
                    patterns = [
                        ".attn1.to_q.weight", ".attn1.to_v.weight", ".attn1.to_out.0.weight",
                        ".attn2.to_q.weight", ".attn2.to_v.weight", ".attn2.to_out.0.weight",
                    ]
                    collected: List[torch.nn.Parameter] = []
                    for n, p in self.pipe.unet.named_parameters():
                        if any(pat in n for pat in patterns):
                            p.requires_grad_(True)
                            collected.append(p)
                    self._trainable_params = collected
                    try:
                        n_train = sum(int(p.numel()) for p in self._trainable_params)
                        print(f"[LoRA-Fallback] Enabled partial UNet finetune on attention linear layers; trainable params={n_train:,}")
                    except Exception:
                        pass
                except Exception as e2:
                    print(f"[WARN] Partial UNet finetune fallback also failed; proceeding without trainable generator params: {e2}")
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

    def generate(self, img: torch.Tensor, mask: torch.Tensor, prompt: str = "", negative_prompt: Optional[str] = None,
                 cond: Optional[Dict[str, Any]] = None, steps: Optional[int] = None, guidance_scale: Optional[float] = None,
                 seed: Optional[int] = None, strength: Optional[float] = None) -> torch.Tensor:
        self._maybe_init()
        assert self.pipe is not None
        # Decide whether to enable gradient path (for LoRA training) or use fast no-grad path
        grad_enabled = bool(self.finetune_cfg.get("use_lora", False)) and self.training

        # Prepare common kwargs
        # Trim prompts to avoid CLIP truncation warnings (77 token max incl. specials)
        prompt_use = self._trim_prompt(prompt)
        neg_use = self._trim_prompt(negative_prompt if negative_prompt else None)

        if grad_enabled:
            # Tensor path with output_type="pt" to keep computation graph for LoRA params
            # Convert images from [-1,1] to [0,1] as expected by diffusers when tensors are provided
            img01 = (img.clamp(-1, 1) * 0.5 + 0.5).to(self.device)
            mask01 = mask.clamp(0, 1).to(self.device)
            # SD Inpaint tensor path is most stable with mask as [B,1,H,W] float, where 1=inpaint, 0=keep
            if mask01.dim() == 3:
                mask02 = mask01.unsqueeze(1)
            elif mask01.dim() == 4 and mask01.size(1) in (1, 3):
                # If accidentally RGB mask, reduce to single channel
                mask02 = mask01[:, :1, :, :]
            else:
                # Coerce to [B,1,H,W]
                mask02 = mask01
                while mask02.dim() < 4:
                    mask02 = mask02.unsqueeze(1)
            # Optional deterministic seed
            gen = None
            if seed is not None:
                try:
                    gen = torch.Generator(device=self.device).manual_seed(int(seed))
                except Exception:
                    gen = torch.Generator().manual_seed(int(seed))
            # Disable classifier-free guidance during training to avoid batch-mismatch in tensor inpaint path
            # (diffusers duplicates mask/image latents internally when CFG is enabled, which can mismatch shapes at concat time)
            cfg_gs = float(guidance_scale) if guidance_scale is not None else self.guidance_scale
            if cfg_gs is None:
                cfg_gs = 7.5
            train_guidance = 1.0  # force off CFG for stable training grads
            kwargs = dict(
                prompt=prompt_use,
                negative_prompt=None,  # disable negative prompts in training to fully avoid CFG branching
                image=img01,
                mask_image=mask02,
                num_inference_steps=int(steps) if steps is not None else self.steps,
                guidance_scale=train_guidance,
                generator=gen,
                output_type="pt",
                return_dict=True,
                num_images_per_prompt=1,
            )
            if strength is not None:
                try:
                    kwargs["strength"] = float(strength)
                except Exception:
                    pass
            # Some pipelines expect 8-bit mask semantics (1=inpaint). Ensure float32 for stability.
            kwargs["mask_image"] = kwargs["mask_image"].to(dtype=torch.float32)
            out = self.pipe(**kwargs).images  # torch.Tensor [B, C, H, W] in [0,1]
            out = out.to(img.device)
            out_t = out * 2 - 1  # to [-1,1]
            # compose with original to ensure background invariance
            composed = img * (1 - mask) + out_t * mask
            return composed
        else:
            # Fast path with PIL + no grad for inference/validation
            b = img.size(0)
            outs: List[torch.Tensor] = []
            with torch.no_grad():
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
                    if strength is not None:
                        try:
                            kwargs["strength"] = float(strength)
                        except Exception:
                            pass
                    out = self.pipe(**kwargs).images[0]
                    out_t = torch.from_numpy(np.array(out)).permute(2, 0, 1).float() / 255.0
                    out_t = out_t * 2 - 1  # to [-1,1]
                    out_t = out_t.to(img.device)
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

    def ensure_initialized(self) -> None:
        """Ensure lazy backends (diffusers) are initialized before optimizer setup.
        This helps make sure LoRA or attention-processor params are created in time.
        """
        if self.backend == "diffusers":
            try:
                # type: ignore[attr-defined]
                self.net._maybe_init()
            except Exception:
                pass

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
