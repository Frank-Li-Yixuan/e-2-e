from typing import Any, Dict, Optional, List

import math
import os
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .detector_wrapper import DetectorWrapper
from .generator_wrapper import GeneratorWrapper
from .discriminator import build_discriminator
from .losses import PerceptualLoss, adversarial_d_loss, adversarial_g_loss, arcface_similarity


def _get_train_prompts(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    gcfg = cfg.get("model", {}).get("generator", {})
    base = gcfg.get("train_prompt", None)
    neg = gcfg.get("negative_prompt", None)
    face = gcfg.get("train_prompt_face", None)
    plate = gcfg.get("train_prompt_plate", None)
    return {"base": base, "neg": neg, "face": face, "plate": plate}


def _compose_dynamic_prompt(base: Optional[str], has_face: bool, has_plate: bool,
                            face_prompt: Optional[str], plate_prompt: Optional[str]) -> str:
    parts = []
    if base:
        parts.append(base)
    if has_face and face_prompt:
        parts.append(face_prompt)
    if has_plate and plate_prompt:
        parts.append(plate_prompt)
    text = ", ".join([p.strip() for p in parts if p and p.strip()])
    return text[:512]


class AnonyLightningModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        # models
        self.detector = DetectorWrapper(cfg)
        self.generator = GeneratorWrapper(cfg)
        self.disc = build_discriminator(cfg) if cfg.get("model", {}).get("discriminator", {}).get("enabled", True) else None
        # losses
        self.perc_loss = PerceptualLoss(device=self.device)
        self.lcfg = cfg.get("loss", {})
        # Use manual optimization to alternate optimizers
        self.automatic_optimization = False
        # Validation buffers
        self.val_orig_buf: List[torch.Tensor] = []
        self.val_anon_buf: List[torch.Tensor] = []
        self.val_max = int(min(32, int(cfg.get("eval", {}).get("max_images", 50) or 50)))
        self.privacy_ok_thr = float(cfg.get("eval", {}).get("privacy_ok_threshold", 0.28))

        # LoRA / trainable params statistics
        try:
            trainable = list(self.generator.trainable_parameters())
            trainable_count = sum(int(p.numel()) for p in trainable)
            total_count = 0
            if getattr(self.generator, "backend", "unet") == "unet":
                total_count = sum(int(p.numel()) for p in self.generator.parameters())
            else:
                net = getattr(self.generator, "net", None)
                pipe = getattr(net, "pipe", None)
                unet = getattr(pipe, "unet", None) if pipe is not None else None
                if unet is not None:
                    total_count = sum(int(p.numel()) for p in unet.parameters())
                else:
                    total_count = trainable_count
            frozen = max(0, total_count - trainable_count)
            print(f"[LoRA/Trainable] trainable={trainable_count:,} frozen={frozen:,} total~={total_count:,}")
        except Exception as e:
            print(f"[WARN] Failed to compute trainable/frozen params: {e}")

    def forward(self, images: torch.Tensor, masks: torch.Tensor, prompt: Optional[str] = None, neg: Optional[str] = None) -> torch.Tensor:
        return self.generator(images, masks, prompt=prompt or "", negative_prompt=neg)

    def configure_optimizers(self):
        train_cfg = self.cfg.get("train", {})
        lr = train_cfg.get("lr", {"detector": 1e-5, "generator": 1e-4, "discriminator": 1e-4})
        opt_det = torch.optim.AdamW(list(self.detector.parameters()), lr=float(lr.get("detector", 1e-5))) if len(list(self.detector.parameters()))>0 else None
        gen_params = list(self.generator.trainable_parameters())
        opt_gen = torch.optim.AdamW(gen_params, lr=float(lr.get("generator", 1e-4))) if len(gen_params)>0 else None
        opt_disc = torch.optim.AdamW(self.disc.parameters(), lr=float(lr.get("discriminator", 1e-4))) if self.disc is not None else None
        opts = []
        if opt_det is not None:
            opts.append(opt_det)
        if opt_gen is not None:
            opts.append(opt_gen)
        if opt_disc is not None:
            opts.append(opt_disc)
        return opts

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        # optimizers order: [det?, gen?, disc?] as configured above
        opts = self.optimizers()
        # Lightning returns single optimizer or list
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        opt_det = None
        opt_gen = None
        opt_disc = None
        # Map by param group lengths heuristics
        for o in opts:
            # crude mapping by param names
            names = [n for n,_ in self.detector.named_parameters()] if self.detector is not None else []
            if any(p is o.param_groups[0]['params'][0] for p in self.detector.parameters()) if self.detector is not None and len(list(self.detector.parameters()))>0 else False:
                opt_det = o
        for o in opts:
            if len(list(self.generator.trainable_parameters()))>0 and any(p is o.param_groups[0]['params'][0] for p in self.generator.trainable_parameters()):
                opt_gen = o
        for o in opts:
            if self.disc is not None and any(p is o.param_groups[0]['params'][0] for p in self.disc.parameters()):
                opt_disc = o
        # fallback by count order
        if opt_gen is None and len(opts) >= 1:
            opt_gen = opts[0]
        if opt_disc is None and len(opts) >= 2:
            opt_disc = opts[-1]

        images = batch["images"].to(self.device)
        masks = batch["masks"].to(self.device)

        # 1) detector update (if GT exists)
        det_loss_val = None
        if opt_det is not None and any(b.numel() > 0 for b in batch["boxes"]):
            self.detector.train()
            opt_det.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=(torch.cuda.is_available() and self.trainer is not None and str(self.trainer.precision).startswith('16'))):
                det_loss, det_scalars = self.detector.compute_loss(batch)
            self.manual_backward(det_loss)
            opt_det.step()
            det_loss_val = float(det_loss.detach().cpu())
            self.log("train/det_loss", det_loss_val, prog_bar=False, on_step=True)

        # 2) discriminator updates
        if self.disc is not None and opt_disc is not None:
            self.disc.train()
            with torch.no_grad():
                fake = self.generator(images, masks)
            opt_disc.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=(torch.cuda.is_available() and self.trainer is not None and str(self.trainer.precision).startswith('16'))):
                pred_real = self.disc(images)
                pred_fake = self.disc(fake.detach())
                d_loss = adversarial_d_loss(pred_real, pred_fake)
            self.manual_backward(d_loss)
            opt_disc.step()
            self.log("train/d_loss", float(d_loss.detach().cpu()), prog_bar=False, on_step=True)

        # 3) generator update
        g_scalars: Dict[str, float] = {}
        if opt_gen is not None:
            prm = _get_train_prompts(self.cfg)
            has_face = any((lbl.numel() > 0 and (lbl == 1).any().item()) for lbl in batch["labels"])  # cat 1 => face
            has_plate = any((lbl.numel() > 0 and (lbl == 2).any().item()) for lbl in batch["labels"])  # cat 2 => plate
            dyn_prompt = _compose_dynamic_prompt(
                prm.get("base"), has_face, has_plate,
                prm.get("face", "realistic anonymized face"),
                prm.get("plate", "randomized license plate characters, realistic metal plate")
            )
            neg_prompt = prm.get("neg", "low quality, artifacts, distorted, blank plate, no characters")
            # International plate prompt enhancement
            if has_plate:
                style_key, style_prompt, style_neg = self._select_plate_style(batch)
                if style_prompt:
                    dyn_prompt = (dyn_prompt + ", " + style_prompt).strip(", ")
                if style_neg:
                    neg_prompt = (neg_prompt + ", " + style_neg).strip(", ")

            opt_gen.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=(torch.cuda.is_available() and self.trainer is not None and str(self.trainer.precision).startswith('16'))):
                fake = self.generator(images, masks, prompt=dyn_prompt, negative_prompt=neg_prompt)
                l1 = torch.tensor(0.0, device=self.device)
                perc = torch.tensor(0.0, device=self.device)
                with_pseudo = [i for i, p in enumerate(batch["pseudos"]) if p is not None]
                if len(with_pseudo) > 0:
                    idx = torch.tensor(with_pseudo, dtype=torch.long, device=self.device)
                    pseudo = torch.stack([p.to(self.device) for p in batch["pseudos"] if p is not None], dim=0)
                    fake_sel = fake[idx]
                    l1 = F.l1_loss(fake_sel, pseudo)
                    perc = self.perc_loss(fake_sel, pseudo)
                loss = self.lcfg.get("l1_weight", 1.0) * l1 + self.lcfg.get("perceptual_weight", 0.1) * perc
                if self.disc is not None:
                    pred_fake = self.disc(fake)
                    g_adv = adversarial_g_loss(pred_fake)
                    loss = loss + self.lcfg.get("adv_weight", 0.1) * g_adv
                    g_scalars["g_adv"] = float(g_adv.detach().cpu())
                if self.lcfg.get("id_suppress_weight", 0.0) > 0:
                    sim = arcface_similarity(images, fake)
                    if sim is not None:
                        id_loss = sim.mean()
                        loss = loss + self.lcfg["id_suppress_weight"] * id_loss
                        g_scalars["id_sim"] = float(sim.mean().detach().cpu())
                if len(with_pseudo) == 0:
                    recon = F.l1_loss(fake, images)
                    loss = loss + 0.01 * recon
            self.manual_backward(loss)
            # optional grad clip (clip only trainable generator params to avoid empty-generator warning)
            clip_val = float(self.cfg.get("grad", {}).get("clip_norm", 0.0) or 0.0)
            if clip_val > 0:
                try:
                    gen_params_list = list(self.generator.trainable_parameters())
                    if len(gen_params_list) == 0:
                        gen_params_list = list(self.generator.parameters())
                    if len(gen_params_list) > 0:
                        torch.nn.utils.clip_grad_norm_(gen_params_list, max_norm=clip_val)
                except Exception:
                    pass
                if self.disc is not None:
                    try:
                        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=clip_val)
                    except Exception:
                        pass
            opt_gen.step()
            self.log("train/l1", float(l1.detach().cpu()), prog_bar=True, on_step=True)
            self.log("train/perc", float(perc.detach().cpu()), prog_bar=False, on_step=True)
            for k,v in g_scalars.items():
                self.log(f"train/{k}", v, on_step=True)

        # Simple checkpoint per steps
        try:
            ck = int(self.cfg.get("checkpoint", {}).get("save_every_steps", 0) or 0)
            out_dir = self.cfg.get("paths", {}).get("outputs", "outputs")
            if ck > 0 and (int(self.global_step) % ck == 0) and int(self.global_step) > 0:
                ck_dir = os.path.join(out_dir, "lightning_ckpts")
                os.makedirs(ck_dir, exist_ok=True)
                torch.save({"generator": self.generator.state_dict()}, os.path.join(ck_dir, f"gen_step{int(self.global_step):06d}.pt"))
        except Exception as e:
            print(f"[WARN] checkpoint save failed: {e}")

        # return loss for logging purposes only
        return None

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if len(self.val_orig_buf) >= self.val_max:
            return None
        images = batch["images"].to(self.device)
        masks = batch["masks"].to(self.device)
        # Simple prompt (reuse train logic partially)
        prm = _get_train_prompts(self.cfg)
        has_face = any((lbl.numel() > 0 and (lbl == 1).any().item()) for lbl in batch["labels"])  # cat 1 => face
        has_plate = any((lbl.numel() > 0 and (lbl == 2).any().item()) for lbl in batch["labels"])  # cat 2 => plate
        dyn_prompt = _compose_dynamic_prompt(
            prm.get("base"), has_face, has_plate,
            prm.get("face", "realistic anonymized face"),
            prm.get("plate", "randomized license plate characters, realistic metal plate")
        )
        neg_prompt = prm.get("neg", "low quality, artifacts, distorted, blank plate, no characters")
        if has_plate:
            style_key, style_prompt, style_neg = self._select_plate_style(batch)
            if style_prompt:
                dyn_prompt = (dyn_prompt + ", " + style_prompt).strip(", ")
            if style_neg:
                neg_prompt = (neg_prompt + ", " + style_neg).strip(", ")

        anon = self.generator(images, masks, prompt=dyn_prompt, negative_prompt=neg_prompt)
        remain = self.val_max - len(self.val_orig_buf)
        take = min(remain, images.size(0))
        self.val_orig_buf.append(images[:take].detach().cpu())
        self.val_anon_buf.append(anon[:take].detach().cpu())

        # Optionally save samples
        try:
            out_dir = self.cfg.get("paths", {}).get("outputs", "outputs")
            e_dir = os.path.join(out_dir, "lightning_val_samples", f"epoch_{int(self.current_epoch)}")
            os.makedirs(e_dir, exist_ok=True)
            for i in range(take):
                o = self._to_uint8(images[i])
                a = self._to_uint8(anon[i])
                Image.fromarray(o).save(os.path.join(e_dir, f"orig_{batch_idx:04d}_{i:02d}.jpg"))
                Image.fromarray(a).save(os.path.join(e_dir, f"anon_{batch_idx:04d}_{i:02d}.jpg"))
        except Exception:
            pass
        return None

    def on_validation_epoch_end(self) -> None:
        if len(self.val_orig_buf) == 0 or len(self.val_anon_buf) == 0:
            return
        try:
            orig = torch.cat(self.val_orig_buf, dim=0)
            anon = torch.cat(self.val_anon_buf, dim=0)
            sim = arcface_similarity(orig, anon)
            mean_sim = float(sim.mean().item()) if sim is not None else float('nan')
            self.log("val/arcface_mean_sim", mean_sim, prog_bar=True, on_epoch=True)
            status = "[PRIVACY_OK]" if (sim is not None and mean_sim < self.privacy_ok_thr) else "[PRIVACY_WARN]"
            print(f"{status} epoch={int(self.current_epoch)} arcface_mean={mean_sim:.4f} thr={self.privacy_ok_thr}")
        except Exception as e:
            print(f"[WARN] validation metrics failed: {e}")
        finally:
            self.val_orig_buf.clear()
            self.val_anon_buf.clear()

    def _to_uint8(self, x: torch.Tensor) -> "np.ndarray":
        try:
            import numpy as np
            y = (x.detach().cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0
            return y.permute(1, 2, 0).numpy().astype(np.uint8)
        except Exception:
            return (x.detach().cpu().clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().permute(1,2,0).numpy()

    def _select_plate_style(self, batch: Dict[str, Any]):
        styles = self.cfg.get("model", {}).get("generator", {}).get("diffusers", {}).get("plate_styles", {})
        whitelist = self.cfg.get("model", {}).get("generator", {}).get("diffusers", {}).get("plate_style_whitelist", None)
        # Default fallbacks
        chosen = "CN"
        prompt_hint = None
        neg_hint = None
        try:
            # Compute average aspect ratio among label=2 boxes
            ratios: List[float] = []
            for b, lbl in zip(batch["boxes"], batch["labels"]):
                if lbl.numel() == 0:
                    continue
                # select those equal to 2
                if (lbl == 2).any().item():
                    for bb in b[(lbl == 2).nonzero(as_tuple=False).view(-1)]:
                        x1, y1, x2, y2 = bb.tolist()
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        ratios.append(float(w / h))
            r = sum(ratios) / len(ratios) if len(ratios) > 0 else None
            # Simple heuristic thresholds
            if r is not None:
                # EU plates are very wide (e.g., ~4.7), CN ~3.1, US often ~2.0
                if r >= 4.2 and (whitelist is None or "EU" in whitelist):
                    chosen = "EU"
                elif r >= 2.6 and r < 4.2 and (whitelist is None or "CN" in whitelist):
                    chosen = "CN"
                elif (whitelist is None or "US" in whitelist):
                    chosen = "US"
            # Extract hints
            if isinstance(styles, dict) and chosen in styles:
                st = styles[chosen]
                prompt_hint = st.get("prompt_hint")
                neg_hint = st.get("neg_hint")
        except Exception:
            pass
        return chosen, prompt_hint, neg_hint
