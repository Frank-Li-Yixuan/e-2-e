from typing import Any, Dict, Optional

import math
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
            # optional grad clip
            clip_val = float(self.cfg.get("grad", {}).get("clip_norm", 0.0) or 0.0)
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=clip_val)
                if self.disc is not None:
                    torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=clip_val)
            opt_gen.step()
            self.log("train/l1", float(l1.detach().cpu()), prog_bar=True, on_step=True)
            self.log("train/perc", float(perc.detach().cpu()), prog_bar=False, on_step=True)
            for k,v in g_scalars.items():
                self.log(f"train/{k}", v, on_step=True)

        # return loss for logging purposes only
        return None
