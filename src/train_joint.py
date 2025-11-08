import argparse
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional
import sys
import multiprocessing as mp
import random
import numpy as np

import yaml
import csv
import math
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .datasets import build_dataloaders
from .detector_wrapper import DetectorWrapper
from .generator_wrapper import GeneratorWrapper
from .discriminator import build_discriminator
from .losses import PerceptualLoss, adversarial_d_loss, adversarial_g_loss, arcface_similarity, get_device
from . import eval_utils

# Ensure child Python processes (e.g., DataLoader workers) use the same interpreter as the parent on Windows
try:
    mp.set_executable(sys.executable)
    # Also configure torch.multiprocessing if available
    try:
        import torch.multiprocessing as tmp  # type: ignore
        tmp.set_executable(sys.executable)
    except Exception:
        pass
    try:
        mp.set_start_method('spawn', force=True)
        try:
            import torch.multiprocessing as tmp  # type: ignore
            tmp.set_start_method('spawn', force=True)
        except Exception:
            pass
    except Exception:
        pass
except Exception:
    pass


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


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


def train(cfg: Dict[str, Any], mode: str = "auto", max_steps_override: Optional[int] = None) -> None:
    # Reproducibility controls
    try:
        seed = int(cfg.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        repro = cfg.get("repro", {})
        if bool(repro.get("deterministic", False)):
            # Turn on deterministic algorithms; note this can reduce throughput
            try:
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                # For closer numerical match, disable TF32 under deterministic mode
                torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
                torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    device = get_device(cfg)
    # Enable cuDNN benchmark for potential speedup with fixed image sizes
    try:
        if torch.cuda.is_available():
            # Respect deterministic preference; otherwise enable benchmark
            if not bool(cfg.get("repro", {}).get("deterministic", False)):
                torch.backends.cudnn.benchmark = True
            # Enable TF32 for Ampere GPUs (e.g., RTX 30 series) to speed up matmul/convs
            try:
                if not bool(cfg.get("repro", {}).get("deterministic", False)):
                    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
            # Slightly favor higher matmul precision which can benefit some convs/BN fusions
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass
    train_loader, val_loader = build_dataloaders(cfg)
    assert train_loader is not None, "No training dataloader built. Check paths in config."

    detector = DetectorWrapper(cfg)
    generator = GeneratorWrapper(cfg)
    # Optional resume of generator weights
    ckpt_resume = cfg.get("checkpoint", {}).get("resume")
    if ckpt_resume and os.path.exists(ckpt_resume):
        try:
            state = torch.load(ckpt_resume, map_location="cpu")
            g_state = state.get("generator", state)
            missing, unexpected = generator.load_state_dict(g_state, strict=False)
            print(f"[INFO] Loaded generator from {ckpt_resume}; missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[WARN] Failed to load generator checkpoint from {ckpt_resume}: {e}")
    disc = build_discriminator(cfg) if cfg.get("model", {}).get("discriminator", {}).get("enabled", True) else None
    if disc is not None:
        disc = disc.to(device)
    # Optional torch.compile for speed
    try:
        compile_gen = bool(cfg.get("model", {}).get("generator", {}).get("torch_compile", False))
        compile_disc = bool(cfg.get("model", {}).get("discriminator", {}).get("torch_compile", False))
        # Require Triton for inductor backend; if missing, skip compile to avoid runtime failure
        triton_ok = False
        try:
            import triton  # type: ignore
            triton_ok = True
        except Exception:
            triton_ok = False
        if not triton_ok and (compile_gen or compile_disc):
            print("[WARN] Triton not available; skipping torch.compile to avoid inductor failure")
            compile_gen = False
            compile_disc = False
        if compile_gen and hasattr(torch, "compile"):
            generator = torch.compile(generator, mode="max-autotune")  # type: ignore
            print("[INFO] torch.compile enabled for generator")
        if compile_disc and hasattr(torch, "compile") and disc is not None:
            disc = torch.compile(disc, mode="max-autotune")  # type: ignore
            print("[INFO] torch.compile enabled for discriminator")
    except Exception as e:
        print(f"[WARN] torch.compile failed or unavailable: {e}")

    # losses
    perc_loss = PerceptualLoss(device=device)

    # optimizers
    det_params = list(detector.parameters())
    opt_det = optim.AdamW(det_params, lr=float(cfg["train"]["lr"]["detector"])) if len(det_params) > 0 else None
    opt_gen = optim.AdamW(generator.trainable_parameters(), lr=float(cfg["train"]["lr"]["generator"])) if list(generator.trainable_parameters()) else None
    opt_disc = optim.AdamW(disc.parameters(), lr=float(cfg["train"]["lr"]["discriminator"])) if disc is not None else None

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.get("precision", "fp32") != "fp32" and torch.cuda.is_available()))

    lcfg = cfg.get("loss", {})

    # --- Schedulers: linear warmup + cosine decay ---
    def make_scheduler(opt: Optional[optim.Optimizer], key: str):
        if opt is None:
            return None
        scfg = cfg.get("schedule", {}).get(key, {})
        warmup = int(scfg.get("warmup_steps", 0))
        total = int(scfg.get("total_steps", cfg.get("train", {}).get("max_steps", 1000) or 1000))
        min_scale = float(scfg.get("min_lr_scale", 0.1))
        def lr_lambda(step: int):
            if total <= 0:
                return 1.0
            if warmup > 0 and step < warmup:
                return max(1e-6, (step + 1) / float(warmup))
            # cosine decay after warmup
            t = min(max(step - warmup, 0), max(total - warmup, 1))
            cos = 0.5 * (1 + math.cos(math.pi * t / max(total - warmup, 1)))
            return min_scale + (1 - min_scale) * cos
        try:
            return optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        except Exception:
            return None
    sch_gen = make_scheduler(opt_gen, "generator")
    sch_disc = make_scheduler(opt_disc, "discriminator")
    sch_det = make_scheduler(opt_det, "detector")

    # gradient clipping config
    clip_norm = float(cfg.get("grad", {}).get("clip_norm", 0.0) or 0.0)

    def run_pretrain(max_steps: int) -> None:
        if opt_gen is None:
            return
        generator.train()
        pbar = tqdm(total=max_steps, desc="pretrain(gen)")
        steps = 0
        while steps < max_steps:
            for batch in train_loader:  # type: ignore
                if steps >= max_steps:
                    break
                images = batch["images"].to(device)
                masks = batch["masks"].to(device)
                pseudos = batch["pseudos"]
                # filter to those with pseudotargets
                has_pseudo = [i for i, p in enumerate(pseudos) if p is not None]
                opt_gen.zero_grad(set_to_none=True)
                if len(has_pseudo) == 0:
                    # Fallback: no pseudotargets available; do self-reconstruction to warm up generator
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        out = generator(images, masks)
                        l1 = torch.nn.functional.l1_loss(out, images)
                        perc = perc_loss(out, images)
                        loss = lcfg.get("l1_weight", 1.0) * l1 + lcfg.get("perceptual_weight", 0.1) * perc
                    scaler.scale(loss).backward()
                    scaler.step(opt_gen)
                    scaler.update()
                    steps += 1
                    pbar.set_postfix({"l1": float(l1.detach().cpu()), "perc": float(perc.detach().cpu()), "mode": "recon"})
                    pbar.update(1)
                    continue

                idx = torch.tensor(has_pseudo, dtype=torch.long)
                images_b = images[idx]
                masks_b = masks[idx]
                pseudos_b = torch.stack([p.to(device) for p in pseudos if p is not None], dim=0)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = generator(images_b, masks_b)
                    l1 = torch.nn.functional.l1_loss(out, pseudos_b)
                    perc = perc_loss(out, pseudos_b)
                    loss = lcfg.get("l1_weight", 1.0) * l1 + lcfg.get("perceptual_weight", 0.1) * perc
                scaler.scale(loss).backward()
                scaler.step(opt_gen)
                scaler.update()
                steps += 1
                pbar.set_postfix({"l1": float(l1.detach().cpu()), "perc": float(perc.detach().cpu()), "mode": "pseudo"})
                pbar.update(1)
        pbar.close()

    # optional pretraining
    pretrain_steps = int(cfg.get("train", {}).get("pretrain", {}).get("steps", 0))
    if max_steps_override is not None and mode == "pretrain":
        pretrain_steps = int(max_steps_override)
    if mode in ("auto", "pretrain") and cfg.get("train", {}).get("pretrain", {}).get("enabled", True):
        if pretrain_steps > 0:
            run_pretrain(pretrain_steps)
            # export a few pretrain samples
            sample_dir = os.path.join(cfg["paths"].get("outputs", "runs/joint_small"), "pretrain_samples")
            os.makedirs(sample_dir, exist_ok=True)
            try:
                import numpy as np
                from PIL import Image
                b = next(iter(train_loader))  # type: ignore
                vi = b["images"].to(device)[:8]
                vm = b["masks"].to(device)[:8]
                with torch.no_grad():
                    vf = generator(vi, vm)
                for i in range(vi.size(0)):
                    o = (vi[i].cpu().clamp(-1,1)*0.5+0.5).permute(1,2,0).numpy()
                    a = (vf[i].cpu().clamp(-1,1)*0.5+0.5).permute(1,2,0).numpy()
                    Image.fromarray((o*255).astype('uint8')).save(os.path.join(sample_dir, f"{i:02d}_orig.jpg"))
                    Image.fromarray((a*255).astype('uint8')).save(os.path.join(sample_dir, f"{i:02d}_anon.jpg"))
            except Exception as e:
                print(f"[WARN] Failed to write pretrain samples: {e}")
            # save pretrain checkpoint
            pre_ckpt = os.path.join(cfg["paths"].get("outputs", "runs/joint_small"), "pretrain_checkpoint.pt")
            save_checkpoint({"generator": generator.state_dict()}, pre_ckpt)
            if mode == "pretrain":
                return

    # main alternating training
    epochs = int(cfg["train"]["epochs"])
    max_steps: Optional[int] = cfg["train"].get("max_steps")
    if max_steps_override is not None and mode in ("auto", "joint"):
        max_steps = int(max_steps_override)

    det_steps = int(cfg["train"]["alternating"].get("det_steps", 1))
    gen_steps = int(cfg["train"]["alternating"].get("gen_steps", 1))
    disc_steps = int(cfg["train"]["alternating"].get("disc_steps", 1))

    global_step = 0
    # metrics CSV setup
    out_dir = cfg["paths"].get("outputs", "runs/joint_small")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "loss_gen", "loss_det", "arcface_mean_sim", "easyocr_plate_acc", "fid", "val_map"]) 
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}")
        for batch in pbar:
            # ensure shapes
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)

            # 1) detector updates (if GT exists)
            det_loss_val = None
            if det_steps > 0 and opt_det is not None and any(b.numel() > 0 for b in batch["boxes"]):
                detector.train()
                for _ in range(det_steps):
                    opt_det.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        det_loss, det_scalars = detector.compute_loss(batch)
                        det_loss_val = float(det_loss.detach().cpu())
                    scaler.scale(det_loss).backward()
                    if clip_norm > 0.0:
                        try:
                            scaler.unscale_(opt_det)
                            torch.nn.utils.clip_grad_norm_(detector.parameters(), clip_norm)
                        except Exception:
                            pass
                    scaler.step(opt_det)
                    scaler.update()

            # 2) discriminator updates
            if disc is not None and disc_steps > 0 and opt_disc is not None and opt_gen is not None:
                disc.train()
                with torch.no_grad():
                    fake = generator(images, masks)
                for _ in range(disc_steps):
                    opt_disc.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        pred_real = disc(images)
                        pred_fake = disc(fake.detach())
                        d_loss = adversarial_d_loss(pred_real, pred_fake)
                    scaler.scale(d_loss).backward()
                    if clip_norm > 0.0:
                        try:
                            scaler.unscale_(opt_disc)
                            torch.nn.utils.clip_grad_norm_(disc.parameters(), clip_norm)
                        except Exception:
                            pass
                    scaler.step(opt_disc)
                    scaler.update()

            # 3) generator updates
            g_scalars = {}
            loss_gen_val = None
            if gen_steps > 0 and opt_gen is not None:
                generator.train()
                prm = _get_train_prompts(cfg)
                has_face = any((lbl.numel() > 0 and (lbl == 1).any().item()) for lbl in batch["labels"])
                has_plate = any((lbl.numel() > 0 and (lbl == 2).any().item()) for lbl in batch["labels"])
                dyn_prompt = _compose_dynamic_prompt(
                    prm.get("base"), has_face, has_plate,
                    prm.get("face", "realistic anonymized face"),
                    prm.get("plate", "randomized license plate characters, realistic metal plate")
                )
                neg_prompt = prm.get("neg", "low quality, artifacts, distorted, blank plate, no characters")
                for _ in range(gen_steps):
                    opt_gen.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        fake = generator(images, masks, prompt=dyn_prompt, negative_prompt=neg_prompt)
                        # supervised losses (if pseudo present)
                        l1 = torch.tensor(0.0, device=device)
                        perc = torch.tensor(0.0, device=device)
                        with_pseudo = [i for i, p in enumerate(batch["pseudos"]) if p is not None]
                        if len(with_pseudo) > 0:
                            idx = torch.tensor(with_pseudo, dtype=torch.long, device=device)
                            pseudo = torch.stack([p.to(device) for p in batch["pseudos"] if p is not None], dim=0)
                            fake_sel = fake[idx]
                            l1 = torch.nn.functional.l1_loss(fake_sel, pseudo)
                            perc = perc_loss(fake_sel, pseudo)
                        loss = lcfg.get("l1_weight", 1.0) * l1 + lcfg.get("perceptual_weight", 0.1) * perc
                        # adversarial
                        if disc is not None:
                            pred_fake = disc(fake)
                            g_adv = adversarial_g_loss(pred_fake)
                            loss = loss + lcfg.get("adv_weight", 0.1) * g_adv
                            g_scalars["g_adv"] = float(g_adv.detach().cpu())
                        # ID suppression (no grad path through ArcFace features)
                        if lcfg.get("id_suppress_weight", 0.0) > 0:
                            sim = arcface_similarity(images, fake)
                            if sim is not None:
                                id_loss = sim.mean()  # minimize similarity
                                loss = loss + lcfg["id_suppress_weight"] * id_loss
                                g_scalars["id_sim"] = float(sim.mean().detach().cpu())
                        # Always add a tiny reconstruction when no pseudo targets to ensure gradient flow
                        if len(with_pseudo) == 0:
                            recon = torch.nn.functional.l1_loss(fake, images)
                            loss = loss + 0.01 * recon
                    scaler.scale(loss).backward()
                    if clip_norm > 0.0:
                        try:
                            scaler.unscale_(opt_gen)
                            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_norm)
                        except Exception:
                            pass
                    scaler.step(opt_gen)
                    scaler.update()

                g_scalars.update({
                    "l1": float(l1.detach().cpu()),
                    "perc": float(perc.detach().cpu()),
                })
                loss_gen_val = float((l1 + perc).detach().cpu())

            global_step += 1
            # logging to pbar
            show = {**g_scalars}
            pbar.set_postfix(show)

            # periodic eval and logging
            eval_cfg = cfg.get("eval", {})
            run_eval_every = int(eval_cfg.get("run_on_val_every_steps", 0) or 0)
            do_eval = (run_eval_every > 0 and (global_step % run_eval_every == 0))

            arcface_mean = math.nan
            easyocr_acc = math.nan
            fid_val = math.nan
            val_map = math.nan

            if do_eval and val_loader is not None:
                # small eval on a subset of val images
                max_images = int(eval_cfg.get("max_images", 50))
                collected = 0
                # save to outputs/val_samples/
                val_dir = os.path.join(out_dir, "val_samples", f"step{global_step}")
                real_dir = os.path.join(val_dir, "orig")
                anon_dir = os.path.join(val_dir, "anon")
                os.makedirs(real_dir, exist_ok=True)
                os.makedirs(anon_dir, exist_ok=True)
                orig_batch_list = []
                anon_batch_list = []
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb["images"].to(device)
                        vm = vb["masks"].to(device)
                        vf = generator(vi, vm)
                        # for ArcFace in-memory
                        orig_batch_list.append(vi.detach().cpu())
                        anon_batch_list.append(vf.detach().cpu())
                        # also dump images for OCR/FID/mAP
                        import numpy as np
                        from PIL import Image
                        for i in range(vi.size(0)):
                            if collected >= max_images:
                                break
                            o = (vi[i].cpu().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
                            a = (vf[i].cpu().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
                            Image.fromarray((o * 255).astype('uint8')).save(os.path.join(real_dir, f"{collected:06d}.jpg"))
                            Image.fromarray((a * 255).astype('uint8')).save(os.path.join(anon_dir, f"{collected:06d}.jpg"))
                            collected += 1
                        if collected >= max_images:
                            break
                # ArcFace similarity
                try:
                    sim = arcface_similarity(torch.cat(orig_batch_list, dim=0), torch.cat(anon_batch_list, dim=0))
                    if sim is not None:
                        arcface_mean = float(sim.mean().item())
                except Exception as e:
                    print(f"[WARN] ArcFace eval skipped: {e}")
                # EasyOCR heuristic metric
                try:
                    words = eval_utils.eval_easyocr(anon_dir, max_images)
                    if words is not None:
                        # convert to a pseudo-accuracy: fewer words => higher 'privacy'
                        easyocr_acc = float(max(0.0, 1.0 - min(1.0, words / 5.0)))
                except Exception as e:
                    print(f"[WARN] EasyOCR eval skipped: {e}")
                # FID
                try:
                    # compare anonymized to originals dumped
                    fid_val = eval_utils.eval_fid(real_dir, anon_dir, max_images)
                except Exception as e:
                    print(f"[WARN] FID eval skipped: {e}")
                # mAP if GT available
                try:
                    val_ann = cfg.get("paths", {}).get("val_annotations")
                    if val_ann:
                        val_map = eval_utils.eval_map(cfg, anon_dir, val_ann, max_images) or math.nan
                except Exception as e:
                    print(f"[WARN] mAP eval skipped: {e}")

            # checkpoints
            ckpt_cfg = cfg.get("checkpoint", {})
            if int(ckpt_cfg.get("save_every_steps", 200)) > 0 and (global_step % int(ckpt_cfg.get("save_every_steps", 200)) == 0):
                save_checkpoint({"detector": detector.state_dict()}, os.path.join(out_dir, f"detector_step{global_step}.pt"))
                if opt_gen is not None:
                    save_checkpoint({"generator": generator.state_dict()}, os.path.join(out_dir, f"generator_step{global_step}.pt"))
                if disc is not None:
                    save_checkpoint({"disc": disc.state_dict()}, os.path.join(out_dir, f"disc_step{global_step}.pt"))

            # write metrics row
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    global_step,
                    ep + 1,
                    loss_gen_val if loss_gen_val is not None else math.nan,
                    det_loss_val if det_loss_val is not None else math.nan,
                    arcface_mean,
                    easyocr_acc,
                    fid_val,
                    val_map,
                ])

            # step LR schedulers
            try:
                if sch_gen is not None: sch_gen.step()
                if sch_disc is not None: sch_disc.step()
                if sch_det is not None: sch_det.step()
            except Exception:
                pass
            # early cap
            if max_steps is not None and global_step >= int(max_steps):
                break
        if max_steps is not None and global_step >= int(max_steps):
            break


if __name__ == "__main__":
    # Improve Windows multiprocessing behavior (avoids unintended re-exec with different Python)
    try:
        mp.freeze_support()
        mp.set_executable(sys.executable)
        try:
            import torch.multiprocessing as tmp  # type: ignore
            tmp.set_executable(sys.executable)
        except Exception:
            pass
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--paths-overlay", type=str, default=None, help="Optional YAML with a 'paths' section to override base config paths")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto","pretrain","joint"], help="Run only pretrain, only joint, or auto (both)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps for current mode")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.paths_overlay and os.path.exists(args.paths_overlay):
        try:
            with open(args.paths_overlay, 'r', encoding='utf-8') as f:
                overlay = yaml.safe_load(f) or {}
            if isinstance(overlay, dict) and 'paths' in overlay and isinstance(overlay['paths'], dict):
                base = cfg.get('paths', {}) or {}
                base.update(overlay['paths'])
                cfg['paths'] = base
                print(f"[INFO] Applied paths overlay from {args.paths_overlay}")
        except Exception as e:
            print(f"[WARN] Failed to apply paths overlay: {e}")
    train(cfg, mode=args.mode, max_steps_override=args.max_steps)
