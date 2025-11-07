import argparse
import os
import sys
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Path to base YAML config")
    p.add_argument("--out", required=True, help="Path to write the generated YAML config")
    p.add_argument("--outputs_dir", required=True, help="Override paths.outputs")
    p.add_argument("--epochs", type=int, required=True, help="Override train.epochs")
    p.add_argument("--save_every", type=int, required=True, help="Override checkpoint.save_every_steps")
    p.add_argument("--resume", required=True, help="Override checkpoint.resume (generator checkpoint)")
    # Optional performance-related overrides
    p.add_argument("--batch_size", type=int, default=None, help="Override train.batch_size")
    p.add_argument("--num_workers", type=int, default=None, help="Override train.num_workers")
    p.add_argument("--precision", type=str, default=None, choices=["fp32","fp16","bf16"], help="Override precision")
    p.add_argument("--image_size", type=int, default=None, help="Override data.image_size")
    p.add_argument("--gen_base_channels", type=int, default=None, help="Override model.generator.base_channels")
    p.add_argument("--disc_base_channels", type=int, default=None, help="Override model.discriminator.base_channels")
    p.add_argument("--gen_torch_compile", action="store_true", help="Enable torch.compile for generator (UNet backend)")
    p.add_argument("--disc_torch_compile", action="store_true", help="Enable torch.compile for discriminator")
    p.add_argument("--gen_steps", type=int, default=None, help="Override train.alternating.gen_steps")
    p.add_argument("--disc_steps", type=int, default=None, help="Override train.alternating.disc_steps")
    p.add_argument("--det_steps", type=int, default=None, help="Override train.alternating.det_steps")
    # Loss overrides
    p.add_argument("--loss_l1_weight", type=float, default=None, help="Override loss.l1_weight")
    p.add_argument("--loss_perceptual_weight", type=float, default=None, help="Override loss.perceptual_weight")
    p.add_argument("--loss_adv_weight", type=float, default=None, help="Override loss.adv_weight")
    p.add_argument("--loss_id_suppress_weight", type=float, default=None, help="Override loss.id_suppress_weight")
    # Reproducibility
    p.add_argument("--repro_deterministic", action="store_true", help="Enable deterministic algorithms for reproducibility")
    args = p.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ensure nested keys exist
    cfg.setdefault("paths", {})
    cfg.setdefault("train", {})
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("repro", {})

    # Apply overrides
    cfg["paths"]["outputs"] = args.outputs_dir
    cfg["train"]["epochs"] = int(args.epochs)
    cfg["checkpoint"]["save_every_steps"] = int(args.save_every)
    cfg["checkpoint"]["resume"] = args.resume
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = int(args.num_workers)
    if args.precision is not None:
        cfg["precision"] = args.precision
    if args.image_size is not None:
        cfg["data"]["image_size"] = int(args.image_size)
    if args.gen_base_channels is not None or args.gen_torch_compile:
        cfg.setdefault("model", {}).setdefault("generator", {})
        if args.gen_base_channels is not None:
            cfg["model"]["generator"]["base_channels"] = int(args.gen_base_channels)
        if args.gen_torch_compile:
            cfg["model"]["generator"]["torch_compile"] = True
    if args.disc_base_channels is not None or args.disc_torch_compile:
        cfg.setdefault("model", {}).setdefault("discriminator", {})
        if args.disc_base_channels is not None:
            cfg["model"]["discriminator"]["base_channels"] = int(args.disc_base_channels)
        if args.disc_torch_compile:
            cfg["model"]["discriminator"]["torch_compile"] = True
    # alternating steps overrides
    if args.gen_steps is not None or args.disc_steps is not None or args.det_steps is not None:
        alt = cfg.setdefault("train", {}).setdefault("alternating", {})
        if args.gen_steps is not None:
            alt["gen_steps"] = int(args.gen_steps)
        if args.disc_steps is not None:
            alt["disc_steps"] = int(args.disc_steps)
        if args.det_steps is not None:
            alt["det_steps"] = int(args.det_steps)

    # Loss overrides
    if args.loss_l1_weight is not None:
        cfg["loss"]["l1_weight"] = float(args.loss_l1_weight)
    if args.loss_perceptual_weight is not None:
        cfg["loss"]["perceptual_weight"] = float(args.loss_perceptual_weight)
    if args.loss_adv_weight is not None:
        cfg["loss"]["adv_weight"] = float(args.loss_adv_weight)
    if args.loss_id_suppress_weight is not None:
        cfg["loss"]["id_suppress_weight"] = float(args.loss_id_suppress_weight)

    # Reproducibility
    if args.repro_deterministic:
        cfg["repro"]["deterministic"] = True

    # Keep in-training eval disabled for long runs unless explicitly enabled in base
    eval_cfg = cfg.setdefault("eval", {})
    if eval_cfg.get("run_on_val_every_steps", 0) is None or eval_cfg.get("run_on_val_every_steps", 0) != 0:
        eval_cfg["run_on_val_every_steps"] = 0
    metrics = eval_cfg.setdefault("metrics", {})
    for k in ("arcface", "easyocr", "fid", "map"):
        if metrics.get(k) not in (False, 0):
            metrics[k] = False

    # Write out
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[INFO] Wrote generated config: {args.out}")
    print(f"[INFO] outputs_dir={args.outputs_dir}, epochs={args.epochs}, save_every={args.save_every}, resume={args.resume}")
    if args.batch_size is not None:
        print(f"[INFO] batch_size={args.batch_size}")
    if args.num_workers is not None:
        print(f"[INFO] num_workers={args.num_workers}")
    if args.precision is not None:
        print(f"[INFO] precision={args.precision}")
    if args.image_size is not None:
        print(f"[INFO] image_size={args.image_size}")
    if args.gen_base_channels is not None:
        print(f"[INFO] gen_base_channels={args.gen_base_channels}")
    if args.disc_base_channels is not None:
        print(f"[INFO] disc_base_channels={args.disc_base_channels}")
    if args.gen_torch_compile:
        print("[INFO] gen_torch_compile=True")
    if args.disc_torch_compile:
        print("[INFO] disc_torch_compile=True")
    if args.gen_steps is not None:
        print(f"[INFO] gen_steps={args.gen_steps}")
    if args.disc_steps is not None:
        print(f"[INFO] disc_steps={args.disc_steps}")
    if args.det_steps is not None:
        print(f"[INFO] det_steps={args.det_steps}")
    if args.loss_l1_weight is not None:
        print(f"[INFO] loss.l1_weight={args.loss_l1_weight}")
    if args.loss_perceptual_weight is not None:
        print(f"[INFO] loss.perceptual_weight={args.loss_perceptual_weight}")
    if args.loss_adv_weight is not None:
        print(f"[INFO] loss.adv_weight={args.loss_adv_weight}")
    if args.loss_id_suppress_weight is not None:
        print(f"[INFO] loss.id_suppress_weight={args.loss_id_suppress_weight}")
    if args.repro_deterministic:
        print("[INFO] repro.deterministic=True")


if __name__ == "__main__":
    main()
