import argparse
import yaml
import torch
import sys

from src.generator_wrapper import GeneratorWrapper
from src.discriminator import build_discriminator
from src.losses import get_device

def probe(cfg_path: str, batch: int, image_size: int, gen_channels: int = None, disc_channels: int = None) -> bool:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # override batch and image size for the probe
    cfg.setdefault('train', {})['batch_size'] = batch
    cfg.setdefault('data', {})['image_size'] = image_size
    if gen_channels is not None:
        cfg.setdefault('model', {}).setdefault('generator', {})['base_channels'] = int(gen_channels)
    if disc_channels is not None:
        cfg.setdefault('model', {}).setdefault('discriminator', {})['base_channels'] = int(disc_channels)
    device = get_device(cfg)
    try:
        gen = GeneratorWrapper(cfg)
        disc = build_discriminator(cfg).to(device)
        b = batch
        h = w = image_size
        img = torch.randn(b, 3, h, w, device=device)
        msk = torch.randn(b, 1, h, w, device=device).clamp(0, 1)
        with torch.cuda.amp.autocast(enabled=(cfg.get('precision','fp32')!='fp32' and torch.cuda.is_available())):
            out = gen(img, msk)
            _ = disc(out)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            return False
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--image_size', type=int, required=True)
    p.add_argument('--gen_channels', type=int, default=None)
    p.add_argument('--disc_channels', type=int, default=None)
    args = p.parse_args()
    ok = probe(args.config, args.batch, args.image_size, args.gen_channels, args.disc_channels)
    print('fit', ok)
    sys.exit(0 if ok else 2)
