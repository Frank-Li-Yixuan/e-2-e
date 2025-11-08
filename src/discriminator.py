import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Any


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, spectral: bool = True) -> None:
        super().__init__()
        sn = spectral_norm if spectral else (lambda x: x)
        c = base_channels
        layers = [
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(c, c * 2, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(c * 4, c * 4, 3, stride=1, padding=1)), nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(c * 4, 1, 3, stride=1, padding=1)),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_discriminator(cfg: dict) -> PatchDiscriminator:
    dcfg = cfg.get("model", {}).get("discriminator", {})
    return PatchDiscriminator(
        in_channels=3,
        base_channels=int(dcfg.get("base_channels", 64)),
        spectral=bool(dcfg.get("spectral_norm", True)),
    )
