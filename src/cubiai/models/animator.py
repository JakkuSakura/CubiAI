"""Pass-through animator with latent controls and domain awareness."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset, DataLoader

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


# -----------------------------------------------------------------------------
# Blocks and encoders
# -----------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, norm: str = "in", act: str = "lrelu") -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
        if norm == "in":
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        elif norm == "bn":
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if act == "lrelu" else nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class DescriptorEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32, descriptor_ch: int = 64) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, stride=2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)
        self.enc4 = ConvBlock(base_ch * 4, descriptor_ch, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        descriptor_map = self.enc4(x)
        descriptor_vec = self.pool(descriptor_map).flatten(1)
        return descriptor_map, descriptor_vec


class ImageUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64, latent_dim: int = 128) -> None:
        super().__init__()
        ch = base_ch
        self.enc1 = ConvBlock(in_ch, ch)
        self.enc2 = ConvBlock(ch, ch * 2, stride=2)
        self.enc3 = ConvBlock(ch * 2, ch * 4, stride=2)
        self.enc4 = ConvBlock(ch * 4, ch * 8, stride=2)
        self.enc5 = ConvBlock(ch * 8, ch * 8, stride=2)

        self.bot = ConvBlock(ch * 8, ch * 8)

        self.up4 = UpBlock(ch * 8, ch * 8)
        self.up3 = UpBlock(ch * 8, ch * 4)
        self.up2 = UpBlock(ch * 4, ch * 2)
        self.up1 = UpBlock(ch * 2, ch)

        self.dec4 = ConvBlock(ch * 8 + ch * 8, ch * 8)
        self.dec3 = ConvBlock(ch * 4 + ch * 4, ch * 4)
        self.dec2 = ConvBlock(ch * 2 + ch * 2, ch * 2)
        self.dec1 = ConvBlock(ch + ch, ch)

        self.rgb_head = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

        self.film4 = nn.Linear(latent_dim, ch * 8 * 2)
        self.film3 = nn.Linear(latent_dim, ch * 4 * 2)
        self.film2 = nn.Linear(latent_dim, ch * 2 * 2)
        self.film1 = nn.Linear(latent_dim, ch * 2 * 2)

    @staticmethod
    def _apply_film(tensor: torch.Tensor, film: torch.Tensor, channels: int) -> torch.Tensor:
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(-1, channels, 1, 1)
        beta = beta.view(-1, channels, 1, 1)
        return tensor * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, control: torch.Tensor | None = None) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bot(e5)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        if control is not None:
            d4 = self._apply_film(d4, self.film4(control), d4.shape[1])
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        if control is not None:
            d3 = self._apply_film(d3, self.film3(control), d3.shape[1])
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        if control is not None:
            d2 = self._apply_film(d2, self.film2(control), d2.shape[1])
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        if control is not None:
            d1 = self._apply_film(d1, self.film1(control), d1.shape[1])

        residual = self.rgb_head(d1)
        return residual


class Animator(nn.Module):
    def __init__(
        self,
        *,
        descriptor_ch: int = 64,
        latent_dim: int = 128,
        num_domains: int = 2,
    ) -> None:
        super().__init__()
        self.descriptor_encoder = DescriptorEncoder(in_ch=3, base_ch=32, descriptor_ch=descriptor_ch)
        self.translator = ImageUNet(in_ch=3 + descriptor_ch, base_ch=64, latent_dim=latent_dim)
        self.delta_proj = nn.Linear(descriptor_ch, latent_dim)
        self.domain_embed = nn.Embedding(num_domains, latent_dim)

    def forward(
        self,
        src_hr: torch.Tensor,
        drv_hr: torch.Tensor,
        *,
        strength: float = 1.0,
        driver_domain: int | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        b, _, h, w = src_hr.shape

        desc_src_map, desc_src_vec = self.descriptor_encoder(src_hr)
        desc_drv_map, desc_drv_vec = self.descriptor_encoder(drv_hr)
        delta_map = desc_drv_map - desc_src_map

        delta_up = F.interpolate(delta_map, size=(h, w), mode="bilinear", align_corners=False)
        latent_vec = desc_drv_vec - desc_src_vec
        latent_vec = self.delta_proj(latent_vec)

        if driver_domain is not None:
            if isinstance(driver_domain, torch.Tensor):
                domain_vec = self.domain_embed(driver_domain.to(latent_vec.device))
            else:
                domain_vec = self.domain_embed(
                    torch.full((b,), driver_domain, device=latent_vec.device, dtype=torch.long)
                )
            latent_vec = latent_vec + domain_vec
        latent_vec = latent_vec / (latent_vec.norm(dim=-1, keepdim=True) + 1e-6)

        translator_input = torch.cat([src_hr, delta_up], dim=1)
        residual = self.translator(translator_input, control=latent_vec)
        output = (src_hr + strength * residual).clamp(-1.0, 1.0)

        desc_out_map, _ = self.descriptor_encoder(output)

        return {
            "output": output,
            "descriptor_ref": desc_src_map,
            "descriptor_drv": desc_drv_map,
            "descriptor_out": desc_out_map,
        }

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class DatasetEntry:
    portrait: Path
    frames: List[Path]
    domain: int


class PortraitVideoDataset(Dataset):
    def __init__(self, root: Path | str, *, size: int = 1024, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> None:
        super().__init__()
        self.root = Path(root)
        self.size = size
        self.exts = exts
        self.entries: List[DatasetEntry] = []

        for portrait_path in sorted(self.root.iterdir()):
            if not portrait_path.is_file() or portrait_path.suffix.lower() not in self.exts:
                continue
            video_dir = self.root / f"{portrait_path.stem}_video"
            if not video_dir.is_dir():
                continue
            frames = [p for p in sorted(video_dir.iterdir()) if p.suffix.lower() in self.exts]
            if not frames:
                continue
            domain = 1 if portrait_path.stem.endswith("_real") else 0
            self.entries.append(DatasetEntry(portrait_path, frames, domain))

        if not self.entries:
            raise ValueError(
                "No portrait/video pairs found. Layout should be `name.png` with `name_video/00.png`."
            )

        self.index: List[Tuple[int, int]] = [
            (entry_idx, frame_idx)
            for entry_idx, entry in enumerate(self.entries)
            for frame_idx in range(len(entry.frames))
        ]

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _prepare(path: Path, size: int) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((size, size), Resampling.BICUBIC)
            arr = np.array(im).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            return tensor * 2.0 - 1.0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entry_idx, frame_idx = self.index[idx]
        entry = self.entries[entry_idx]
        src = self._prepare(entry.portrait, self.size)
        drv = self._prepare(entry.frames[frame_idx], self.size)
        tgt = drv.clone()
        domain = torch.tensor(entry.domain, dtype=torch.long)
        return src, drv, tgt, domain

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class TrainerConfig:
    lambda_align: float = 0.3
    lambda_motion: float = 0.1
    lambda_identity: float = 0.1
    clip_grad: float = 1.0
    strength: float = 1.0


class PassThroughTrainer:
    def __init__(self, model: Animator, *, lr: float = 2e-4, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], cfg: TrainerConfig) -> Dict[str, float]:
        src, drv, tgt, domain = batch
        src = src.to(self.device)
        drv = drv.to(self.device)
        tgt = tgt.to(self.device)
        domain = domain.to(self.device, dtype=torch.long)

        out = self.model(src, drv, strength=cfg.strength, driver_domain=domain)
        pred = out["output"]

        loss_rec = (pred - tgt).abs().mean()
        descriptor_ref = out["descriptor_ref"]
        descriptor_drv = out["descriptor_drv"]
        descriptor_out = out["descriptor_out"]

        delta_target = descriptor_drv - descriptor_ref
        delta_output = descriptor_out - descriptor_ref
        align_diff = delta_output - delta_target

        loss_align = align_diff.pow(2).mean()
        loss_motion = total_variation(align_diff)

        loss = loss_rec + cfg.lambda_align * loss_align + cfg.lambda_motion * loss_motion

        if cfg.lambda_identity > 0:
            out_id = self.model(src, src, strength=0.2, driver_domain=domain)
            loss_id = (out_id["output"] - src).abs().mean()
            loss = loss + cfg.lambda_identity * loss_id
        else:
            loss_id = torch.tensor(0.0, device=self.device)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad)
        self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "l_rec": float(loss_rec.detach().cpu()),
            "l_align": float(loss_align.detach().cpu()),
            "l_motion": float(loss_motion.detach().cpu()),
            "l_id": float(loss_id.detach().cpu()),
            "residual_mag": float((pred - src).abs().mean().detach().cpu()),
        }

# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------


def create_dataloader(root: Path | str, *, size: int = 1024, batch_size: int = 1, num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    dataset = PortraitVideoDataset(root=root, size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


__all__ = [
    "Animator",
    "PassThroughTrainer",
    "TrainerConfig",
    "PortraitVideoDataset",
    "create_dataloader",
    "total_variation",
]
