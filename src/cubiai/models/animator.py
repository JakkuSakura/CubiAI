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

# -----------------------------------------------------------------------------
# Warping utilities
# -----------------------------------------------------------------------------


def make_base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1)


def warp_with_flow(img: torch.Tensor, flow_norm: torch.Tensor) -> torch.Tensor:
    b_img, _, h, w = img.shape
    b_flow = flow_norm.shape[0]
    if b_flow != b_img:
        if b_flow == 1:
            flow_norm = flow_norm.expand(b_img, -1, -1, -1)
        elif b_img == 1:
            img = img.expand(b_flow, -1, -1, -1)
            b_img = b_flow
        else:
            raise RuntimeError(
                f"grid_sampler expects matching batch sizes, got img batch {b_img}, flow batch {b_flow}"
            )
    base = make_base_grid(h, w, img.device).unsqueeze(0).repeat(b_img, 1, 1, 1)
    grid = base + flow_norm.permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


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


class MotionEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32, latent_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_ch, base_ch, stride=2),
            ConvBlock(base_ch, base_ch * 2, stride=2),
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),
            ConvBlock(base_ch * 4, base_ch * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_ch * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        latent = self.head(pooled)
        return latent


class FlowUNet(nn.Module):
    def __init__(self, in_ch: int = 6, base_ch: int = 64, latent_dim: int = 128) -> None:
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

        self.flow_head = nn.Conv2d(ch, 2, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(ch, 1, kernel_size=3, padding=1)

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

    def forward(self, x: torch.Tensor, control: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bot(e5)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        if control is not None:
            d4 = self._apply_film(d4, self.film4(control), d4.shape[1])
            d3 = self._apply_film(d3, self.film3(control), d3.shape[1])
            d2 = self._apply_film(d2, self.film2(control), d2.shape[1])
            d1 = self._apply_film(d1, self.film1(control), d1.shape[1])

        flow = torch.tanh(self.flow_head(d1))
        mask = torch.sigmoid(self.mask_head(d1))
        return flow, mask


class PassThroughAnimator(nn.Module):
    def __init__(
        self,
        *,
        low_res: int = 256,
        latent_dim: int = 128,
        max_flow: float = 0.08,
        mask_bias: float = 0.3,
        num_domains: int = 2,
    ) -> None:
        super().__init__()
        self.low_res = low_res
        self.max_flow = max_flow
        self.mask_bias = mask_bias
        self.flow_net = FlowUNet(in_ch=6, base_ch=64, latent_dim=latent_dim)
        self.motion_encoder = MotionEncoder(in_ch=3, base_ch=32, latent_dim=latent_dim)
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
        src_lr = F.interpolate(src_hr, size=(self.low_res, self.low_res), mode="bilinear", align_corners=True)
        drv_lr = F.interpolate(drv_hr, size=(self.low_res, self.low_res), mode="bilinear", align_corners=True)
        flow_input = torch.cat([src_lr, drv_lr], dim=1)

        latent = self.motion_encoder(drv_lr)
        if driver_domain is not None:
            if isinstance(driver_domain, torch.Tensor):
                domain_vec = self.domain_embed(driver_domain.to(latent.device))
            else:
                domain_vec = self.domain_embed(
                    torch.full((b,), driver_domain, device=latent.device, dtype=torch.long)
                )
            latent = latent + domain_vec
        latent = latent / (latent.norm(dim=-1, keepdim=True) + 1e-6)

        flow_lr, mask_lr = self.flow_net(flow_input, control=latent)
        flow_lr = flow_lr * (self.max_flow * strength)
        mask_lr = torch.sigmoid(mask_lr - self.mask_bias)

        flow_hr = F.interpolate(flow_lr, size=(h, w), mode="bilinear", align_corners=True)
        mask_hr = F.interpolate(mask_lr, size=(h, w), mode="bilinear", align_corners=True)

        warped = warp_with_flow(src_hr, flow_hr)
        out = mask_hr * warped + (1.0 - mask_hr) * src_hr

        return {
            "output": out,
            "warped": warped,
            "mask": mask_hr,
            "flow": flow_hr,
            "control": latent,
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
    lambda_tv: float = 0.1
    lambda_mask: float = 0.02
    lambda_identity: float = 0.1
    lambda_control: float = 0.01
    clip_grad: float = 1.0
    strength: float = 1.0


class PassThroughTrainer:
    def __init__(self, model: PassThroughAnimator, *, lr: float = 2e-4, device: str | torch.device = "cuda") -> None:
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
        flow = out["flow"]
        mask = out["mask"]
        control = out["control"]

        loss_rec = (pred - tgt).abs().mean()
        loss_tv = total_variation(flow)
        loss_mask = mask.mean()
        loss_control = control.pow(2).mean()

        loss = loss_rec + cfg.lambda_tv * loss_tv + cfg.lambda_mask * loss_mask + cfg.lambda_control * loss_control

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
            "l_tv": float(loss_tv.detach().cpu()),
            "l_mask": float(loss_mask.detach().cpu()),
            "l_control": float(loss_control.detach().cpu()),
            "l_id": float(loss_id.detach().cpu()),
            "flow_mag": float(flow.abs().mean().detach().cpu()),
        }

# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------


def create_dataloader(root: Path | str, *, size: int = 1024, batch_size: int = 1, num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    dataset = PortraitVideoDataset(root=root, size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


__all__ = [
    "PassThroughAnimator",
    "PassThroughTrainer",
    "TrainerConfig",
    "PortraitVideoDataset",
    "create_dataloader",
    "warp_with_flow",
    "total_variation",
]
