"""Lightweight pass-through animator with learned flow and mask."""
from __future__ import annotations

import random
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
    """Return a normalized [-1, 1] meshgrid suitable for grid_sample."""
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)  # (..., 2) order: x then y
    return grid

def warp_with_flow(img: torch.Tensor, flow_norm: torch.Tensor) -> torch.Tensor:
    """Warp image using normalized flow (dx, dy in [-1,1])."""
    b, _, h, w = img.shape
    base = make_base_grid(h, w, img.device).unsqueeze(0).repeat(b, 1, 1, 1)
    flow = flow_norm.permute(0, 2, 3, 1)
    grid = base + flow
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Isotropic TV loss."""
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, norm: str = "in", act: str = "lrelu") -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
        if norm == "in":
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        elif norm == "bn":
            layers.append(nn.BatchNorm2d(out_ch))
        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, norm: str = "in", act: str = "lrelu") -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class FlowUNet(nn.Module):
    """UNet backbone that predicts low-res flow and mask."""

    def __init__(self, in_ch: int = 6, base_ch: int = 64) -> None:
        super().__init__()
        ch = base_ch
        self.enc1 = ConvBlock(in_ch, ch)
        self.enc2 = ConvBlock(ch, ch * 2, stride=2)
        self.enc3 = ConvBlock(ch * 2, ch * 4, stride=2)
        self.enc4 = ConvBlock(ch * 4, ch * 8, stride=2)
        self.enc5 = ConvBlock(ch * 8, ch * 8, stride=2)

        self.bot = ConvBlock(ch * 8, ch * 8)

        self.up4 = UpBlock(ch * 8, ch * 8)
        self.dec4 = ConvBlock(ch * 8 + ch * 8, ch * 8)

        self.up3 = UpBlock(ch * 8, ch * 4)
        self.dec3 = ConvBlock(ch * 4 + ch * 4, ch * 4)

        self.up2 = UpBlock(ch * 4, ch * 2)
        self.dec2 = ConvBlock(ch * 2 + ch * 2, ch * 2)

        self.up1 = UpBlock(ch * 2, ch)
        self.dec1 = ConvBlock(ch + ch, ch)

        self.flow_head = nn.Conv2d(ch, 2, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        flow = torch.tanh(self.flow_head(d1))  # normalized [-1,1]
        mask = torch.sigmoid(self.mask_head(d1))
        return flow, mask


class PassThroughAnimator(nn.Module):
    """High-res animator with low-res flow estimation and blending."""

    def __init__(self, *, low_res: int = 256, max_flow: float = 0.08, mask_bias: float = 0.3) -> None:
        super().__init__()
        self.low_res = low_res
        self.max_flow = max_flow
        self.mask_bias = mask_bias
        self.flow_net = FlowUNet(in_ch=6, base_ch=64)

    def forward(self, src_hr: torch.Tensor, drv_hr: torch.Tensor, *, strength: float = 1.0) -> Dict[str, torch.Tensor]:
        b, _, h, w = src_hr.shape

        src_lr = F.interpolate(src_hr, size=(self.low_res, self.low_res), mode="bilinear", align_corners=True)
        drv_lr = F.interpolate(drv_hr, size=(self.low_res, self.low_res), mode="bilinear", align_corners=True)
        inp = torch.cat([src_lr, drv_lr], dim=1)

        flow_lr, mask_lr = self.flow_net(inp)
        flow_lr = flow_lr * (self.max_flow * strength)
        mask_lr = torch.clamp(mask_lr - self.mask_bias, min=-10, max=10)
        mask_lr = torch.sigmoid(mask_lr)

        flow_hr = F.interpolate(flow_lr, size=(h, w), mode="bilinear", align_corners=True)
        mask_hr = F.interpolate(mask_lr, size=(h, w), mode="bilinear", align_corners=True)

        warped = warp_with_flow(src_hr, flow_hr)
        y = mask_hr * warped + (1.0 - mask_hr) * src_hr

        return {
            "output": y,
            "warped": warped,
            "mask": mask_hr,
            "flow": flow_hr,
        }


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PairSample:
    source: torch.Tensor
    driver: torch.Tensor
    target: torch.Tensor

class FolderPairDataset(Dataset):
    """Dataset organised as root/video_id/frame.png."""

    def __init__(self, root: Path | str, *, size: int = 1024, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> None:
        super().__init__()
        self.root = Path(root)
        self.size = size
        self.exts = exts
        self.videos: List[List[Path]] = []
        for video_dir in sorted(self.root.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = [p for p in sorted(video_dir.iterdir()) if p.suffix.lower() in self.exts]
            if len(frames) < 4:
                continue
            self.videos.append(frames)
        self.index: List[Tuple[int, int]] = [(vid_idx, f_idx) for vid_idx, frames in enumerate(self.videos) for f_idx in range(len(frames))]
        if not self.index:
            raise ValueError(f"No video folders with enough frames found in {self.root}")

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _prepare_image(path: Path, size: int) -> torch.Tensor:
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
            tensor = tensor * 2.0 - 1.0
            return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vid_idx, src_idx = self.index[idx]
        frames = self.videos[vid_idx]
        drv_idx = random.randrange(0, len(frames))
        src = self._prepare_image(frames[src_idx], self.size)
        drv = self._prepare_image(frames[drv_idx], self.size)
        tgt = self._prepare_image(frames[drv_idx], self.size)
        return src, drv, tgt


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class TrainerConfig:
    lambda_tv: float = 0.1
    lambda_mask: float = 0.02
    clip_grad: float = 1.0
    strength: float = 1.0

class PassThroughTrainer:
    def __init__(self, model: PassThroughAnimator, *, lr: float = 2e-4, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], cfg: TrainerConfig) -> Dict[str, float]:
        src, drv, tgt = [tensor.to(self.device) for tensor in batch]
        outputs = self.model(src, drv, strength=cfg.strength)
        pred = outputs["output"]
        flow = outputs["flow"]
        mask = outputs["mask"]

        loss_rec = (pred - tgt).abs().mean()
        loss_tv = total_variation(flow)
        loss_mask = mask.mean()

        loss = loss_rec + cfg.lambda_tv * loss_tv + cfg.lambda_mask * loss_mask

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad)
        self.opt.step()

        metrics = {
            "loss": float(loss.detach().cpu()),
            "l_rec": float(loss_rec.detach().cpu()),
            "l_tv": float(loss_tv.detach().cpu()),
            "l_mask": float(loss_mask.detach().cpu()),
            "flow_mag": float(flow.abs().mean().detach().cpu()),
        }
        return metrics


# -----------------------------------------------------------------------------
# Convenience factory
# -----------------------------------------------------------------------------

def create_dataloader(root: Path | str, *, size: int = 1024, batch_size: int = 1, num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    dataset = FolderPairDataset(root=root, size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


__all__ = [
    "PassThroughAnimator",
    "PassThroughTrainer",
    "TrainerConfig",
    "FolderPairDataset",
    "create_dataloader",
    "warp_with_flow",
    "total_variation",
]
