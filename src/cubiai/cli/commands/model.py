"""CLI commands for the pass-through animator model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from PIL.Image import Resampling

from ...models.animator import (
    PassThroughAnimator,
    PassThroughTrainer,
    TrainerConfig,
    create_dataloader,
)

app = typer.Typer(help="Commands for training and evaluating the pass-through animator.")


@app.command()
def train(
    data_root: Path = typer.Argument(..., help="Dataset organised as root/video_id/frame.png"),
    workdir: Path = typer.Argument(Path("runs/pass_through"), help="Where to save artifacts"),
    size: int = typer.Option(1024, help="High-resolution size"),
    low_res: int = typer.Option(256, help="Resolution for flow estimation"),
    steps: int = typer.Option(2000, help="Total optimisation steps"),
    batch: int = typer.Option(1, help="Batch size"),
    epochs: int = typer.Option(1, help="Epochs (upper bound if dataset small)"),
    lr: float = typer.Option(2e-4, help="Learning rate"),
    lambda_tv: float = typer.Option(0.1, help="Weight for flow total variation"),
    lambda_mask: float = typer.Option(0.02, help="Weight for mask penalty"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Torch device"),
    num_workers: int = typer.Option(2, help="DataLoader workers"),
) -> None:
    """Train the pass-through animator on the given dataset."""

    dataloader = create_dataloader(data_root, size=size, batch_size=batch, num_workers=num_workers)
    model = PassThroughAnimator(low_res=low_res)
    trainer = PassThroughTrainer(model, lr=lr, device=device)
    cfg = TrainerConfig(lambda_tv=lambda_tv, lambda_mask=lambda_mask)

    global_step = 0
    for epoch in range(epochs):
        for batch_data in dataloader:
            metrics = trainer.training_step(batch_data, cfg)
            if global_step % 50 == 0:
                typer.echo(f"step {global_step}: {metrics}")
            global_step += 1
            if global_step >= steps:
                break
        if global_step >= steps:
            break

    # Save checkpoint and quick preview
    workdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), workdir / "pass_through.pt")

    src, drv, _ = next(iter(dataloader))
    src = src.to(device)
    drv = drv.to(device)
    with torch.no_grad():
        out = model(src, drv, strength=0.8)
        result = out["output"].clamp(-1, 1)
        result = ((result[0].cpu() + 1.0) * 0.5).permute(1, 2, 0).numpy()
        from PIL import Image

        Image.fromarray((result * 255).astype("uint8")).save(workdir / "preview.png")
        typer.echo(f"Saved preview to {workdir / 'preview.png'}")


@app.command()
def infer(
    source: Path = typer.Argument(..., help="Path to source image"),
    driver: Path = typer.Argument(..., help="Path to driver image"),
    checkpoint: Path = typer.Option(Path("runs/pass_through/pass_through.pt"), help="Model weights"),
    size: int = typer.Option(1024, help="Resize/crop size"),
    low_res: int = typer.Option(256, help="Flow estimation resolution"),
    strength: float = typer.Option(1.0, help="Deformation strength"),
    output: Path = typer.Option(Path("output.png"), help="Where to save the rendered result"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Run the animator on a single source/driver pair."""

    def load_image(path: Path) -> torch.Tensor:
        from PIL import Image
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

    src = load_image(source).unsqueeze(0)
    drv = load_image(driver).unsqueeze(0)

    model = PassThroughAnimator(low_res=low_res)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    with torch.no_grad():
        out = model(src.to(device), drv.to(device), strength=strength)
        result = out["output"].clamp(-1, 1)
        result = ((result[0].cpu() + 1.0) * 0.5).permute(1, 2, 0).numpy()
        from PIL import Image

        Image.fromarray((result * 255).astype("uint8")).save(output)
        typer.echo(f"Saved output to {output}")
