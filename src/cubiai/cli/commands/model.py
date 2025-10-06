"""CLI commands for the pass-through animator model."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import typer
from PIL.Image import Resampling

from tqdm.auto import tqdm

from ...models.animator import Animator, PassThroughTrainer, TrainerConfig, create_dataloader

app = typer.Typer(help="Commands for training and evaluating the pass-through animator.")


@app.command()
def train(
        data_root: Path = typer.Argument(..., help="Dataset with portrait.png + portrait_video/frame.png"),
        workdir: Path = typer.Argument(Path("runs/animator"), help="Where to save artifacts"),
        size: int = typer.Option(1024, help="High-resolution size"),
        steps: int = typer.Option(200, help="Total optimisation steps"),
        batch: int = typer.Option(1, help="Batch size"),
        epochs: int = typer.Option(1, help="Epochs (upper bound if dataset small)"),
        lr: float = typer.Option(2e-4, help="Learning rate"),
        lambda_align: float = typer.Option(0.3, help="Weight for driver-aligned motion"),
        lambda_motion: float = typer.Option(0.1, help="Weight for motion magnitude penalty"),
        device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Torch device"),
        num_workers: int = typer.Option(2, help="DataLoader workers"),
) -> None:
    """Train the pass-through animator on the given dataset."""

    dataloader = create_dataloader(data_root, size=size, batch_size=batch, num_workers=num_workers)
    model = Animator()
    trainer = PassThroughTrainer(model, lr=lr, device=device)
    cfg = TrainerConfig(
        lambda_align=lambda_align,
        lambda_motion=lambda_motion,
    )

    global_step = 0
    start_time = perf_counter()

    progress = tqdm(total=steps, desc="training", unit="step", dynamic_ncols=True)

    for epoch in range(epochs):
        for batch_data in dataloader:
            metrics = trainer.training_step(batch_data, cfg)
            global_step += 1

            progress.update(1)
            if global_step % 50 == 0:
                progress.set_postfix(
                    {k: f"{v:.3f}" for k, v in metrics.items()}, refresh=False
                )

            elif global_step % 50 == 0:
                elapsed = perf_counter() - start_time
                rate = elapsed / global_step if global_step > 0 else 0.0
                remaining = max(steps - global_step, 0)
                eta = remaining * rate
                typer.echo(
                    f"step {global_step}/{steps} | eta {eta:0.1f}s | metrics {metrics}"
                )

            if global_step >= steps:
                break
        if global_step >= steps:
            break

    progress.close()

    total_elapsed = perf_counter() - start_time
    typer.echo(f"training finished in {total_elapsed:0.1f}s over {global_step} steps")

    # Save checkpoint and quick preview
    workdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), workdir / "pass_through.pt")

    sample = next(iter(dataloader))
    if not isinstance(sample, (list, tuple)):
        raise ValueError("Expected the dataloader to return a sequence")

    if len(sample) < 2:
        raise ValueError("Dataloader batch should include at least source and driver tensors")

    src = sample[0].to(device)
    drv = sample[1].to(device)

    domain = sample[-1] if len(sample) > 2 else None
    if isinstance(domain, torch.Tensor):
        # Flatten in case the batch dimension is present
        domain_id = int(domain.view(-1)[0].item())
    elif domain is not None:
        domain_id = int(domain)
    else:
        domain_id = 0
    with torch.no_grad():
        out = model(src, drv, strength=0.8, driver_domain=domain_id)
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
        strength: float = typer.Option(1.0, help="Deformation strength"),
        driver_domain: int = typer.Option(0, help="Driver domain id (0=default, 1=real, etc.)"),
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

    model = Animator()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    with torch.no_grad():
        out = model(src.to(device), drv.to(device), strength=strength, driver_domain=driver_domain)
        result = out["output"].clamp(-1, 1)
        result = ((result[0].cpu() + 1.0) * 0.5).permute(1, 2, 0).numpy()
        from PIL import Image

        Image.fromarray((result * 255).astype("uint8")).save(output)
        typer.echo(f"Saved output to {output}")
