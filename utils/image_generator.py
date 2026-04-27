"""
Local image generator using Stable Diffusion via HuggingFace diffusers.

The pipeline is loaded once (singleton) and reused across all scenes
to avoid the multi-GB model reload penalty on every image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class _SDPipeline:
    """Lazy-loaded singleton Stable Diffusion pipeline."""

    _pipe = None
    _loaded_model: Optional[str] = None
    _loaded_device: Optional[str] = None

    @classmethod
    def get(cls, model_id: str, device: str):
        if cls._pipe is None or cls._loaded_model != model_id:
            cls._load(model_id, device)
        return cls._pipe

    @classmethod
    def _load(cls, model_id: str, device: str) -> None:
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        console.print(f"[cyan]  Loading SD model [bold]{model_id}[/bold] on [bold]{device}[/bold]…[/cyan]")
        console.print("  [dim](First run downloads ~4 GB — subsequent runs use cache)[/dim]")

        dtype = torch.float16 if device == "cuda" else torch.float32
        PipelineCls = StableDiffusionXLPipeline if "xl" in model_id.lower() else StableDiffusionPipeline

        pipe = PipelineCls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()   # saves VRAM

        cls._pipe = pipe
        cls._loaded_model = model_id
        cls._loaded_device = device
        console.print("  [green]Model loaded.[/green]")


def resolve_device(setting: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if setting != "auto":
        return setting
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def generate_image(
    prompt: str,
    output_path: str,
    model_id: str,
    device: str,
    width: int = 768,
    height: int = 512,
    steps: int = 25,
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, ugly, "
        "watermark, text, signature, extra limbs"
    ),
) -> str:
    """Generate one image and save it to output_path. Returns output_path."""
    pipe = _SDPipeline.get(model_id, device)

    console.print(f"  [dim]Rendering {Path(output_path).name} …[/dim]")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    )
    image = result.images[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
