"""
Text-to-3D generation using Shap-E (OpenAI).

Usage:
    python generate_3d.py --prompt "a red wooden chair" --output-dir outputs
    python generate_3d.py --prompt "a ceramic mug" --gif --obj --guidance-scale 15.0
"""

import argparse
import os
import torch
import imageio
import numpy as np
from pathlib import Path

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device: torch.device):
    print("Loading Shap-E models (first run downloads ~1.5 GB) …")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    return xm, model, diffusion


def generate_latents(
    prompt: str,
    model,
    diffusion,
    device: torch.device,
    batch_size: int = 1,
    guidance_scale: float = 15.0,
) -> list:
    print(f'Generating latents for: "{prompt}"')
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs={"texts": [prompt] * batch_size},
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    return latents


def export_gif(latents, xm, device: torch.device, output_path: Path, size: int = 128):
    print(f"Rendering GIF → {output_path}")
    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode="nerf")
        path = output_path if len(latents) == 1 else output_path.with_stem(f"{output_path.stem}_{i}")
        imageio.mimsave(
            str(path),
            [np.array(img) for img in images],
            fps=8,
            loop=0,
        )
        print(f"  Saved: {path}")


def export_obj(latents, xm, output_path: Path):
    print(f"Exporting OBJ mesh → {output_path}")
    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        path = output_path if len(latents) == 1 else output_path.with_stem(f"{output_path.stem}_{i}")
        with open(path, "w") as f:
            t.write_obj(f)
        print(f"  Saved: {path}")


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-to-3D via Shap-E")
    parser.add_argument("--prompt", required=True, help="Text description of the 3D object")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated files")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Classifier-free guidance scale")
    parser.add_argument("--gif-size", type=int, default=128, help="Render resolution for GIF frames")
    parser.add_argument("--gif", action="store_true", default=True, help="Export rotating GIF (default: on)")
    parser.add_argument("--no-gif", dest="gif", action="store_false")
    parser.add_argument("--obj", action="store_true", default=True, help="Export .obj mesh (default: on)")
    parser.add_argument("--no-obj", dest="obj", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xm, model, diffusion = load_models(device)
    latents = generate_latents(args.prompt, model, diffusion, device, args.batch_size, args.guidance_scale)

    name = slug(args.prompt)

    if args.gif:
        export_gif(latents, xm, device, out_dir / f"{name}.gif", size=args.gif_size)

    if args.obj:
        export_obj(latents, xm, out_dir / f"{name}.obj")

    print("\nDone.")


if __name__ == "__main__":
    main()
