"""Text-to-image on CPU: SD1.5-class model + LCM-LoRA (4-6 steps instead of 30+)."""
import os

import torch
from PIL import Image, ImageFilter

_pipe = None


def load_pipe(cfg):
    global _pipe
    if _pipe is not None:
        return _pipe
    from diffusers import AutoPipelineForText2Image, LCMScheduler

    torch.set_num_threads(os.cpu_count() or 4)
    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(cfg["lcm_lora"])
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    return pipe


def generate(cfg, p):
    pipe = load_pipe(cfg)
    gen = torch.Generator("cpu").manual_seed(p["seed"])
    with torch.inference_mode():
        img = pipe(
            prompt=p["prompt"], negative_prompt=p["negative"],
            num_inference_steps=p["steps"], guidance_scale=cfg["guidance"],
            width=p["width"], height=p["height"], generator=gen,
        ).images[0]
    if p["hd"]:  # cheap 2x: Lanczos + mild sharpening
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=90, threshold=2))
    return img
