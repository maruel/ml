#!/usr/bin/env python3

"""Runs Stable Diffusion on an image."""

import argparse
import os
import sys

try:
  import diffusers
  import torch
except ImportError:
  print("Run: pip install --user diffusers", file=sys.stderr)
  sys.exit(1)

def run(model, engine, prompt, out):
  pipe = diffusers.StableDiffusionPipeline.from_pretrained(
      model, torch_dtype=torch.float16)
  pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
      pipe.scheduler.config)
  pipe = pipe.to(engine)
  # Necessary with 6GB of VRAM.
  pipe.enable_attention_slicing()
  image = pipe(prompt).images[0]
  if out:
    image.save(out)
  return image

def main():
  parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
  parser.add_argument("out", help="png to write", required=True)
  parser.add_argument("prompt", help="prompt to generate", required=True)
  parser.add_argument("engine", choices=("cpu", "cuda", "mps"), default="cpu")
  parser.add_argument("model", default="stabilityai/stable-diffusion-2-1")
  args = parser.parse_args()
  run(args.model, args.engine, args.prompt, args.out)

if __name__ == "__main__":
  main()
