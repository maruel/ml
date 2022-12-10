#!/usr/bin/env python3

"""Runs Stable Diffusion on an image."""

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
  # Not used here but will fail at runtime after downloading.
  import accelerate
  import diffusers
  # Not used here but will fail at runtime.
  import transformers
  import torch
except ImportError:
  print("Run: pip install accelerate diffusers transformers torch", file=sys.stderr)
  sys.exit(1)

def run(model, engine, prompt, steps, out):
  """Runs the txt2img algorithm.

  See documentation at https://pypi.org/project/diffusers/
  """
  # TODO(maruel): I'm getting really bad behavior on "mps" on a MBP M1 Pro with
  # 16GB of RAM. Failed in mps+float16, and mps+float32 swaps like crazy, while
  # cpu+float32 is fine.
  kwargs = {}
  if engine == "cuda":
    kwargs["revision"] == "fp16"
  pipe = diffusers.StableDiffusionPipeline.from_pretrained(
      model,
      torch_dtype=torch.float16 if engine == "cuda" else torch.float32,
      **kwargs)
  pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
      pipe.scheduler.config)
  pipe = pipe.to(engine)
  # Necessary with 6GB of VRAM.
  pipe.enable_attention_slicing()
  if engine == "mps":
    # Work around a temporary bug where first inference is broken according to
    # the internet. Remove in 2023?
    pipe(prompt, num_inference_steps=1)
  image = pipe(prompt, num_inference_steps=steps).images[0]
  if out:
    image.save(out)
  return image

def main():
  parser = argparse.ArgumentParser(
      description=sys.modules[__name__].__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-p", "--prompt", help="prompt to generate", required=True)
  parser.add_argument("--out", help="png to write, defaults to out/<prompt>.png")
  parser.add_argument(
      "--steps", default=25, type=int,
      help="steps to run, more the better; use low value for quick experiment")
  parser.add_argument(
      "--engine", choices=("cpu", "cuda", "mps"), default="cpu",
      help="Processor (CPU/GPU) to use")
  parser.add_argument(
      "--model", default="stabilityai/stable-diffusion-2-1",
      help="Stable Diffusion model to use"  )
  args = parser.parse_args()
  if not args.out:
    args.out = os.path.join(THIS_DIR, "out", args.prompt.replace(".", "") + ".png")
  run(args.model, args.engine, args.prompt, args.steps, args.out)

if __name__ == "__main__":
  main()
