#!/usr/bin/env python3

"""Runs Stable Diffusion on a text prompt to generate an image.

See documentation at https://pypi.org/project/diffusers/

Look at https://stability.ai/sdv2-prompt-book for ideas!

# https://civitai.com/models/117761/tiny-home-concept
# https://civitai.com/api/download/models/125833
# https://civitai.com/models/133700/curly-hair-slider-lora
# https://medium.com/mlearning-ai/using-civitai-models-with-diffusers-package-45e0c475a67e
"""

import argparse
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import sdcommon

import diffusers
import torch


class StableDiffusionPipeline(diffusers.StableDiffusionPipeline):
  """Adds the support to accept a seed directly, instead of a generator."""
  def __call__(self, seed, **kwargs):
    if self._execution_device.type == "cuda":
      generator = torch.Generator("cuda").manual_seed(seed)
    else:
      generator = torch.Generator()
      generator.manual_seed(seed)
    return super(StableDiffusionPipeline, self).__call__(generator=generator, return_dict=False, **kwargs)


class ML(object):
    """Runs a generation job."""
    _diffusers = sdcommon.classesdict(StableDiffusionPipeline)
    _schedulers = sdcommon.classesdict(
        diffusers.DDIMScheduler,
        diffusers.DPMSolverMultistepScheduler,
        diffusers.EulerAncestralDiscreteScheduler,
        diffusers.EulerDiscreteScheduler,
        diffusers.LMSDiscreteScheduler,
        diffusers.PNDMScheduler,
    )

    def __init__(self,
                 diffusername="StableDiffusionPipeline",
                 schedulername="DPMSolverMultistepScheduler",
                 model="stabilityai/stable-diffusion-2-1",
                 #revision=None,
                 engine=sdcommon.getdefaultengine()):
        assert diffusername in self._diffusers, diffusername
        assert schedulername in self._schedulers, schedulername
        assert isinstance(model, str) and model, model
        #assert revision
        assert engine in ("cuda", "cpu", "mps"), engine
        self.algo = "txt2img"
        self.diffusername = diffusername
        self.schedulername = schedulername
        self.model = model
        #self.revision = revision or ("fp32" if engine == "cpu" else "fp16")
        self.engine = engine

    @sdcommon.autogc
    def run(self, params, **kwargs):
        """Automatically cleans up GPU memory after use."""
        return self.get(**kwargs)(**params.kwargs())

    def get(self, local_files_only=True, **kwargs):
        """Do not try to download by default"""
        # See https://huggingface.co/docs/diffusers/optimization/fp16
        # for more optimizations.
        # TODO(maruel): bf16 / tf32
        if self.engine in ("cpu", "mps"):
          dtype = torch.float32
        else:
          dtype = torch.float16
        pipe = self._diffusers[self.diffusername].from_pretrained(
            self.model,
            #revision=self.revision,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=local_files_only,
            **kwargs)
        pipe.scheduler = self._schedulers[self.schedulername].from_config(
            pipe.scheduler.config)
        if self.engine == "cuda":
          vram = torch.cuda.get_device_properties(0).total_memory
          if vram < 7*1024*1024*1024 and kwargs["width"]*kwargs["height"] > 1024*1024:
            # If the image is very large, generate tiles.
            # It's less nice due to boundary errors.
            # Automatically turned off when the image is 512x512 or smaller.
            pipe.enable_vae_tiling()
          if vram < 2*1024*1024*1024:
            # Reduce memory usage by loading the items at each steps.
            # Will slow down inference! Has to be done before pipe.to(engine).
            pipe.enable_sequential_cpu_offload()
          #pipe.enable_model_cpu_offload()
        pipe.to(self.engine)
        if self.engine == "cuda":
          # Memory efficient attention is only available on GPU.
          pipe.enable_xformers_memory_efficient_attention()
        else:
          # TODO(maruel): Get ram size.
          ram = 128*1024*1024*1024
          if ram < 20*1024*1024*1024:
            # If the image is very large, generate tiles.
            # It's less nice due to boundary errors.
            # Automatically turned off when the image is 512x512 or smaller.
            pipe.enable_vae_tiling()

        # Useful once we support mulitple prompts at once. Ask to do one prompt at a time.
        pipe.enable_vae_slicing()
        return pipe


class Params(object):
    """Text to Image Parameters as a class so it can be easily serialized."""
    def __init__(self, prompt, neg_prompt="", seed=1, steps=50, guidance=7.5, num_images=1,
                width=768, height=768):
        assert isinstance(prompt, str) and prompt
        assert isinstance(neg_prompt, str)
        assert isinstance(seed, int) and 1 <= seed <= 2147483647
        assert isinstance(steps, int) and 1 <= steps <= 1000
        assert isinstance(guidance, float) and 0. <= guidance <= 15.
        assert isinstance(num_images, int) and 1 <= num_images <= 1024
        assert isinstance(width, int) and 16 <= width <= 8192
        assert isinstance(height, int) and 16 <= height <= 8192
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.num_images = num_images
        self.width = width
        self.height = height

    def kwargs(self):
        return {
            "prompt": self.prompt,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "negative_prompt": self.neg_prompt,
            "num_images_per_prompt": self.num_images,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
        }


def run(model, prompt, steps, engine="cpu"):
  """Runs a single job."""
  # This modifies the global state. Only has effect on Ampere.
  if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

  ml = ML(model=model, engine=engine)
  params = Params(prompt=prompt, steps=steps)
  img, _ = ml.run(params, local_files_only=False)
  return ml, params, img


def main():
  parser = argparse.ArgumentParser(
      description=sys.modules[__name__].__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-p", "--prompt", help="prompt to generate", required=True)
  parser.add_argument("--out", help="png to write, defaults to next available out/txt2img00000.png")
  parser.add_argument(
      "--steps", default=25, type=int,
      help="steps to run, more the better; use low value for quick experiment")
  parser.add_argument(
      "--engine", choices=("cpu", "cuda", "mps"), default=sdcommon.getdefaultengine(),
      help="Processor (CPU/GPU) to use")
  parser.add_argument(
      "--model", default="stabilityai/stable-diffusion-2-1",
      help="Stable Diffusion model to use")
  # TODO(maruel): Size
  args = parser.parse_args()
  start = time.time()
  ml, params, img = run(
    model=args.model,
    prompt=args.prompt,
    steps=args.steps,
    engine=args.engine,
  )
  print("Took %.1fs" % (time.time()-start))
  if args.out:
    args.out = os.path.join(THIS_DIR, "out", args.prompt.replace(".", "") + ".png")
    data = {"ml": sdcommon.to_dict(ml), "params": sdcommon.to_dict(params)}
    base = args.out.rsplit(".", 2)[0]
    with open(base + ".json", "w") as f:
      json.dump(data, f, sort_keys=True, indent=2)
    img[0].save(args.out)
  else:
    print("Saved as", sdcommon.save(ml, params, img[0]))
  return 0


if __name__ == "__main__":
  sys.exit(main())
