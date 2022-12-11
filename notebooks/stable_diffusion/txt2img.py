#!/usr/bin/env python3

"""Runs Stable Diffusion on a text prompt to generate an image.

See documentation at https://pypi.org/project/diffusers/

Look at https://stability.ai/sdv2-prompt-book for ideas!
"""

import argparse
import os
import sys

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
        return self.get(**kwargs)(**d2iargs.kwargs())

    def get(self, local_files_only=True, **kwargs):
        """Do not try to download by default"""
        pipe = self._diffusers[self.diffusername].from_pretrained(
            self.model,
            #revision=self.revision,
            torch_dtype=torch.float32 if self.engine == "cpu" else torch.float16,
            local_files_only=local_files_only,
            **kwargs)
        pipe.to(self.engine)
        pipe.scheduler = self._schedulers[self.schedulername].from_config(
            pipe.scheduler.config)
        if self.engine == "cuda":
            pipe.enable_attention_slicing()
        return pipe

class Params(object):
    def __init__(self, prompt, neg_prompt="", seed=1, steps=50, guidance=7.5, num_images=1):
        assert isinstance(prompt, str) and prompt
        assert isinstance(neg_prompt, str)
        assert isinstance(seed, int) and 1 <= seed <= 2147483647
        assert isinstance(steps, int) and 1 <= steps <= 1000
        assert isinstance(guidance, float) and 0. <= guidance <= 15.
        assert isinstance(num_images, int) and 1 <= num_images <= 1024
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.num_images = num_images

    def kwargs(self):
        return {
            "prompt": self.prompt,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "negative_prompt": self.neg_prompt,
            "num_images_per_prompt": self.num_images,
            # Converted into generator by _DepthBase.__call__().
            "seed": self.seed,
        }

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
      help="Stable Diffusion model to use"  )
  args = parser.parse_args()
  if not args.out:
    args.out = os.path.join(THIS_DIR, "out", args.prompt.replace(".", "") + ".png")

  ml = depth2img.ML(model=args.model, engine=args.engine)
  params = depth2img.Params(prompt=args.prompt, steps=args.steps)
  start = time.time()
  img, _ = ml.run(params) #, local_files_only=False)
  print("Took %.1fs" % (time.time()-start))
  if args.out:
    data = {"ml": to_dict(ml), "params": to_dict(params)}
    base = args.out.rsplit(".", 2)[0]
    with open(base + ".json", "w") as f:
      json.dump(data, f, sort_keys=True, indent=2)
    img[0].save(args.out)
  else:
    print("Saved as", sdcommon.save(ml, params, img[0]))
  return 0

if __name__ == "__main__":
  sys.exit(main())
