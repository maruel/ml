#!/usr/bin/env python3

"""Runs Stable Diffusion on an image and a text prompt to generate an image.

Uses Depth-to-Image Stable Diffusion Model as described at
https://stability.ai/blog/stable-diffusion-v2-release

See documentation at https://pypi.org/project/diffusers/

Look at https://stability.ai/sdv2-prompt-book for ideas!
"""

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import sdcommon

import PIL
import diffusers
import numpy as np
import torch

class StableDiffusionPipeline(diffusers.StableDiffusionDepth2ImgPipeline):
    """Adds the support to accept a seed directly, instead of a generator."""
    # To be overriden by subclasses.
    depth_map = None

    def __call__(self, seed, **kwargs):
        """Also return the depth_map saved by subclass."""
        # pytorch device:
        if self._execution_device.type == "cuda":
            generator = torch.Generator("cuda").manual_seed(seed)
        else:
            generator = torch.Generator()
            generator.manual_seed(seed)
        out = super(StableDiffusionPipeline, self).__call__(generator=generator, return_dict=True, **kwargs)
        return out[0], self.depth_map

class Depth(StableDiffusionPipeline):
    def prepare_depth_map(self, image, depth_map, batch_size, do_classifier_free_guidance, dtype, device):
        """Same as original but saves the depth map.

        https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py#L367
        """
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            width, height = map(lambda dim: dim - dim % 32, (width, height))  # resize to integer multiple of 32
            image = image.resize((width, height), resample=diffusers.utils.PIL_INTERPOLATION["lanczos"])
            width, height = image.size
        else:
            image = [img for img in image]
            width, height = image[0].shape[-2:]

        if depth_map is None:
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=device)
            # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
            # So we use `torch.autocast` here for half precision inference.
            context_manger = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
            with context_manger:
                depth_map = self.depth_estimator(pixel_values).predicted_depth
        else:
            depth_map = depth_map.to(device=device, dtype=dtype)

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="bicubic",
            align_corners=False,
        )

        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
        # MARUEL: New
        self.depth_map = depth_map
        depth_map = depth_map.to(dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if depth_map.shape[0] < batch_size:
            depth_map = depth_map.repeat(batch_size, 1, 1, 1)

        depth_map = torch.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map
        return depth_map

class ReverseDepth(StableDiffusionPipeline):
    def prepare_depth_map(self, image, depth_map, batch_size, do_classifier_free_guidance, dtype, device):
        """Reverse the output of the function, so that attention is about everything except the important part.

        https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py#L367
        """
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
            width, height = map(lambda dim: dim - dim % 32, (width, height))  # resize to integer multiple of 32
            image = image.resize((width, height), resample=diffusers.utils.PIL_INTERPOLATION["lanczos"])
            width, height = image.size
        else:
            image = [img for img in image]
            width, height = image[0].shape[-2:]

        if depth_map is None:
            pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=device)
            # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
            # So we use `torch.autocast` here for half precision inference.
            context_manger = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
            with context_manger:
                depth_map = self.depth_estimator(pixel_values).predicted_depth
        else:
            depth_map = depth_map.to(device=device, dtype=dtype)

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="bicubic",
            align_corners=False,
        )

        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        # MARUEL: Original
        #depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
        # MARUEL: New
        depth_map = -2.0 * (depth_map - depth_min) / (depth_max - depth_min) + 1.0
        self.depth_map = depth_map
        depth_map = depth_map.to(dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if depth_map.shape[0] < batch_size:
            depth_map = depth_map.repeat(batch_size, 1, 1, 1)

        depth_map = torch.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map
        return depth_map

class ML(object):
    _diffusers = sdcommon.classesdict(Depth, ReverseDepth)
    _schedulers = sdcommon.classesdict(
        diffusers.DDIMScheduler,
        diffusers.DPMSolverMultistepScheduler,
        diffusers.EulerAncestralDiscreteScheduler,
        diffusers.EulerDiscreteScheduler,
        diffusers.LMSDiscreteScheduler,
        diffusers.PNDMScheduler,
    )

    def __init__(self,
                 diffusername="ReverseDepth",
                 schedulername="DPMSolverMultistepScheduler",
                 model="stabilityai/stable-diffusion-2-depth",
                 #revision=None,
                 engine=sdcommon.getdefaultengine()):
        assert diffusername in self._diffusers, diffusername
        assert schedulername in self._schedulers, schedulername
        assert isinstance(model, str) and model, model
        #assert revision
        assert engine in ("cuda", "cpu", "mps"), engine
        self.algo = "depth2img"
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
    def __init__(self, imagename, prompt, neg_prompt="", seed=1, steps=50,
                 strength=0.6, guidance=7.5, num_images=1):
        assert isinstance(imagename, str) and imagename
        assert isinstance(prompt, str) and prompt
        assert isinstance(neg_prompt, str)
        assert isinstance(seed, int) and 1 <= seed <= 2147483647
        assert isinstance(steps, int) and 1 <= steps <= 1000
        assert isinstance(strength, float) and 0. <= strength <= 1.
        assert isinstance(guidance, float) and 0. <= guidance <= 15.
        assert isinstance(num_images, int) and 1 <= num_images <= 1024
        self.imagename = imagename
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.seed = seed
        self.steps = steps
        self.strength = strength
        self.guidance = guidance
        self.num_images = num_images

    def kwargs(self):
        return {
            "prompt": self.prompt,
            "image": sdcommon.getimg(self.imagename),
            "strength": self.strength,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "negative_prompt": self.neg_prompt,
            "num_images_per_prompt": self.num_images,
            # Converted into generator by <ml>.__call__().
            "seed": self.seed,
        }

def main():
  parser = argparse.ArgumentParser(
      description=sys.modules[__name__].__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-src", help="source image to alter", required=True)
  parser.add_argument("-p", "--prompt", help="prompt to generate", required=True)
  parser.add_argument("--out", help="png to write, defaults to next available out/depth2img00000.png")
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
