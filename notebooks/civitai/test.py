#!/usr/bin/env python3
#
# Took inspiration from:
# https://medium.com/mlearning-ai/using-civitai-models-with-diffusers-package-45e0c475a67e

import os

# Silence tensorflow even if we don't use it here.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import diffusers
import torch
import transformers
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32


def get_pipe(model_path, clip_skip):
    """Returns a loaded ML pipeline.

    clip_skip follows community convention:
      clip_skip = 1 uses the all text encoder layers.
      clip_skip = 2 skips the last text encoder layer.
    """
    if clip_skip > 1:
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            text_encoder=transformers.CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder",
                num_hidden_layers=12 - (clip_skip - 1),
                torch_dtype=torch_dtype
            ),
        )
    else:
        pipe = diffusers.DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    pipe = pipe.to(device)
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)
    return pipe


def get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character=",",
):
    """Prompt embeddings to overcome CLIP 77 token limit.
    https://github.com/huggingface/diffusers/issues/2136
    """
    max_length = pipe.tokenizer.model_max_length
    # Simple method of checking if the prompt is longer than the negative
    # prompt - split the input strings using `split_character`.
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # If prompt is longer than negative prompt.
    if count_prompt >= count_negative_prompt:
        input_ids = pipe.tokenizer(
            prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt,
            truncation=False,
            padding="max_length",
            max_length=shape_max_length,
            return_tensors="pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
            max_length=shape_max_length,
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])
    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


def generate_images(pipe, out, prompt, negative_prompt):
    """Generate a set of images and save them to out."""
    # Number of inference steps.
    num_inference_steps = 20
    # Guidance scale.
    guidance_scale = 7
    # Image dimensions - limited to GPU memory.
    width = 768
    height = 512
    # Seed and batch size.
    start_idx = 0
    batch_size = 10
    seeds = [i for i in range(start_idx , start_idx + batch_size, 1)]

    # Set to True to use prompt embeddings for long prompts, and False to
    # use the prompt strings.
    use_prompt_embeddings = True
    if not use_prompt_embeddings:
        def generate_image(seed):
            return pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=torch.manual_seed(seed),
            ).images[0]
    else:
        prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
            pipe, prompt, negative_prompt)
        def generate_image(seed):
            return pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=torch.manual_seed(seed),
            ).images[0]


    images = []
    for seed in seeds:
        img = generate_image(seed)
        # Save them as we go.
        img.save(out % seed)
        images.append(img)
    return images

if not os.path.isdir("out"):
  os.mkdir("out")

imgs = generate_images(
    pipe=get_pipe(
        model_path="ChineseLandscapeArt_v10",
        clip_skip=1,
    ),
    out="out/temple_%05d.png",
    #prompt="beautiful Chinese Landscape Art, best quality, intricate, water colors, snowy mountains, glacier, snow, starry night sky, stars, milkyway",
    prompt="beautiful Chinese Landscape Art, best quality, intricate, water colors, temple, mountains, glacier, snow, starry night sky, stars, milkyway",
    negative_prompt="deformed, weird, bad resolution, bad depiction, not Chinese style, weird, has people, worst quality, ugly, worst resolution, too blurry, not relevant",
)

# Upscale later with https://huggingface.co/spaces/doevent/Face-Real-ESRGAN
