#!/usr/bin/env python3
#
# Took inspiration from:
# https://medium.com/mlearning-ai/using-civitai-models-with-diffusers-package-45e0c475a67e
#
# https://huggingface.co/blog/lora
#
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py
#
# https://github.com/kohya-ss/sd-scripts

"""Generates images using a civitai model."""

import argparse
import os
import sys

import torch

import common


known_models = {
    "chinese_landscape_art": common.CivitaiCheckpointModel(
        doc_url="https://civitai.com/models/120298/chinese-landscape-art",
        model_url="https://civitai.com/api/download/models/130803",
        filename="ChineseLandscapeArt_v10.safetensors",
        keywords=("beautiful Chinese Landscape Art", "Chinese Landscape Art", "water colors"),
        base_model=None,
        # I'm not sure why it's not necessary. I suspect that since the file is
        # huge, it is embedded.
        #base_model="runwayml/stable-diffusion-v1-5",
        clip_skip=1,
    ),
    "curly_hair": common.CivitaiLoRAModel(
      doc_url="https://civitai.com/models/133700/curly-hair-slider-lora",
      model_url="https://civitai.com/api/download/models/147196",
      filename="curly_hair_slider_v1.safetensors",
      keywords=("curly hair", "straight hair"),
      base_model="runwayml/stable-diffusion-v1-5",
      clip_skip=0,
    ),
    "tiny_home": common.CivitaiLoRAModel(
      doc_url="https://civitai.com/models/117761/tiny-home-concept",
      model_url="https://civitai.com/api/download/models/128150",
      filename="ARWinteriortinyhouse.safetensors",
      keywords=("interiortinyhouse",),
      base_model="runwayml/stable-diffusion-v1-5",
      clip_skip=2,
    ),
    #"sd21": common.HuggingFaceModel(
    #    model="stabilityai/stable-diffusion-2-1",
    #),
}


def generate_images(pipe, out, params):
    """Generate a set of images and save them to out."""
    kwargs = params.kwargs()
    seeds = kwargs.pop("seeds")
    # Set to True to use prompt embeddings for long prompts, and False to
    # use the prompt strings.
    use_prompt_embeddings = True
    if not use_prompt_embeddings:
        def generate_image(seed):
            return pipe(
                generator=torch.manual_seed(seed),
                num_images_per_prompt=1,
                **kwargs).images[0]
    else:
        prompt_embeds, negative_prompt_embeds = common.get_prompt_embeddings(
            pipe, kwargs.pop("prompt"), kwargs.pop("negative_prompt"))
        def generate_image(seed):
            return pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                generator=torch.manual_seed(seed),
                num_images_per_prompt=1,
                **kwargs).images[0]

    images = []
    for seed in seeds:
        img = generate_image(seed)
        # Save them as we go.
        img.save(out % seed)
        images.append(img)
    return images



def main():
    if not os.path.isdir("out"):
      os.mkdir("out")

    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p", "--prompt", help="prompt to generate",
        required=True,
    )
    parser.add_argument(
        "-n", "--neg_prompt", help="negative prompt to improve image quality",
        required=True,
    )
    parser.add_argument(
        "--model", choices=known_models.keys(),
        required=True,
        help="Stable Diffusion model to use")
    args = parser.parse_args()

    #default="beautiful Chinese Landscape Art, best quality, intricate, water colors, temple, mountains, glacier, snow, starry night sky, stars, milkyway",
    #default="beautiful Chinese Landscape Art, best quality, intricate, water colors, snowy mountains, glacier, snow, starry night sky, stars, milkyway",
    #default="deformed, weird, bad resolution, bad depiction, not Chinese style, weird, has people, worst quality, ugly, worst resolution, too blurry, not relevant",

    params = common.Params(prompt=args.prompt, neg_prompt=args.neg_prompt)
    imgs = generate_images(
        pipe=known_models[args.model].to_pipe(),
        out="out/sd_%05d.png",
        params=params,
    )
    # TODO(maruel): Upscale later with https://huggingface.co/spaces/doevent/Face-Real-ESRGAN
    return 0


if __name__ == "__main__":
    sys.exit(main())
