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
    "chinese_landscape_art": common.Model(
        doc_url="https://civitai.com/models/120298/chinese-landscape-art",
        model_url="https://civitai.com/api/download/models/130803",
        filename="ChineseLandscapeArt_v10.safetensors",
        type="checkpoint",
    ),
    "sd21": common.Model(
        doc_url="stabilityai/stable-diffusion-2-1",
        model_url="todo",
        filename="foo.safetensors",
        type="checkpoint",
    ),
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
        #required=True,
        default="beautiful Chinese Landscape Art, best quality, intricate, water colors, temple, mountains, glacier, snow, starry night sky, stars, milkyway",
    )
    #default="beautiful Chinese Landscape Art, best quality, intricate, water colors, snowy mountains, glacier, snow, starry night sky, stars, milkyway",
    parser.add_argument(
        "-n", "--neg_prompt", help="negative prompt to improve image quality",
        default="deformed, weird, bad resolution, bad depiction, not Chinese style, weird, has people, worst quality, ugly, worst resolution, too blurry, not relevant",
    )
    parser.add_argument(
        "--model", choices=known_models.keys(),
        required=True,
        help="Stable Diffusion model to use")
    args = parser.parse_args()

    params = common.Params(prompt=args.prompt, neg_prompt=args.neg_prompt)
    imgs = generate_images(
        pipe=known_models[args.model].to_pipe(),
        out="out/temple4_%05d.png",
        params=params,
    )
    # TODO(maruel): Upscale later with https://huggingface.co/spaces/doevent/Face-Real-ESRGAN
    return 0


if __name__ == "__main__":
    sys.exit(main())
