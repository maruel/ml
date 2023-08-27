#!/bin/bash

set -eu

if [ ! -f ChineseLandscapeArt_v10.safetensors ]; then
  # https://civitai.com/user/Celsia/models
  # beautiful landscape scenes in the style of Chinese watercolour paintings.
  wget https://civitai.com/api/download/models/130803 --content-disposition
fi

if [ ! -f convert_original_stable_diffusion_to_diffusers.py ]; then
  wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py
fi

export TF_CPP_MIN_LOG_LEVEL=2

# Requires pip3 install omegaconf
# This script will break:
# FutureWarning: The class CLIPFeatureExtractor is deprecated and will be
# removed in version 5 of Transformers. Please use CLIPImageProcessor instead.
python convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path ChineseLandscapeArt_v10.safetensors \
    --dump_path ChineseLandscapeArt_v10/ \
    --from_safetensors
