#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

source bin/activate
pip3 install --upgrade \
  Pillow \
  accelerate \
  diffusers \
  ftfy \
  gradio \
  ipympl \
  jupyter jupyterlab \
  jupyterlab-nvdashboard \
  kaggle \
  matplotlib \
  scipy \
  tensorboard_plugin_profile \
  tensorflow \
  transformers \
  triton

# https://colab.research.google.com/github/qunash/stable-diffusion-2-gui/blob/main/stable_diffusion_2_0.ipynb#scrollTo=78HoqRAB-cES
# diffusers is extremely active but is released often. Alternative:
#   pip install --upgrade git+https://github.com/huggingface/diffusers.git@main
# transformers is also very active.
#   pip install --upgrade git+https://github.com/huggingface/transformers/
#   pip install https://github.com/metrolobo/xformers_wheels/releases/download/1d31a3ac_various_6/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl
#   pip install xformers

# Note that gradio seems not great, it creates http server and contacts an
# external web site. Better to not use it.

pip3 freeze > requirements.txt
