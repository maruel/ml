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

# Told this help performances, but no official release since September?
#   pip install --upgrade git+https://github.com/facebookresearch/xformers@main
#   pip install xformers

# Note that gradio seems not great, it creates http server and contacts an
# external web site. Better to not use it.
# pip3 install --upgrade gradio

pip3 freeze > requirements.txt
