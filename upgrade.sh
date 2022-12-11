#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

source bin/activate

UNAME=$(uname)

diffusion() {
  echo "Installing stable diffusion packages"
  pip3 install --upgrade \
    accelerate \
    diffusers \
    torch \
    torchvision \
    transformers

  # Told this help performance, but no official release since September?
  #   pip install --upgrade git+https://github.com/facebookresearch/xformers@main
  #   pip install xformers
}

general() {
  echo "Installing general packages"
  pip3 install --upgrade \
    Pillow \
    ftfy \
    scipy
  #  triton
}

jupyter() {
  echo "Installing jupyter packages"
  pip3 install --upgrade \
    ipympl \
    jupyter jupyterlab \
    matplotlib
}

tensorflow() {
  echo "Installing tensorflow packages"
  pip3 install --upgrade \
    kaggle \
    tensorboard_plugin_profile \
    tensorflow
}

cuda() {
  echo "Installing nvidia/CUDA packages"
  pip3 install --upgrade jupyterlab-nvdashboard
}

if [ "$UNAME" = "Darwin" ]; then
  diffusion
else
  diffusion
  general
  jupyter
  tensorflow
  cuda
fi

pip3 freeze > requirements.txt
