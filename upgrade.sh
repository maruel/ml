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
    transformers \
    xformers

  # Told this help performance:
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
    ipyplot \
    ipympl \
    jupyter \
    jupyterlab \
    jupyterlab-lsp \
    matplotlib \
    python-language-server
}

tensorflow() {
  echo "Installing tensorflow packages"
  pip3 install --upgrade \
    kaggle \
    pycocotools \
    tensorboard_plugin_profile \
    tensorflow \
    tensorflow_datasets
}

cuda() {
  echo "Installing nvidia/CUDA packages"
  # TODO(maruel): Currently incompatible with jupyterlab 4.
  pip3 install --upgrade jupyterlab-nvdashboard
}

pip install --upgrade pip
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
