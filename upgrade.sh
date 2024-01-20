#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"
if [ ! -f venv/bin/activate ]; then
  echo "Run ./setup.sh first"
  exit 1
fi
source venv/bin/activate

UNAME=$(uname)

diffusion() {
  echo "Installing stable diffusion packages"
  pip3 install --upgrade \
    accelerate \
    diffusers \
    numpy \
    tiktoken \
    omegaconf \
    torch \
    torchvision \
    transformers

  # Work around "ModuleNotFoundError: No module named 'torch'"
  # https://github.com/facebookresearch/xformers/issues/740#issuecomment-1780727152
  pip3 install --upgrade wheel
  pip3 install --upgrade --no-dependencies xformers

  # https://github.com/JamesQFreeman/LoRA-ViT
  #pip3 install --upgrade git+https://github.com/Passiolife/minLoRAplus@main
  pip3 install --upgrade git+https://github.com/cccntu/minLoRA@main
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

lint() {
  # failed to install: rope_completion, rope_rename
  pip3 install --upgrade \
    autopep8 \
    mccabe \
    pycodestyle \
    pydocstyle \
    pyflakes \
    pylint \
    yapf
}

# Manually upgrade dependencies that are known to have security issues.
security() {
  pip3 install --upgrade \
    IPython \
    Werkzeug \
    aiohttp \
    certifi \
    cryptography \
    grpcio \
    markdown-it-py \
    pygments \
    requests \
    starlette \
    tornado \
    urllib3
}

pip3 install --upgrade pip
if [ "$UNAME" = "Darwin" ]; then
  diffusion
  jupyter
  lint
  security
else
  diffusion
  general
  jupyter
  lint
  tensorflow
  #cuda
  security
fi

pip3 freeze > "requirements-$(uname).txt"
