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
  # Work around "ModuleNotFoundError: No module named 'torch'"
  # https://github.com/facebookresearch/xformers/issues/740#issuecomment-1780727152
  pip3 install --upgrade wheel

  pip3 install --upgrade \
    accelerate \
    compel \
    diffusers \
    omegaconf \
    peft \
    tiktoken \
    torch \
    torchvision \
    transformers

  # Qwen
  #pip3 install --upgrade \
  #  deepspeed einops flash-attn transformers_stream_generator

  pip3 install --upgrade --no-dependencies xformers

  # https://github.com/JamesQFreeman/LoRA-ViT
  #pip3 install --upgrade git+https://github.com/Passiolife/minLoRAplus@main
  #pip3 install --upgrade git+https://github.com/cccntu/minLoRA@main
}

general() {
  echo "Installing general packages"
  pip3 install --upgrade \
    Pillow \
    ftfy \
    immutabledict \
    numpy \
    sentencepiece \
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
    pyls \
    python-language-server
}

tensorflow() {
  echo "Installing tensorflow packages"
  # datasets is not really needed, was used when I was doing tutorials.
  pip3 install --upgrade \
    kaggle \
    pycocotools \
    tensorboard_plugin_profile \
    tensorflow \
    tensorflow-text \
    tensorflow_datasets

  pip3 install --upgrade \
    jax
  # tensorflow depends on keras 2. Duh. So we need to install keras after.
  pip3 install --upgrade \
    keras \
    keras-nlp \

}

intel() {
  echo "Installing Intel extension"
  pip3 install --upgrade intel-extension-for-pytorch
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

openinterpreter() {
  pip install --upgrade \
    open-interpreter \
    opencv-python
}

pip3 install --upgrade pip
if [ "$UNAME" = "Darwin" ]; then
  BREW="$(dirname $(dirname $(which brew)))"
  export PATH="$BREW/opt/llvm/bin:$PATH"
  export CC="$BREW/opt/llvm/bin/clang"
  export CXX="$BREW/opt/llvm/bin/clang++"
  diffusion
  jupyter
  lint
  general
  security
else
  diffusion
  general
  jupyter
  lint
  tensorflow
  intel
  #cuda
  security
  openinterpreter
fi

pip3 freeze > "requirements-$(uname).txt"
