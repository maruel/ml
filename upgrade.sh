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
  echo "- Installing wheel"
  # Work around "ModuleNotFoundError: No module named 'torch'"
  # https://github.com/facebookresearch/xformers/issues/740#issuecomment-1780727152
  pip3 install --upgrade wheel
  echo ""

  echo "- Installing stable diffusion packages"
  pip3 install --upgrade \
    accelerate \
    compel \
    diffusers \
    omegaconf \
    peft \
	starlette \
    tiktoken \
    torch \
    torchvision \
    transformers
  echo ""

  # Qwen
  #pip3 install --upgrade \
  #  deepspeed einops flash-attn transformers_stream_generator

  if [ "$UNAME" = "Linux" ]; then
    echo "- Installing xformers"
    pip3 install --upgrade --no-dependencies xformers
    echo ""
  fi

  # https://github.com/JamesQFreeman/LoRA-ViT
  #pip3 install --upgrade git+https://github.com/Passiolife/minLoRAplus@main
  #pip3 install --upgrade git+https://github.com/cccntu/minLoRA@main
}

general() {
  echo "- Installing general packages"
  pip3 install --upgrade \
    Pillow \
    ftfy \
    immutabledict \
    numpy \
    sentencepiece \
    scipy
  #  triton
  echo ""
}

jupyter() {
  echo "- Installing jupyter packages"
  # See https://jupyter-ai.readthedocs.io/en/latest/users/index.html#model-providers
  pip3 install --upgrade \
    ipyplot \
    ipympl \
    jupyter \
    jupyterlab \
    jupyter-ai \
    jupyterlab-lsp \
    langchain-anthropic \
    langchain-google-genai \
    langchain-ollama \
    matplotlib \
    notebook \
    python-lsp-server
  echo ""
}

tensorflow() {
  echo "- Installing tensorflow packages"
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
  echo ""
}

intel() {
  echo "- Installing Intel extension"
  pip3 install --upgrade intel-extension-for-pytorch
  echo ""
}

cuda() {
  echo "- Installing nvidia/CUDA packages"
  # TODO(maruel): Currently incompatible with jupyterlab 4.
  pip3 install --upgrade jupyterlab-nvdashboard
  echo ""
}

lint() {
  echo "- Installing linters"
  # failed to install: rope_completion, rope_rename
  pip3 install --upgrade \
    autopep8 \
    mccabe \
    pycodestyle \
    pydocstyle \
    pyflakes \
    pylint \
    yapf
  echo ""
}

# Manually upgrade dependencies that are known to have security issues.
security() {
  echo "- Installing security updates"
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
  echo ""
}

openinterpreter() {
  pip install --upgrade \
    open-interpreter \
    opencv-python
  echo ""
}

pip3 install --upgrade pip
if [ "$UNAME" = "Darwin" ]; then
  BREW="$(dirname $(dirname $(which brew)))"
  export PATH="$BREW/opt/llvm/bin:$PATH"
  export CC="$BREW/opt/llvm/bin/clang"
  export CXX="$BREW/opt/llvm/bin/clang++"
  diffusion
  general
  jupyter
  lint
  # security
else
  diffusion
  general
  jupyter
  lint
  # tensorflow
  intel
  # openinterpreter
  if ! lspci | grep -i nvidia > /dev/null; then
	echo "- no cuda found"
  else
	cuda
  fi
  # security
fi

echo "- Updating requirements-$(uname).txt"
pip3 freeze > "requirements-$(uname).txt"
