@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.

call venv\Scripts\activate.bat

echo Installing base tools
python -m pip install --upgrade pip
:: Work around "ModuleNotFoundError: No module named 'torch'"
:: https://github.com/facebookresearch/xformers/issues/740#issuecomment-1780727152
pip3 install --upgrade setuptools wheel

echo Installing pytorch with nvidia/CUDA packages
::pip3 install nvidia-pyindex
::pip3 install nvidia-cuda-runtime-cu12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing stable diffusion packages
pip3 install --upgrade ^
    accelerate ^
    diffusers ^
    numpy ^
    tiktoken ^
    omegaconf ^
    transformers

pip3 install --upgrade --no-dependencies xformers

pip3 install --upgrade einops

:: https://github.com/JamesQFreeman/LoRA-ViT
:: pip3 install --upgrade git+https://github.com/Passiolife/minLoRAplus@main
:: pip3 install --upgrade git+https://github.com/cccntu/minLoRA@main

echo Installing general packages
pip3 install --upgrade ^
    Pillow ^
    ftfy ^
    scipy
::  triton

echo Installing jupyter packages
pip3 install --upgrade ^
    ipyplot ^
    ipympl ^
    jupyter ^
    jupyterlab ^
    jupyterlab-lsp ^
    matplotlib ^
    python-language-server

echo Installing tensorflow packages
pip3 install --upgrade ^
    kaggle ^
    pycocotools ^
    tensorboard_plugin_profile ^
    tensorflow ^
    tensorflow_datasets

::echo Installing Intel extension
:: This version doesn't work on Windows, only on WSL2. (!!)
::pip3 install --upgrade intel-extension-for-pytorch
:: If you have an Intel Arc video card. Untested. This version does work on Windows. (!!)
:: pip3 install --upgrade intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

::echo Installing nvidia/CUDA packages
:: TODO(maruel): Currently incompatible with jupyterlab 4.
::pip3 install --upgrade jupyterlab-nvdashboard

: The whole thing doesn't seem to work well. I even got
:: parso to state that it doesn't support python 3.11, which is already
:: quite old. Surprisingly, removing these didn't remove the exception.
:: failed to install: rope_completion, rope_rename
::echo Installing linters
::pip3 install --upgrade ^
::    autopep8 ^
::    mccabe ^
::    pycodestyle ^
::    pydocstyle ^
::    pyflakes ^
::    pylint ^
::    yapf

:: Manually upgrade dependencies that are known to have security issues.
echo Security updates
pip3 install --upgrade ^
    IPython ^
    Werkzeug ^
    aiohttp ^
    certifi ^
    cryptography ^
    grpcio ^
    markdown-it-py ^
    pygments ^
    requests ^
    starlette ^
    tornado ^
    urllib3

pip3 freeze > "requirements-Windows.txt"
