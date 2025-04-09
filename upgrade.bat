@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.
setlocal enableextensions

call venv\Scripts\activate.bat


echo - Updating pip
python -m pip install --upgrade pip
echo.


echo - Updating setuptools and whell
:: Work around "ModuleNotFoundError: No module named 'torch'"
:: https://github.com/facebookresearch/xformers/issues/740#issuecomment-1780727152
pip3 install --upgrade setuptools wheel
echo.


echo - Installing pytorch with nvidia/CUDA packages
::pip3 install nvidia-pyindex
::pip3 install nvidia-cuda-runtime-cu12
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

::echo Installing xformers without dependencies
::pip3 install --upgrade --no-dependencies xformers


:: https://github.com/JamesQFreeman/LoRA-ViT
:: pip3 install --upgrade git+https://github.com/Passiolife/minLoRAplus@main
:: pip3 install --upgrade git+https://github.com/cccntu/minLoRA@main


::echo - Installing general packages
set PACKAGES=%PACKAGES% ^
    Pillow ^
    ftfy ^
    immutabledict ^
    numpy ^
    sentencepiece ^
    scipy


::echo - Installing jupyter packages
set PACKAGES=%PACKAGES% ^
    arxiv ^
    ipyplot ^
    ipympl ^
    jupyter ^
    jupyterlab ^
    jupyter-ai ^
    jupyterlab-lsp ^
    langchain-anthropic ^
    langchain-google-genai ^
    langchain-ollama ^
    matplotlib ^
    notebook ^
    python-lsp-server

::echo Installing stable diffusion packages
set PACKAGES=%PACKAGES% ^
    accelerate ^
    compel ^
    diffusers ^
    einops ^
    numpy ^
    omegaconf ^
    peft ^
    sentencepiece ^
    starlette ^
    tiktoken ^
    transformers




::echo Installing tensorflow packages
::pip3 install --upgrade ^
::    kaggle ^
::    pycocotools ^
::    tensorboard_plugin_profile ^
::    tensorflow ^
::    tensorflow_datasets

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
::echo Security updates
::set PACKAGES=%PACKAGES% ^
::    IPython ^
::    Werkzeug ^
::    aiohttp ^
::    certifi ^
::    cryptography ^
::    grpcio ^
::    markdown-it-py ^
::    parso ^
::    pyls ^
::    pygments ^
::    requests ^
::    starlette ^
::    tornado ^
    urllib3

pip3 install --upgrade %PACKAGES%

pip3 freeze > "requirements-Windows.txt"
