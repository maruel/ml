#!/bin/bash
# Copyright 2021 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

# I think gcc is required but not sure. I had it preinstalled for other reasons.
# If you face challenges, install with: sudo apt install gcc

HAS_NVIDIA=1
if ! lspci | grep -i nvidia > /dev/null; then
  echo "Warning: No nvidia card found"
  # Continue on, it still runs at a bearable speed on CPU.
  HAS_NVIDIA=0
fi

if [ ! -f bin/activate ]; then
  echo "Setting up virtualenv"
  #python3 -m venv .
  virtualenv .
fi

source bin/activate

# sudo apt install build-essential libssl-dev python3-dev
# ./upgrade.sh ?
pip3 install -q -r requirements.txt

if [ "$HAS_NVIDIA" == "1" ]; then
  if [ ! -d /usr/local/cuda/lib64/ ]; then
    echo "CUDA is not properly installed."
    echo "Visit https://gist.github.com/maruel/e99622298891cc856044e8c158a83fdd"
    exit 1
  fi

  if [ ! -f /usr/lib/x86_64-linux-gnu/libcudnn.so ]; then
    echo "cuDNN is not properly installed."
    echo "Visit https://gist.github.com/maruel/e99622298891cc856044e8c158a83fdd"
    exit 1
  fi
fi
