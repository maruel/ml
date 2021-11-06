#!/bin/bash
# Copyright 2021 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

# sudo apt install gcc

if ! lspci | grep -i nvidia > /dev/null; then
  echo "No nvidia card found"
  exit 1
fi

if [ ! -f bin/activate ]; then
  echo "Setting up virtualenv"
  python3 -m venv .
fi

source bin/activate

# From:
#   pip install tensorflow Pillow
#   pip freeze > requirements.txt
pip install -q -r requirements.txt


if [ ! -d /usr/local/cuda/lib64/ ]; then
  echo "Visit https://gist.github.com/maruel/e99622298891cc856044e8c158a83fdd"
  exit 1
fi

if [ ! -f /usr/lib/x86_64-linux-gnu/libcudnn.so ]; then
  echo "Visit https://gist.github.com/maruel/e99622298891cc856044e8c158a83fdd"
  exit 1
fi
