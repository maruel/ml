#!/bin/bash
# Copyright 2021 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

UNAME=$(uname)

if ! which uv > /dev/null 2>&1; then
  echo "uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [ "$UNAME" = "Darwin" ]; then
  HAS_NVIDIA=0
  # Apple's llvm doesn't support -fopenmp.
  brew install llvm libomp
  BREW="$(dirname $(dirname $(which brew)))"
  export PATH="$BREW/opt/llvm/bin:$PATH"
  export CC="$BREW/opt/llvm/bin/clang"
  export CXX="$BREW/opt/llvm/bin/clang++"
else
  HAS_NVIDIA=1
  if ! lspci | grep -i nvidia > /dev/null; then
    echo "Warning: No nvidia card found"
    HAS_NVIDIA=0
  fi
fi

# Install all dependencies including lint group.
GROUPS="--group lint"

if [ "$UNAME" != "Darwin" ]; then
  if lspci | grep -i intel > /dev/null 2>&1; then
    GROUPS="$GROUPS --group intel"
  fi
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
    GROUPS="$GROUPS --group cuda"
  fi
fi

echo "Running: uv sync $GROUPS"
uv sync $GROUPS
