#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"
source venv/bin/activate

UNAME=$(uname)

if [ "$UNAME" = "Linux" ]; then
  if which nvidia-smi > /dev/null; then
    OLD_PIDS="$(nvidia-smi | grep python3 | awk '{ print $5 }')"
    if [ "$OLD_PIDS" != "" ]; then
      echo "Killing stale python processes"
      echo $OLD_PIDS | xargs -n1 kill -9
    fi
  fi
fi

# Not used anymore, just in case.
# https://gradio.app/docs/#interface
export GRADIO_ANALYTICS_ENABLED=0

# Override Huggingface default ~/.cache/huggingface path to have it in here.
# See https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/__init__.py#L69
export HF_HOME=$PWD/cache/huggingface

export TF_CPP_MIN_LOG_LEVEL=2
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Override Jupyter directories to not use ~/.jupyter and ~/.local/share/jupyter
export JUPYTER_CONFIG_DIR=$PWD/config
export JUPYTER_DATA_DIR=$PWD/data
export JUPYTER_RUNTIME_DIR=$PWD/runtime

# --watch ?
# --autoreload ?
jupyter lab -y --no-browser --ip 0.0.0.0 --notebook-dir "$PWD/notebooks"
