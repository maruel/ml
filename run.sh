#!/bin/bash
# Copyright 2022 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu
cd "$(dirname $0)"

source bin/activate

OLD_PIDS="$(nvidia-smi | grep python3 | awk '{ print $5 }')"
if [ "$OLD_PIDS" != "" ]; then
  echo "Killing stale python processes"
  echo $OLD_PIDS | xargs -n1 kill -9
fi

# https://gradio.app/docs/#interface
export GRADIO_ANALYTICS_ENABLED=0

export TF_CPP_MIN_LOG_LEVEL=2
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
# --watch ?
# --autoreload ?
jupyter lab -y --no-browser --ip 0.0.0.0 --notebook-dir notebooks
