@echo off
:: Copyright 2024 Marc-Antoine Ruel. All rights reserved.
:: Use of this source code is governed under the Apache License, Version 2.0
:: that can be found in the LICENSE file.

:: Bypass "Terminate Batch Job" prompt.
IF "%~1"=="-FIXED_CTRL_C" (
   :: Remove the -FIXED_CTRL_C parameter.
   SHIFT
) ELSE (
   :: Run the batch with <NUL and -FIXED_CTRL_C
   CALL <NUL %0 -FIXED_CTRL_C %*
   GOTO :EOF
)

CALL venv\Scripts\activate.bat

:: Not used anymore, just in case.
:: https://gradio.app/docs/#interface
SET GRADIO_ANALYTICS_ENABLED=0

:: Override Huggingface default ~/.cache/huggingface path to have it in here.
:: See https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/__init__.py#L69
SET HF_HOME=%CD%\cache\huggingface

SET TF_CPP_MIN_LOG_LEVEL=2
:: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
SET TF_FORCE_GPU_ALLOW_GROWTH=true
:: --watch ?
:: --autoreload ?
jupyter lab -y --no-browser --ip 0.0.0.0 --notebook-dir notebooks
