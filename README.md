# Machine Learning toolkit

Includes:
- Notebooks for [Stable Diffusion
  v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) and [Stable LM
  2 1.6B](https://stability.ai/news/introducing-stable-lm-2) including
  ([txt2img, depth2img, stablelm](notebooks/stable_diffusion)).
- General examples in [notebooks/learning](notebooks/learning)
- Deepdream notebooks and scripts.
  [DeepDream](https://en.wikipedia.org/wiki/DeepDream) was really hot a few
  years back.


## Google Colab

- Visit https://colab.research.google.com/ or directly load one of the
  notebooks:
  - [txt2img](https://colab.research.google.com/github/maruel/ml/blob/main/notebooks/stable_diffusion/txt2img.ipynb)
  - [depth2img](https://colab.research.google.com/github/maruel/ml/blob/main/notebooks/stable_diffusion/depthimg.ipynb)
- Select Runtime / Select Runtime type
  - Choose TPU or GPU. One may be out of stock while the other is still
    available, depending on the time of the day.
    TODO: Test with TPU.
- Run the first cell to install everything (shift-enter)
  - It will ask for confirmation that you trust the code. Do as you wish. :)
- Run the second cell to generate stuff.


## Local

### Ubuntu 22.04

1. Install CUDA from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
    - You don't have to register.
1. You can install then confirm it runs:
    ```
    sudo apt install cuda-11-8 libcudnn8 tensorrt-libs
    python3 -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
    ```
1. Run `./setup.sh` to create the virtual environment and install pip packages
1. To start the server, run `./run.sh`
1. Tested on Ubuntu 22.04 with a Nvidia RTX 2060. Works great remotely via a Chromebook!


### Perf

List memory usage: `nvidia-smi`

Summary:

```
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
```

On exceptions, Jupyter tends to leave zombie python processes that will keep GPU
VRAM allocations. Kill with:

```
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill
```

### Windows 11

1. Get python3.11 from the Microsoft Store until
   https://github.com/pytorch/pytorch/issues/110436 is fixed and it becomes
   compatible with 3.12.
1. Get CUDA from
   https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network
    - You don't have to register.


## Random notes

https://huggingface.co/blog/lcm_lora
https://github.com/huggingface/diffusers/blob/main/examples/consistency_distillation/README_sdxl.md
https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/LCM-LoRA-Technical-Report.pdf
