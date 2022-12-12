# Machine Learning toolkit

Includes:
- [Stable Diffusion
  v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) notebooks
  ([txt2img, depth2img](notebooks/stable_diffusion))
- General examples in [notebooks/learning](notebooks/learning)
- Deepdream notebooks and scripts.
  [DeepDream](https://en.wikipedia.org/wiki/DeepDream) was really hot a few
  years back.


## Local

- To get started, run `./setup.sh`
- To start the server, run `./run.sh`
- Tested on Ubuntu 22.04 with a Nvidia RTX 2060. Works great remotely via a Chromebook!


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


## Perf

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
