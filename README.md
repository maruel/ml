# Machine Learning toolkit

Includes examples in [notebooks/learning](notebooks/learning)

Includes deepdream notebooks and scripts.
[DeepDream](https://en.wikipedia.org/wiki/DeepDream) was really hot a few years
back.

To get started, run `./setup.sh`

To start the server, run `./run.sh`

Tested on Ubuntu 20.04 with a Nvidia RTX 2060 and a cheap Chromebook.

## Perf

List memory usage: `nvidia-smi`

Summary:

```
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
```
