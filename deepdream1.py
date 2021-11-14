#!bin/python3
# Copyright 2021 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Includes tools to deep dream an image.

Inspired by https://www.tensorflow.org/tutorials/generative/deepdream
"""

import logging
import os
import subprocess
import sys

import deepdream


def main():
  logging.basicConfig(
      level=logging.DEBUG,
      format="%(relativeCreated)6d %(message)s")
  logging.info('Loaded')
  # 'https://pbs.twimg.com/profile_images/80041186/chicken.gif'
  dreams = deepdream.run_deep_dream_simple(
      img=deepdream.download("chicken.gif"),
      steps=1000,
      step_size=0.01,
      steps_per_output=10)

  imgs = [
      deepdream.PIL.Image.fromarray(deepdream.np.array(deepdream.denormalize(img)))
      for img in dreams
  ]
  if not os.path.isdir("out"):
    os.mkdir("out")
  imgs[0].save(
      os.path.join("out", "troubled_chicken.gif"),
      save_all=True,
      append_images=imgs[1:],
      duration=[1000] + ([40] * (len(imgs)-2)) + [1000],
      loop=0)
  return 0


if __name__ == "__main__":
  sys.exit(main())
