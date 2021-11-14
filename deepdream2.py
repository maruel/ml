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
from deepdream import np, tf


def random_roll(img, maxroll):
  """Randomly shift the image to avoid tiled boundaries."""
  shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
  return shift, tf.roll(img, shift=shift, axis=[0,1])


class TiledGradients(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),)
  )
  def __call__(self, img, tile_size=512):
    shift, img_rolled = random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)

    # Skip the last tile, unless there's only one tile.
    xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
    if not tf.cast(len(xs), bool):
      xs = tf.constant([0])
    ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
    if not tf.cast(len(ys), bool):
      ys = tf.constant([0])

    for x in xs:
      for y in ys:
        # Calculate the gradients for this tile.
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img_rolled`.
          # `GradientTape` only watches `tf.Variable`s by default.
          tape.watch(img_rolled)

          # Extract a tile out of the image.
          img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
          loss = deepdream.calc_loss(img_tile, self.model)

        # Update the image gradients for this tile.
        gradients = gradients + tape.gradient(loss, img_rolled)

    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(gradients, shift=-shift, axis=[0,1])

    # Normalize the gradients.
    return gradients / (tf.math.reduce_std(gradients) + 1e-8)


def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01,
                                octaves=range(-2,3), octave_scale=1.3):
  """Returns images normalized at [-1, 1]."""
  get_tiled_gradients = TiledGradients(deepdream.dream_model())
  base_shape = tf.shape(img)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  initial_shape = img.shape[:-1]
  img = tf.image.resize(img, initial_shape)
  out = []
  for octave in octaves:
    # Scale the image based on the octave.
    new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
    img = tf.image.resize(img, tf.cast(new_size, tf.int32))
    for step in range(steps_per_octave):
      gradients = get_tiled_gradients(img)
      img = img + gradients*step_size
      img = tf.clip_by_value(img, -1, 1)
      if step % 10 == 0:
        out.append(img)
        print("Octave {}, Step {}".format(octave, step))
  return out


def main():
  logging.basicConfig(
      level=logging.DEBUG,
      format="%(relativeCreated)6d %(message)s")
  logging.info('Loaded')
  # 'https://pbs.twimg.com/profile_images/80041186/chicken.gif'
  original_img = deepdream.download("chicken.gif")

  # TODO(maruel): Doesn't crash but the image generated isn't correct.

  OCTAVE_SCALE = 1.30

  # TODO(maruel): Is this making a copy of the image?
  img = tf.constant(np.array(original_img))
  base_shape = tf.shape(img)[:-1]
  float_base_shape = tf.cast(base_shape, tf.float32)
  for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)
    img = tf.image.resize(img, new_shape).numpy()
    imgs = deepdream.run_deep_dream_simple(img=img, steps=50, step_size=0.01)
    # Discard all but the last image, and denormalize it.
    img = deepdream.denormalize(imgs[-1])
  img = tf.image.resize(img, base_shape)
  img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
  #deepdream.save(img)

  # TODO(maruel): Is this making a copy of the image?
  shift, img_rolled = random_roll(np.array(original_img), 512)
  #deepdream.save(img_rolled)

  imgs = run_deep_dream_with_octaves(img=original_img, step_size=0.01)
  for i in range(len(imgs)):
    imgs[i] = deepdream.denormalize(imgs[i])
    imgs[i] = tf.image.resize(imgs[i], base_shape)
    imgs[i] = tf.image.convert_image_dtype(imgs[i]/255.0, dtype=tf.uint8)
    #deepdream.save(imgs[i])

  imgs = [
      deepdream.PIL.Image.fromarray(deepdream.np.array(img))
      for img in imgs
  ]
  imgs[0].save(
      os.path.join("out", "very_troubled_chicken.gif"),
      save_all=True,
      append_images=imgs[1:],
      duration=[1000] + ([40] * (len(imgs)-2)) + [1000],
      loop=0)
  return 0


if __name__ == "__main__":
  sys.exit(main())
