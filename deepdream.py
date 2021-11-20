# Copyright 2021 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Includes tools to deep dream an image.

Inspired by https://www.tensorflow.org/tutorials/generative/deepdream
"""

import logging
import os
import urllib.parse

# Tell tensorflow to shut up.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Between '0' and '3'

# https://www.tensorflow.org/api_docs/python/tf/all_symbols
import tensorflow as tf
# https://numpy.org/doc/stable/user/quickstart.html
import numpy as np
# https://pillow.readthedocs.io/en/stable/reference/Image.html
import PIL.Image


# Do not run this by default since it slows things down slightly.
# Reminder to make sure I properly configured my GPU:
#if not tf.config.list_physical_devices('GPU'):
#  print('WARNING: No GPU detected')


_SAVE_INDEX = 0
_DEEP_DREAM = None
_DREAM_MODEL = None
# TODO(maruel): It'd be fun to make it work in float16 (Nvidia RTX 20x0) or
# bfloat16 (Cloud TPU or Nvidia RTX 30x0).
_TYPE = tf.float32


def download(url, max_dim=None):
  """Opens an images and returns it as a np.array with [0, 255] range."""
  if urllib.parse.urlparse(url).scheme:
    url = tf.keras.utils.get_file(url.split('/')[-1], origin=url)
  img = PIL.Image.open(url)
	# Convert indexed (gif) or RGBA (png) into RGA with a white background.
  if img.mode != "RGB":
    if img.mode == "P":
      img = img.convert("RGBA")
    background = PIL.Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("RGB")
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  # Convert to numpy array.
  return np.array(img)


def denormalize(img):
  """Denormalizes an image from [-1, +1] to [0, 255]."""
  return tf.cast((255/2.)*(img + 1.0), tf.uint8)


def save(img):
  """Saves an image represented as a normalized uint8 ft.tensor."""
  global _SAVE_INDEX
  _SAVE_INDEX = _SAVE_INDEX + 1
  if not os.path.isdir('out'):
    os.mkdir('out')
  p = os.path.join('out', '%03d.png' % _SAVE_INDEX)
  if os.path.isfile(p):
    os.remove(p)
  PIL.Image.fromarray(np.array(img)).save(p)
  print("Saving image %s" % p)


def calc_loss(img, model):
  """Pass forward the image through the model to retrieve the activations.

  Converts the image into a batch of size 1.

  Returns tf.float32.
  """
  layer_activations = model(tf.expand_dims(img, axis=0))
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]
  # Sum of the medians for each layers.
  medians = [tf.math.reduce_mean(tf.cast(act, _TYPE)) for act in layer_activations]
  return tf.math.reduce_sum(medians)


class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=_TYPE),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=_TYPE),)
  )
  def __call__(self, img, steps, step_size):
      loss = tf.constant(0.0, dtype=_TYPE)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)
        # Calculate the gradient of the loss with respect to the pixels of the
        # input image.
        gradients = tape.gradient(loss, img)
        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        # In gradient ascent, the "loss" is maximized so that the input image
        # increasingly "excites" the layers. You can update the image by
        # directly adding the gradients (because they're the same shape!)
        p = gradients*step_size
        img = img + p
        img = tf.clip_by_value(img, -1, 1)
      return loss, img


def dream_model():
  """Lazy create the deep dream model."""
  global _DREAM_MODEL
  if not _DREAM_MODEL:
    # Deepdream model.
    _base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    # Maximize the activations of these layers.
    _layers = [_base_model.get_layer(name).output for name in ['mixed3', 'mixed5']]
    # Create the feature extraction model.
    _DREAM_MODEL = tf.keras.Model(inputs=_base_model.input, outputs=_layers)
  return _DREAM_MODEL


def deepdream(*args, **kwargs):
  """Runs deep dream on an image.

  Lazy load the model.
  """
  global _DEEP_DREAM
  if not _DEEP_DREAM:
    _DEEP_DREAM = DeepDream(dream_model())
  return _DEEP_DREAM(*args, **kwargs)


def run_deep_dream_simple(img, steps, step_size, steps_per_output=100):
  """Generates multiple deep dream images.

  Args:
    img: np.array

  Returns:
    list of normalized [-1, +1] deep dreamed images. Includes the original
    image.
  """
  # Convert from uint8 [0, +255] to the range expected by the model, which is
  # float32 [-1, +1].
  # https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  # https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor
  img = tf.convert_to_tensor(img, dtype=_TYPE)
  step_size = tf.constant(step_size, dtype=_TYPE)
  steps_remaining = steps
  step = 0
  run_steps = tf.constant(steps_per_output)
  out = [img]
  while steps_remaining:
    if steps_remaining < run_steps:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    loss, img = deepdream(img, run_steps, step_size)
    out.append(img)
    step += run_steps
    logging.info("Step {}, loss {:.2f}".format(step, loss))
  return out
