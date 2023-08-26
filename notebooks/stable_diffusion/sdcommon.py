import functools
import gc
import json
import os
import sys

# Silence tensorflow even if we don't use it here.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
  import PIL
  # Not used here but will fail at runtime after downloading.
  import accelerate
  # Imported elsewhere.
  import diffusers
  # Not used here but will fail at runtime.
  import transformers
  import torch
except ImportError as e:
  print(e)
  print("Run: pip3 install accelerate diffusers transformers torch torchvision Pillow", file=sys.stderr)
  sys.exit(1)


def getdefaultengine():
  """Returns the ML engine to use. One of cuda or cpu."""
  if torch.cuda.is_available():
    return "cuda"
  # The MPS engine on a M1 Pro is slower than pure CPU while also affecting
  # system performance significantly.
  # Keeping this in case this improves later.
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def autogc(f):
  """Runs gc.collect() after the function returns."""
  @functools.wraps(f)
  def w(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    finally:
      gc.collect()
  return w


def getimg(name):
  """Retrieve a picture and resize it to be less than 1024x768 or 768x1024.

  It injects a .name field in the returned object.
  """
  img = PIL.Image.open(name)

  # Convert indexed (gif) or RGBA (png) into RGA with a white background.
  if img.mode != "RGB":
    if img.mode == "P":
      img = img.convert("RGBA")
    background = PIL.Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("RGB")

  # Max is 1024x768 or 768x1024?
  size = img.size
  while size[0] > 1024 or size[1] > 1024: # or size[0] * size[1] > 786432:
    # TODO: Round to nearest 32.
    size = (size[0]//2, size[1]//2)
  if size != img.size:
    print("Resized from", img.size, "to", size)
    img = img.resize(size, PIL.Image.Resampling.LANCZOS)
  # Inject the original name back into the object.
  img.name = name
  return img


def classesdict(*classes):
  return {cls.__name__: cls for cls in classes}


def to_dict(obj):
  return {k: v for k, v in obj.__dict__.items() if not k.startswith("_") and not callable(v)}


def save(ml, params, *images):
  """Saves ML and its parameters, plus resulting imags."""
  # Do a binary search. Lame but "good enough".
  data = {"ml": to_dict(ml), "params": to_dict(params)}
  base = _find("out/" + ml.algo)
  if not os.path.isdir("out"):
    os.mkdir("out")
  with open(base + ".json", "w") as f:
    json.dump(data, f, sort_keys=True, indent=2)
  for i, img in enumerate(images):
    if not i:
      img.save(base + ".png")
    else:
      img.save(base + ("_%d.png" % i))
  return base


def unroll(**kwargs):
  """Unrolls list/tuple elements in a kwargs; annotate each call with a label."""
  import itertools
  loops = {k: v for k, v in kwargs.items() if isinstance(v, (list, range, tuple))}
  if not loops:
    return [kwargs]
  kwargs = {k: v for k, v in kwargs.items() if k not in loops}
  keys = sorted(loops)
  iters = list(itertools.product(*[loops[k] for k in keys]))
  out = []
  for line in iters:
    label = " ".join("%s:%s" % (k, line[i]) for i, k in enumerate(keys))
    d = {k: line[i] for i, k in enumerate(keys)}
    d.update(kwargs)
    out.append((label, d))
  return out


## Private stuff.


def _find(prefix=""):
  """Finds the next available output file by looking for json files."""
  fmt = prefix + "%05d"
  return fmt % _search(lambda x: not os.path.isfile((fmt % x) + ".json"), 0, 9999)


def _search(isavail, low, high):
  if low >= high:
    return low if isavail(low) else -1
  mid = (high + low) // 2
  if isavail(mid):
    # Search smaller values.
    out = _search(isavail, low, mid - 1)
    if out == -1:
      return mid
    return out
  # Search larger values.
  return _search(isavail, mid + 1, high)
