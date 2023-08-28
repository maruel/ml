import cgi
import contextlib
import os
import sys
import urllib.request

# Silence tensorflow even if we don't use it here.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import diffusers
import torch
import transformers


if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32


class Params(object):
    """Text to Image Parameters as a class so it can be easily serialized."""
    def __init__(self, prompt, neg_prompt="", seeds=[1], steps=25, guidance=7.5,
                width=768, height=768):
        assert isinstance(prompt, str) and prompt
        assert isinstance(neg_prompt, str)
        assert isinstance(seeds, (list, tuple)) and all(
            1 <= seed <= 2147483647 for seed in seeds)
        assert isinstance(steps, int) and 1 <= steps <= 1000
        assert isinstance(guidance, float) and 0. <= guidance <= 15.
        assert isinstance(width, int) and 16 <= width <= 8192
        assert isinstance(height, int) and 16 <= height <= 8192
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.seeds = seeds
        self.steps = steps
        self.guidance = guidance
        self.width = width
        self.height = height

    def kwargs(self):
        return {
            "prompt": self.prompt,
            "negative_prompt": self.neg_prompt,
            "seeds": self.seeds,
            "num_inference_steps": self.steps,
            "guidance_scale": self.guidance,
            "width": self.width,
            "height": self.height,
        }


class CivitaiModel(object):
    """The model to run."""
    def __init__(
        self, doc_url, model_url, filename, type, base_model="runwayml/stable-diffusion-v1-5",
        clip_skip=1):
        """The model to use.
        See
        https://github.com/civitai/civitai/wiki/How-to-use-models

        type is either:
          "checkpoint" for a Checkpoint Merge
          "lora" for a LoRA

        base_model is only needed if clip_skip > 1.

        clip_skip follows community convention:
          clip_skip = 1 uses the all text encoder layers.
          clip_skip = 2 skips the last text encoder layer.

        encoder is only used if clip_skip > 1.
        """
        assert isinstance(doc_url, str)
        assert isinstance(model_url, str)
        assert type in ("checkpoint", "lora")
        assert not filename or os.path.splitext(filename)[1] == ".safetensors", filename
        self.doc_url = doc_url
        self.model_url = model_url
        # Sadly the Cloudflare web worker that civitai uses doesn't allow HEAD.
        self.filename = filename
        self.type = type
        self.base_model = base_model
        self.clip_skip = clip_skip

    def download(self):
        if self.filename and os.path.isfile(self.filename):
          print("Loading", self.filename)
          return self.filename
        print("Downloading", self.model_url)
        req = urllib.request.Request(url=self.model_url, headers={"User-Agent": "curl/7.81.0"})
        last = [""]
        def reporthook(blocknum, bs, size):
          n = "\r%.01f%%" % (100.*float(blocknum*bs)/float(size))
          if n != last[0]:
            sys.stderr.write(n)
            last[0] = n
          sys.stderr.flush()
        self.filename, _ = _urlretrieve(req, self.filename, reporthook=reporthook)
        sys.stderr.write("\rDownloaded %s\n" % self.filename)
        sys.stderr.flush()
        return self.filename

    def convert(self):
        """Converts a safetensors into something usable.

        There's something magical here that I (M-A) don't understand yet.

        """
        filename = self.download()
        dump_path = os.path.splitext(filename)[0]
        if not os.path.isdir(dump_path):
          print("Converting", filename)
          controlnet = False
          # See
          # https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
          # for example. This is calling into
          # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py
          # I'm not sure why it's called "download", it doesn't seem to be downloading
          # anything.

          # Late import since it won't be used often.
          from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
          pipe = download_from_original_stable_diffusion_ckpt(
              checkpoint_path=filename,
              from_safetensors=True,
              controlnet=controlnet,
              local_files_only=True,
          )
          # Not sure why to do that, beside saving loading time? Probably worth
          # when transfering to a VM.
          #pipe.to(torch_dtype=torch.float16)
          if controlnet:
              # Only save the controlnet model.
              pipe.controlnet.save_pretrained(dump_path, safe_serialization=True)
          else:
              pipe.save_pretrained(dump_path, safe_serialization=True)
        return dump_path

    def to_pipe(self):
        """Returns a loaded ML pipeline."""
        # TODO(maruel): Implement LoRA assert self.type == "checkpoint"
        model_path = self.convert()
        if self.clip_skip > 1:
            pipe = diffusers.DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
                text_encoder=transformers.CLIPTextModel.from_pretrained(
                    self.base_model,
                    subfolder="text_encoder",
                    num_hidden_layers=12 - (self.clip_skip - 1),
                    torch_dtype=torch_dtype,
                    local_files_only=True,
                ),
            )
        else:
            pipe = diffusers.DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
        pipe = pipe.to(device)
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config)
        if True: # self.engine == "cuda":
          # Memory efficient attention is only available on GPU.
          # Doesn't seem to have any impact?
          pipe.enable_xformers_memory_efficient_attention()
        return pipe


def get_prompt_embeddings(pipe, prompt, negative_prompt, split_character=","):
    """Prompt embeddings to overcome CLIP 77 token limit.
    https://github.com/huggingface/diffusers/issues/2136
    """
    # Check if the prompt is longer than the negative prompt by splitting the
    # string with split_character.
    if len(prompt.split(split_character)) >= len(negative_prompt.split(split_character)):
        # If prompt is longer than or equal to negative prompt.
        input_ids = pipe.tokenizer(
            prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_maxlen = input_ids.shape[-1]
        neg_ids = pipe.tokenizer(
            negative_prompt,
            truncation=False,
            padding="max_length",
            max_length=shape_maxlen,
            return_tensors="pt"
        ).input_ids.to(device)
    else:
      # If negative prompt is longer than prompt.
        neg_ids = pipe.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_maxlen = neg_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
            max_length=shape_maxlen,
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    prompt_embeds = []
    neg_embeds = []
    maxlen = pipe.tokenizer.model_max_length
    for i in range(0, shape_maxlen, maxlen):
        prompt_embeds.append(pipe.text_encoder(input_ids[:, i: i + maxlen])[0])
        neg_embeds.append(pipe.text_encoder(neg_ids[:, i: i + maxlen])[0])
    return torch.cat(prompt_embeds, dim=1), torch.cat(neg_embeds, dim=1)



def _urlretrieve(req, filename, reporthook=None):
  """Simpler urlretrieve() that enables specifying a request object instead of
  an URL, and leverages content-disposition.
  """
  with contextlib.closing(urllib.request.urlopen(req)) as fp:
      headers = fp.info()
      #size = int(headers["content-length"]) if "content-length" in headers else -1
      size = int(headers.get("content-length", -1))
      ct = headers.get("content-disposition")
      if not filename:
        if not ct:
          raise urllib.request.URLError(
              "filename not specified and content-disposition header not present")
        cdisp, pdict = cgi.parse_header(ct)
        filename = pdict['filename']
      bs = 1024*8
      read = 0
      blocknum = 0
      result = (filename, headers)
      with open(filename, 'wb') as tfp:
          if reporthook:
            reporthook(blocknum, bs, size)
          while True:
              block = fp.read(bs)
              if not block:
                  break
              read += len(block)
              tfp.write(block)
              blocknum += 1
              if reporthook:
                reporthook(blocknum, bs, size)
  if size >= 0 and read < size:
      raise urllib.request.ContentTooShortError(
          "retrieval incomplete: got only %i out of %i bytes"
          % (read, size), result)
  return result
