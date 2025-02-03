r"""Benchmark LLM serving throughput.

Source: https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
This script is for sending requests with prompts to LLM server and benchmark
the latency and throughput.

Aggregated results are saved in the file (prefix = --name):
  <prefix>_aggregated_results.json
If --save-full-results is set, full results are saved in the file:
  <prefix>_full_results.json

These result files can be loaded with pandas.read_json():
  df = pandas.read_json("file_name.json", orient="records", lines=True)

===================================
Example run for naive HF benchmark:
===================================

On the server side, launch the docker built from peft/dockerfile/serve.Dockerfile:
MODEL_SIZE=13b
PORT=7090
PRECISION=float16

docker run \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="0" \
  -e TASK="causal-language-modeling-lora" \
  -e PRECISION_LOADING_MODE=${PRECISION} \
  -e MODEL_ID=llama2-${MODEL_SIZE}-chat-hf \
  --rm --name ${MODEL_SIZE}-${PRECISION} \
  -p ${PORT}:7080 \
  -p 7091:7081 \
  -it <hf-serve-docker-image-tag>


On the client side, run:
NOTE: You need to run `pip install --upgrade transformers accelerate` once
on the client environment before running this script.
MODEL_SIZE=13b
PORT=7090

python benchmark_serving.py \
  --backend=naive_transformers --host=localhost --port=${PORT} \
  --dataset=ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer=llama2-${MODEL_SIZE}-chat-hf --request-rate=200 \
  --endpoint=predictions/peft_serving

===============================
Example run for TGI benchmark:
===============================

It is recommended to test vLLM and TGI with the OpenAI ChatCompletions API.

On the server side, run:
PORT=1234
VOLUME="${PWD?}/data"
MODEL="openlm-research/open_llama_13b"

docker run \
  --gpus all \
  --shm-size 16g \
  -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  -p ${PORT?}:80 \
  -v ${VOLUME?}:/data \
  --rm \
  ghcr.io/huggingface/text-generation-inference:1.0.0 \
  --model-id ${MODEL?} \
  --sharded false  \
  --max-input-length 1024 \
  --max-total-tokens 2048 \
  --max-concurrent-requests 5000 \
  --max-best-of 1

On the client side, run the chat_completions benchmark (below).

===============================
Example run for vLLM benchmark:
===============================

python -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 \
  --port=7080 \
  --model=openlm-research/open_llama_13b \
  --tensor-parallel-size=1 \
  --swap-space=16 \
  --gpu-memory-utilization=0.95 \
  --disable-log-stats

On the client side, run the chat_completions benchmark (below).

==============================================
Example run for chat_completions benchmark:
==============================================

Note: The --max-input-length, -max-output-length and --c can all be lists. All
combinations will be benchmarked.

python benchmark_serving.py \
  --backend=chat_completions \
  --model=/llama/Meta-Llama-3-8B-Instruct \
  --host=127.0.0.1 \
  --port=8000 \
  --dataset=ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer=openlm-research/open_llama_13b \
  --max-input-length=64,128,256,512,1024 \
  --max-output-length=64,128,256,512,1024 \
  --num-prompts=1000 \
  --save-full-results=True \
  --c=1,10,30,100 \
  --name=vllm_llama3_8b_chat

==============================================
Example run for TensorRT-LLM Triton benchmark:
==============================================

Launch the Triton server following:
https://github.com/triton-inference-server/tensorrtllm_backend/tree/v0.9.0

python3 benchmark_serving.py \
  --backend=trt_llm_stream \
  --endpoint=v2/models/ensemble/generate_stream \
  --model=/Meta-Llama-3-8B-Instruct \
  --host=127.0.0.1 \
  --port=8000 \
  --dataset=ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer=/Meta-Llama-3-8B-Instruct \
  --max-input-length=128,1024 \
  --max-output-length=64,1024 \
  --num-prompts=1000 \
  --save-full-results=True \
  --c=1,32,128 \
  --name=trt_llama3_8b

==============================
Example run for SAX benchmark:
==============================

Launch the SAX admin and model servers and launch an asynchronous API server.

On the client side, run:
NOTE: You need to run `pip install --upgrade transformers accelerate` once
on the client environment before running this script.
python benchmark_serving.py \
  --backend=sax \
  --model=llama2-7b \
  --endpoint=v1/completions \
  --request-rate=200 \
  --host=127.0.0.1 \
  --port=8000 \
  --dataset=ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer=<path-to-tokenizer>

==============================
Example run for chat completions on publisher endpoint:
==============================

export API_KEY=$(gcloud auth print-access-token) && \
python3 benchmark_serving.py \
  --backend=chat_completions \
  --model=meta/llama3-405b-instruct-maas \
  --host=https://us-central1-aiplatform.googleapis.com  \
  --endpoint=v1beta1/projects/cloud-llm-preview1/locations/us-central1/endpoints/openapi/chat/completions \
  --dataset=sonnet.txt  \
  --tokenizer=<path-to-tokenizer>  \
  --max-input-length=1200  \
  --max-output-length=250  \
  --num-prompts=5  \
  --fixed-qps=5 \
  -v=info

==============================
Example run for prefix caching benchmark on local server:
==============================

python benchmark_serving.py \
  --backend=chat_completions \
  --model=$HOME/data/Meta-Llama-3.1-8B-Instruct \
  --tokenizer=$HOME/data/Meta-Llama-3.1-8B-Instruct \
  --seed=0 \
  --port=7080 \
  --endpoint=v1/chat/completions \
  --num-prompts=50 \
  --max-input-length=2048  \
  --max-output-length=50 \
  --concurrent-requests=1,10,20 \
  --cache-hit-ratio=0.0,0.5,1.0 \
  --prefill-len-padding=512 \
  --dataset=sonnet.txt \
  --sonnet-prefix-len=50 \
  --token-id-start=8000

"""

# pylint: disable=g-multiple-import
# pylint: disable=g-importing-member
# pylint: disable=logging-fstring-interpolation
# pylint: disable=f-string-without-interpolation

from abc import ABC
from abc import abstractmethod
import argparse
import asyncio
import base64
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import io
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from typing import Any, AsyncGenerator, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from PIL import Image
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer


# Identify whether we are running in google3.
# Executing googleinit is required for the google3 runs.
try:
  from google3.base.python.clif import googleinit  # pylint: disable=g-import-not-at-top
except ImportError:
  googleinit = None


MIN_SEQ_LEN = 4
CLIENT_TIMEOUT_SEC = 3 * 60 * 60
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SEC)
NEW_TEXT_KEY = "\nOutput:\n"


class Counter:

  def __init__(self, start: int = 0) -> None:
    self.counter = start

  def __next__(self) -> int:
    i = self.counter
    self.counter += 1
    return i

  def reset(self, start: int = 0) -> None:
    self.counter = start


token_id_counter = Counter(start=0)


class BaseTokenizer(ABC):
  """Abstract class for tokenizers.
  """

  @abstractmethod
  def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
    pass

  @abstractmethod
  def decode(self, token_ids: List[int]) -> str:
    pass

  @abstractmethod
  def apply_chat_template(
      self,
      message: List[dict[str, Any]],
      add_generation_prompt: bool = True,
      tokenize: bool = False,
  ) -> str:
    pass

  @abstractmethod
  def all_special_ids(self) -> List[int]:
    pass

  @abstractmethod
  def get_vocab(self) -> dict[str, int]:
    pass

  @abstractmethod
  def bos_token(self) -> str:
    pass


class Llama3Tokenizer(BaseTokenizer):
  """Llama3 specific tokenizer, based on Tiktoken.

  Regular thirdparty google3 transformers could not load the llama3 tokenizer.
  """

  def __init__(self, tokenizer_path: str):
    from saxml.server.pax.lm import vocabularies  # pylint: disable=g-import-not-at-top

    self._tokenizer = vocabularies.LLama3Vocabulary(tokenizer_path)

  def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
    del add_special_tokens
    return list(self._tokenizer.encode(text))

  def decode(self, token_ids: List[int]) -> str:
    return self._tokenizer.decode(token_ids)

  def apply_chat_template(
      self,
      message: List[dict[str, Any]],
      add_generation_prompt: bool = True,
      tokenize: bool = False
  ) -> str:
    del add_generation_prompt, tokenize, message
    # This is not required for the servomatic backend.
    # The formatted prompt is ignored and regular prompt is used.
    logging.debug("apply_chat_template is not supported for Llama3Tokenizer.")
    return ""

  def all_special_ids(self) -> List[int]:
    raise NotImplementedError("Not implemented for Llama3Tokenizer.")

  def get_vocab(self) -> dict[str, int]:
    raise NotImplementedError("Not implemented for Llama3Tokenizer.")

  def bos_token(self) -> str:
    raise NotImplementedError("Not implemented for Llama3Tokenizer.")


class GeneralTokenizer(BaseTokenizer):
  """General tokenizer, based on transformers.AutoTokenizer, used for OSS runs.
  """

  def __init__(self, tokenizer_path: str, trust_remote_code: bool = False):
    self._tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )

  def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
    return list(
        self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
    )

  def decode(self, token_ids: List[int]) -> str:
    return self._tokenizer.decode(token_ids)

  def apply_chat_template(
      self,
      message: List[dict[str, Any]],
      add_generation_prompt: bool = True,
      tokenize: bool = False
  ) -> str:
    return self._tokenizer.apply_chat_template(
        message, add_generation_prompt=add_generation_prompt, tokenize=tokenize
    )

  def all_special_ids(self) -> List[int]:
    return self._tokenizer.all_special_ids

  def get_vocab(self) -> dict[str, int]:
    return self._tokenizer.get_vocab()

  def bos_token(self) -> str:
    return self._tokenizer.bos_token


def next_valid_token_id(tokenizer: BaseTokenizer) -> int:
  vocab_size = len(tokenizer.get_vocab())
  token_id = next(token_id_counter) % vocab_size
  while token_id in tokenizer.all_special_ids():
    token_id = next(token_id_counter) % vocab_size
  return token_id


def str2bool(v: str) -> Optional[bool]:
  if v is None:
    return None
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


def sample_sharedgpt_requests(
    dataset_path: str,
    num_requests: int,
    max_input_len: int,
    max_output_len: int,
    tokenizer: BaseTokenizer,
    use_dummy_text: bool,
    human_prompts_only: bool = False,
    fixed_output_len: bool = False,
) -> List[Tuple[str, int, int]]:
  """Samples requests from the SharedGPT dataset or creates dummy requests."""
  if use_dummy_text:
    dummy_prompt_token_ids = [0] * max_input_len
    dummy_prompt = tokenizer.decode(dummy_prompt_token_ids)
    dummy_requests = [(
        dummy_prompt,
        max_input_len,
        max_output_len,
    )] * num_requests
    return dummy_requests

  # Load the dataset.
  with open(dataset_path) as f:
    dataset = json.load(f)
  # Filter out the conversations with less than 2 turns.
  dataset = [data for data in dataset if len(data["conversations"]) >= 2]
  # Only keep the first two turns of each conversation.
  dataset = [
      (data["conversations"][0]["value"], data["conversations"][1]["value"])
      for data in dataset
      if (not human_prompts_only or data["conversations"][0]["from"] == "human")
  ]

  # Tokenize the prompts and completions.
  prompts = [prompt for prompt, _ in dataset]
  prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
  completions = [completion for _, completion in dataset]
  completion_token_ids = [
      tokenizer.encode(completion) for completion in completions
  ]
  tokenized_dataset = []
  for i in range(len(dataset)):
    output_len = len(completion_token_ids[i])
    tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

  # Filter out too long sequences.
  filtered_dataset: List[Tuple[str, int, int]] = []
  for prompt, prompt_token_ids, output_len in tokenized_dataset:
    prompt_len = len(prompt_token_ids)
    if prompt_len < MIN_SEQ_LEN or output_len < MIN_SEQ_LEN:
      # Prune too short sequences.
      # This is because TGI causes errors when the input or output length
      # is too short.
      continue
    if prompt_len > max_input_len or (
        not fixed_output_len and output_len > max_output_len
    ):
      # Prune too long sequences.
      continue
    filtered_dataset.append(
        (prompt, prompt_len, max_output_len if fixed_output_len else output_len)
    )

  # Sample the requests.
  sampled_requests = random.sample(filtered_dataset, num_requests)
  return sampled_requests


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    min_input_len: int,
    max_input_len: int,
    min_output_len: int,
    max_output_len: int,
    prefix_len: int,
    tokenizer: BaseTokenizer,
    fixed_input_length: Optional[int] = None,
    fixed_output_length: Optional[int] = None,
) -> List[Tuple[str, str, int, int, int]]:
  """Samples requests from the Sonnet dataset.

  Args:
    dataset_path: Path to the Sonnet dataset.
    num_requests: Number of requests to sample.
    min_input_len: Minimum input length.
    max_input_len: Maximum input length.
    min_output_len: Minimum output length.
    max_output_len: Maximum output length.
    prefix_len: Number of prefix tokens per request.
    tokenizer: Tokenizer to use.
    fixed_input_length: If specified, forces input_len to be fixed_input_length.
    fixed_output_length: If specified, forces output_len to be
      fixed_output_length.

  Returns:
    A list of tuples containing the prompt, formatted prompt, prompt length,
    formatted prompt length, and output length.
  """

  # Load the dataset.
  with open(dataset_path) as f:
    poem_lines = f.readlines()
  poem_lines = poem_lines * 100

  # Tokenize the poem lines.
  poem_token_ids = [tokenizer.encode(poem_line) for poem_line in poem_lines]
  average_poem_len = sum(len(token_ids) for token_ids in poem_token_ids) / len(
      poem_token_ids
  )

  # Base prefix for all requests.
  base_prompt = "Pick as many lines as you can from these poem lines:\n"
  base_message = [{
      "role": "user",
      "content": base_prompt,
  }]
  base_prompt_formatted = tokenizer.apply_chat_template(
      base_message, add_generation_prompt=True, tokenize=False
  )
  base_prompt_offset = len(tokenizer.encode(base_prompt_formatted))

  logging.info("prefix_len: %s", prefix_len)
  logging.info("base_prompt_offset: %s", base_prompt_offset)
  logging.info("base_prompt_formatted: %s", base_prompt_formatted)
  logging.info(
      "base_prompt_formatted.input_ids: %s",
      tokenizer.encode(base_prompt_formatted),
  )

  # First approximately `prefix_len` number of tokens in the
  # prompt are fixed poem lines.
  assert (
      prefix_len > base_prompt_offset
  ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

  num_prefix_lines = round((prefix_len - base_prompt_offset) / average_poem_len)
  prefix_lines = poem_lines[:num_prefix_lines]

  # Sample the rest of lines per request.
  sampled_requests: List[Tuple[str, str, int, int, int]] = []
  for _ in range(num_requests):
    if fixed_input_length:
      input_len = fixed_input_length
    else:
      input_len = (
          random.randrange(min_input_len, max_input_len)
          if max_input_len > min_input_len
          else min_input_len
      )
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."
    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round((input_len - base_prompt_offset) / average_poem_len)

    if fixed_output_length:
      output_len = fixed_output_length
    else:
      output_len = (
          random.randrange(min_output_len, max_output_len)
          if max_output_len > min_output_len
          else min_output_len
      )

    sampled_lines = "".join(
        prefix_lines
        + random.sample(poem_lines, num_input_lines - num_prefix_lines)
    )

    prompt = f"{base_prompt}{sampled_lines}"
    message = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    prompt_formatted = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, tokenize=False
    )

    prompt_len = len(tokenizer.encode(prompt))
    prompt_formatted_len = len(tokenizer.encode(prompt_formatted))
    sampled_requests.append(
        (prompt, prompt_formatted, prompt_len, prompt_formatted_len, output_len)
    )

  return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
) -> AsyncGenerator[Tuple[str, int, int], None]:
  """Gets request async."""
  input_requests = iter(input_requests)
  for request in input_requests:
    yield request


@dataclass
class RequestFuncInput:
  """Input to the request function.

  Attributes:
    backend: Backend to benchmark.
    api_url: The API URL to send the request to.
    prompt: The prompt to send to the model.
    prompt_len: The length of the prompt.
    output_len: Expected output length.
    best_of: Pick the best of N samples.
    use_beam_search: Whether to use beam search.
    top_k: Top K for sampling.
    model: Model name.
    extra_body: Extra body to send in the request.
    images: List of image URLs to send in the request.
    tokenized_prompt: Tokenized prompt, required for servomatic and evergreen.
    stub: Stub for the backend, currently used to send requests to servomatic
      and evergreen.
  """
  backend: str = ""
  api_url: str = ""
  prompt: str = ""
  prompt_len: int = 0
  output_len: int = 0
  best_of: Optional[int] = None
  use_beam_search: Optional[bool] = None
  top_k: Optional[int] = None
  model: str = ""
  extra_body: str | dict[str, Any] | None = None
  images: List[str] = field(default_factory=list)
  tokenized_prompt: Optional[List[int]] = None
  stub: Optional[Any] = None


@dataclass
class RequestFuncOutput:
  """Output of the request function.

  Attributes:
    backend: Backend to benchmark.
    model: Model name.
    generated_text: Generated text in case of non-servomatic.
    generated_token_ids: List of generated token ids in case of servomatic and
      evergreen.
    success: Whether the request was successful.
    latency: total request latency
    prompt_len: input prompt length
    error: Error message if any
    ttft: Time to first token
    itl: Inter-token latencies
    requested_output_len:
  """
  backend: str = ""
  model: str = ""
  generated_text: str = ""
  generated_token_ids: Optional[List[int]] = None
  success: bool = False
  latency: float = 0.0
  prompt_len: int = 0
  error: str = ""
  ttft: Optional[float] = None  # Time to first token
  itl: List[float] = field(
      default_factory=list)  # List of inter-token latencies
  requested_output_len: Optional[int] = None


def remove_prefix(text: str, prefix: str) -> str:
  if text.startswith(prefix):
    return text[len(prefix):]
  return text


async def send_hex_llm_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = True,
) -> RequestFuncOutput:
  """Sends a request to Hex-LLM."""
  del stream
  async with sem:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
      payload = {
          "prompt": request_input.prompt,
          "max_tokens": (
              request_input.output_len or 1
          ),  # For hex-llm, 0 means max length, 1 is prefilling only.
          "ignore_eos": True,
          "streaming_tokens": 2048,
      }
      output = RequestFuncOutput()
      output.backend = request_input.backend
      output.model = request_input.model
      output.prompt_len = request_input.prompt_len
      output.requested_output_len = request_input.output_len

      ttft = None
      st = time.perf_counter()
      try:
        async with session.post(
            url=request_input.api_url,
            json=payload,
        ) as response:
          if response.status == 200:
            async for chunk in response.content.iter_chunked(8192):
              if not chunk:
                continue
              if ttft is None:
                ttft = time.perf_counter() - st
                output.ttft = ttft
              data = json.loads(chunk.decode("utf-8"))
              output.generated_text += data["predictions"][0]
            output.success = True
            output.latency = time.perf_counter() - st
          else:
            output.error = response.reason or ""
            output.success = False
            print(output.error)
      except Exception:  # pylint: disable=broad-except
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        print(output.error)

      if pbar:
        pbar.update(1)
      return output


async def send_vllm_stream_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = True,
) -> RequestFuncOutput:
  """Sends a streaming request to vLLM."""
  if not stream:
    stream = True
  assert stream
  async with sem:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
      payload = {
          "prompt": request_input.prompt,
          "n": 1,
          "best_of": request_input.best_of,
          "use_beam_search": request_input.use_beam_search,
          "temperature": 0.0 if request_input.use_beam_search else 1.0,
          "top_p": 1.0,
          "max_tokens": request_input.output_len,
          "ignore_eos": True,
          "stream": stream,
      }

      output = RequestFuncOutput()
      output.backend = request_input.backend
      output.model = request_input.model
      output.prompt_len = request_input.prompt_len
      output.requested_output_len = request_input.output_len

      ttft = 0.0
      st = time.perf_counter()
      most_recent_timestamp = st
      try:
        async with session.post(
            url=request_input.api_url, json=payload
        ) as response:
          if response.status == 200:
            async for chunk_bytes in response.content.iter_chunks():
              chunk_bytes = chunk_bytes[0].strip()
              if not chunk_bytes:
                continue

              timestamp = time.perf_counter()
              output.generated_text += json.loads(
                  chunk_bytes.decode("utf-8").strip().strip("\0")
              )["predictions"][0]

              # First token
              if ttft == 0.0:
                if output.generated_text:
                  ttft = timestamp - st
                  output.ttft = ttft

              # Decoding phase
              else:
                output.itl.append(timestamp - most_recent_timestamp)

              most_recent_timestamp = timestamp

            output.success = True
            output.latency = time.perf_counter() - st
          else:
            output.error = response.reason or ""
            output.success = False
            print(output.error)
      except Exception:  # pylint: disable=broad-except
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        print(output.error)

      if pbar:
        pbar.update(1)
      return output


async def send_trt_llm_stream_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = True,
) -> RequestFuncOutput:
  """Sends a streaming request to OpenAI Chat Completions API."""
  assert request_input.api_url.endswith(
      "generate_stream"
  ), "TRT LLM API URL must end with 'generate_stream'."
  if not stream:
    stream = True
  assert stream

  async with sem:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
      payload = {
          "accumulate_tokens": True,
          "text_input": request_input.prompt,
          "temperature": 0.0,
          "top_p": 1.0,
          "max_tokens": request_input.output_len,
          "stream": stream,
      }

      output = RequestFuncOutput()
      output.backend = request_input.backend
      output.model = request_input.model
      output.prompt_len = request_input.prompt_len

      ttft = 0.0
      st = time.perf_counter()
      most_recent_timestamp = st
      try:
        async with session.post(
            url=request_input.api_url, json=payload
        ) as response:
          if response.status == 200:
            async for chunk_bytes in response.content:
              chunk_bytes = chunk_bytes.strip()
              if not chunk_bytes:
                continue

              chunk = chunk_bytes.decode("utf-8").removeprefix("data:").strip()
              timestamp = time.perf_counter()
              data = json.loads(chunk)
              # First token
              if ttft == 0.0:
                ttft = time.perf_counter() - st
                output.ttft = ttft

              # Decoding phase
              else:
                output.itl.append(timestamp - most_recent_timestamp)

              output.generated_text += data["text_output"]

              most_recent_timestamp = timestamp

            output.success = True
            output.latency = time.perf_counter() - st
          else:
            output.error = response.reason or ""
            output.success = False
            print(output.error)
      except Exception:  # pylint: disable=broad-except
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        print(output.error)

      if pbar:
        pbar.update(1)
      return output


async def send_chat_completions_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = True,
) -> RequestFuncOutput:
  """Sends a streaming request to OpenAI Chat Completions API."""
  assert request_input.api_url.endswith(
      "chat/completions"
  ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

  if stream is None:
    stream = True  # defaults to True

  async with sem:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
      if not request_input.images:
        content = request_input.prompt
      else:
        content = []
        for image_url in request_input.images:
          content.append({"image_url": {"url": image_url}, "type": "image_url"})
        content.append({"text": request_input.prompt, "type": "text"})

      payload = {
          "model": request_input.model,
          "messages": [
              {
                  "role": "user",
                  "content": content,
              },
          ],
          "temperature": 0.0,
          "max_tokens": request_input.output_len,
          "stream": stream,
      }
      if request_input.extra_body:
        payload["extra_body"] = request_input.extra_body
      api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", ""))
      headers = {
          "Content-Type": "application/json",
      }
      if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

      output = RequestFuncOutput()
      output.backend = request_input.backend
      output.model = request_input.model
      output.prompt_len = request_input.prompt_len
      output.requested_output_len = request_input.output_len

      generated_text = ""
      ttft = 0.0
      st = time.perf_counter()
      most_recent_timestamp = st
      try:
        logging.debug("request: %s", json.dumps(payload, indent=2))
        async with session.post(
            url=request_input.api_url, json=payload, headers=headers
        ) as response:
          if response.status == 200:
            output.success = True
            async for chunk_bytes in response.content:
              chunk_bytes = chunk_bytes.strip()
              if not chunk_bytes:
                continue

              chunk = chunk_bytes.decode("utf-8").removeprefix("data:").strip()
              logging.debug("chunk: %s", chunk)
              if chunk != "[DONE]":
                try:
                  data = json.loads(chunk)
                except json.decoder.JSONDecodeError:
                  logging.error(f"Failed to parse response chunk: {chunk}")
                  output.success = False
                  continue
                timestamp = time.perf_counter()
                if "choices" not in data or not data["choices"]:
                  logging.info("empty chunk: %s", chunk)
                  continue
                if stream:
                  if "delta" not in data["choices"][0]:
                    logging.info("empty delta in chunk: %s", chunk)
                    continue
                  delta = data["choices"][0]["delta"]
                  if delta.get("content", None):
                    # First token
                    if ttft == 0.0:
                      ttft = time.perf_counter() - st
                      output.ttft = ttft

                    # Decoding phase
                    else:
                      output.itl.append(timestamp - most_recent_timestamp)
                    generated_text += delta["content"]
                else:
                  assert not generated_text
                  if "message" not in data["choices"][0]:
                    logging.info("empty message in chunk: %s", chunk)
                    continue
                  if "content" not in data["choices"][0]["message"]:
                    logging.info("empty message.content in chunk: %s", chunk)
                    continue
                  generated_text = data["choices"][0]["message"]["content"]

                most_recent_timestamp = timestamp

            output.generated_text = generated_text
            output.latency = time.perf_counter() - st
          else:
            output.error = response.reason or ""
            output.success = False
            logging.warning(output.error)
      except Exception:  # pylint: disable=broad-except
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        logging.warning(output.error)

      if pbar:
        pbar.update(1)
      return output


async def send_servomatic_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = True,
) -> RequestFuncOutput:
  """Sends request to a servomatic with or without streaming, depending on the value of stream parameter."""
  assert request_input.stub is not None

  async with sem:
    output = RequestFuncOutput()
    output.backend = request_input.backend
    output.model = request_input.model
    output.prompt_len = request_input.prompt_len
    output.requested_output_len = request_input.output_len
    output.generated_token_ids = []
    output.itl = []

    request = predict_pb2.PredictRequest()  # pylint: disable=undefined-variable
    request.model_spec.name = request_input.model  # servo model name
    request.model_spec.signature_name = "serving_default"

    tokens_tensor = tf.make_tensor_proto(  # pylint: disable=undefined-variable
        [request_input.tokenized_prompt],
        dtype=tf.int32,  # pylint: disable=undefined-variable
        shape=[1, len(request_input.tokenized_prompt)],
    )

    variant_tensor_data = tensor_pb2.VariantTensorDataProto(  # pytype: disable=wrong-arg-types # pylint: disable=undefined-variable
        type_name=b"tensorflow::Tensor",
        tensors=[tokens_tensor],
    )

    tensor_proto = tensor_pb2.TensorProto(  # pylint: disable=undefined-variable
        dtype=tf.variant.as_datatype_enum,  # pylint: disable=undefined-variable
        tensor_shape=tf.TensorShape([1]).as_proto(),  # pylint: disable=undefined-variable
        variant_val=[variant_tensor_data],
    )

    request.inputs["tokens"].CopyFrom(tensor_proto)
    request.inputs["temperature"].CopyFrom(
        tf.make_tensor_proto(  # pylint: disable=undefined-variable
            [0.0],
            dtype=tf.float32,  # pylint: disable=undefined-variable
            shape=[1],
        )
    )

    request.inputs["per_example_max_decode_steps"].CopyFrom(
        tf.make_tensor_proto(  # pylint: disable=undefined-variable
            [request_input.output_len],
            dtype=tf.int32,  # pylint: disable=undefined-variable
            shape=[1],
        )
    )

    if stream:
      stream_predict = request_input.stub.CreatePredictStreamed()
      previous_token_timestamp = 0.0
      event = asyncio.Event()
      try:
        def on_message(curr_res):
          if not stream or not curr_res:
            return
          timestamp = time.perf_counter()
          nonlocal previous_token_timestamp
          # Processing the very first response token.
          if not output.ttft:
            output.ttft = timestamp - request_start_time
          else:
            output.itl.append(timestamp - previous_token_timestamp)
          previous_token_timestamp = timestamp
          if curr_res.outputs["topk_ids"]:
            output.generated_token_ids.extend(
                curr_res.outputs["topk_ids"].int_val
            )

        def on_done():
          status = stream_predict.GetStatus()
          output.latency = time.perf_counter() - request_start_time
          output.success = status.ok()
          if not output.success:
            output.error = status.error_message()
            logging.warning(
                "Error while processing streaming response: %s", output.error)
          stream_predict.Shutdown()
          event.set()

        stream_predict.Start(on_message, on_done)
        request_start_time = time.perf_counter()
        stream_predict.SendAndHalfClose(request)
        while not event.is_set():
          await asyncio.sleep(0.001)

      except Exception:  # pylint: disable=broad-except
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        logging.warning(output.error)

      await event.wait()
    else:
      request_start_time = time.perf_counter()
      try:
        response = await async_stubby.AsyncStub(request_input.stub).Predict(  # pylint: disable=undefined-variable
            request
        )
        output.latency = time.perf_counter() - request_start_time

        if response.outputs["topk_ids"]:
          output.generated_token_ids = response.outputs["topk_ids"].int_val
          output.success = True
        else:
          output.success = False
          logging.warning("Error while processing unariy response: %s",
                          response)
      except Exception:  # pylint: disable=broad-except
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        logging.warning(output.error)

    if pbar:
      pbar.update(1)
    return output


async def send_evergreen_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = True,
) -> RequestFuncOutput:
  """Sends request to a evergreen backend."""
  assert request_input.stub is not None
  assert stream

  async with sem:
    output = RequestFuncOutput()
    output.backend = request_input.backend
    output.model = request_input.model
    output.prompt_len = request_input.prompt_len
    output.requested_output_len = request_input.output_len
    output.generated_token_ids = []
    output.itl = []

    previous_token_timestamp = 0.0
    client = request_input.stub
    request_start_time = time.perf_counter()
    try:
      async for content in client.generate_stream(request_input.prompt):
        if not content.as_tokenized_text().tokens.token_ids:
          continue
        output.generated_token_ids.extend(
            content.as_tokenized_text().tokens.token_ids
        )
        timestamp = time.perf_counter()
        if not output.ttft:
          output.ttft = timestamp - request_start_time
        else:
          output.itl.append(timestamp - previous_token_timestamp)
        previous_token_timestamp = timestamp
    except Exception:  # pylint: disable=broad-except
      output.success = False
      output.error = "".join(traceback.format_exception(*sys.exc_info()))
      logging.warning(output.error)

    output.latency = time.perf_counter() - request_start_time
    output.success = True
    if pbar:
      pbar.update(1)
    return output


async def send_request(
    request_input: RequestFuncInput,
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
    stream: Optional[bool] = False,
):
  """Sends request to server."""
  if not stream:
    stream = False
  assert not stream
  async with sem:
    output = RequestFuncOutput()
    output.backend = request_input.backend
    output.model = request_input.model
    output.success = False
    output.prompt_len = request_input.prompt_len

    headers = {"User-Agent": "Benchmark Client"}
    if request_input.backend == "vllm":
      logging.debug("sending %d tokens", request_input.output_len)
      pload = {
          "prompt": request_input.prompt,
          "n": 1,
          "best_of": request_input.best_of,
          "use_beam_search": request_input.use_beam_search,
          "temperature": 0.0 if request_input.use_beam_search else 1.0,
          "top_p": 1.0,
          "max_tokens": request_input.output_len,
          "ignore_eos": True,
          "stream": False,
      }
    elif request_input.backend == "tgi":
      assert not request_input.use_beam_search
      params = {
          "best_of": request_input.best_of,
          "max_new_tokens": request_input.output_len,
          "do_sample": True,
      }
      pload = {
          "inputs": request_input.prompt,
          "parameters": params,
      }
    elif request_input.backend == "naive_transformers":
      # If max_length or top_k is not specified _MAX_LENGTH_DEFAULT = 200 and
      # _TOP_K_DEFAULT = 10 in peft/handler.py will be used.
      pload = {
          "instances": [{
              "prompt": request_input.prompt,
              "max_length": request_input.output_len,
              "top_k": request_input.top_k,
          }]
      }
    elif request_input.backend == "tensorrt_llm_triton":
      pload = {
          "text_input": request_input.prompt,
          "max_tokens": request_input.output_len,
          "beam_width": (
              1 if not request_input.use_beam_search else request_input.best_of
          ),
          "temperature": 0.0 if request_input.use_beam_search else 1.0,
          "top_p": 1.0,
          "bad_words": "",
          "stop_words": "",
          "stream": False,
      }
    elif request_input.backend == "sax":
      pload = {
          "model": request_input.model,
          "prompt": request_input.prompt,
          "n": 1,
          "best_of": request_input.best_of,
          "use_beam_search": request_input.use_beam_search,
          "temperature": 0.0 if request_input.use_beam_search else 1.0,
          "top_p": 1.0,
          "top_k": 50,
          "max_tokens": request_input.output_len,
          "stream": False,
      }
    else:
      raise ValueError(f"Unknown backend: {request_input.backend}")

    request_start_time = time.perf_counter()
    # Set client timeout to be 3 hrs.
    timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
      while True:
        async with session.post(
            request_input.api_url, headers=headers, json=pload
        ) as response:
          chunks = []
          async for chunk, _ in response.content.iter_chunks():
            chunks.append(chunk)
        generated_text = b"".join(chunks).decode("utf-8")
        result_json = json.loads(generated_text)
        output.success = True

        # Re-send the request if it failed.
        if "error" not in result_json:
          break

    output.latency = time.perf_counter() - request_start_time
    # Naive HF transformers generation and TensorRT-LLM generation stops at EOS
    # tokens and the generation may be shorter than the ground-truth output
    # sequence length.
    if request_input.backend == "naive_transformers":
      complete_pred = result_json["predictions"][0][0]["generated_text"]
      new_text_start_index = complete_pred.find(NEW_TEXT_KEY) + len(
          NEW_TEXT_KEY
      )
      output.generated_text = complete_pred[new_text_start_index:]
    elif request_input.backend == "tensorrt_llm_triton":
      output.generated_text = result_json["text_output"]
    elif request_input.backend == "sax":
      output.generated_text = result_json["choices"][0]["text"]
    elif request_input.backend == "tgi":
      output.generated_text = result_json["generated_text"]
    elif request_input.backend == "vllm":
      output.generated_text = result_json["predictions"][0]

    if pbar:
      pbar.update(1)
    return output


@dataclass
class BenchmarkMetrics:
  """Aggregated metrics for a benchmark run."""

  completed: int
  total_input: int
  total_output: int
  request_throughput: float
  input_throughput: float
  output_throughput: float
  mean_ttft_ms: Optional[float]
  median_ttft_ms: Optional[float]
  p99_ttft_ms: Optional[float]
  mean_tpot_ms: Optional[float]
  median_tpot_ms: Optional[float]
  p99_tpot_ms: Optional[float]
  mean_latency_ms: Optional[float]
  median_latency_ms: Optional[float]
  p99_latency_ms: Optional[float]


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    duration_sec: float,
    tokenizer: BaseTokenizer,
) -> tuple[BenchmarkMetrics, pd.DataFrame]:
  """Calculates the aggregated metrics for a benchmark run.

  Args:
    outputs: Benchmark outputs.
    duration_sec: Duration of the benchmark run.
    tokenizer: Tokenizer used for the benchmark.

  Returns:
    A BenchmarkMetrics.
    A dataframe with the detailed per-request benchmark results.
  """
  actual_output_lens = []
  total_input = 0
  completed = 0
  results = []
  tpots = []
  ttfts = []
  latencies = []

  for i in range(len(outputs)):
    dt = dataclasses.asdict(outputs[i])
    if outputs[i].success:
      dt.pop("generated_text")
      dt.pop("generated_token_ids")

      if outputs[i].generated_token_ids:
        output_len = len(outputs[i].generated_token_ids)
      else:
        output_len = len(tokenizer.encode(outputs[i].generated_text))
      if output_len != outputs[i].requested_output_len:
        logging.debug(
            "Output length mismatch: requested len: %d vs actual len:%d",
            outputs[i].requested_output_len,
            output_len,
        )

      if outputs[i].backend == "vllm":
        output_len -= outputs[i].prompt_len
      dt["output_len"] = output_len
      actual_output_lens.append(output_len)
      total_input += outputs[i].prompt_len
      completed += 1
      latencies.append(outputs[i].latency)
      if outputs[i].ttft:
        if output_len > 1:
          tpots.append(
              (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
        ttfts.append(outputs[i].ttft)
    else:
      dt["output_len"] = 0
      actual_output_lens.append(0)
    results.append(dt)

  metrics = BenchmarkMetrics(
      # number of successful requests
      completed=completed,
      # sum of input prompts length
      total_input=total_input,
      # sum of output length
      total_output=sum(actual_output_lens),
      # throughput requests / sec
      request_throughput=completed / duration_sec,
      # input throughput input tokens / sec
      input_throughput=total_input / duration_sec,
      # output throughtput output tokens / sec
      output_throughput=sum(actual_output_lens) / duration_sec,
      mean_ttft_ms=np.mean(ttfts or 0) * 1000 if ttfts else None,
      median_ttft_ms=np.median(ttfts or 0) * 1000 if ttfts else None,
      p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000 if ttfts else None,
      mean_tpot_ms=np.mean(tpots) * 1000 if tpots else None,
      median_tpot_ms=np.median(tpots) * 1000 if tpots else None,
      p99_tpot_ms=np.percentile(tpots, 99) * 1000 if tpots else None,
      mean_latency_ms=np.mean(latencies or 0) * 1000 if latencies else None,
      median_latency_ms=np.median(latencies or 0) * 1000 if latencies else None,
      p99_latency_ms=np.percentile(latencies or 0, 99) * 1000
      if latencies
      else None,
  )

  return metrics, pd.DataFrame.from_dict(results)  # pytype: disable=wrong-arg-types


def get_images(images: str, inline_image: bool) -> List[str]:
  """Returns a list of PIL images."""
  images_list = []
  for image in images.split(","):
    image = image.strip()
    if "?" in image or "*" in image:
      if image.startswith("gs://"):
        import tensorflow as tf  # pylint: disable=g-import-not-at-top

        images_list.extend(tf.io.gfile.glob(image))
      else:
        import glob  # pylint: disable=g-import-not-at-top

        images_list.extend(glob.glob(image))
    else:
      images_list.append(image)
  if not inline_image:
    return images_list

  def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_str

  inlined_images_list = []
  for image in images_list:
    if image.startswith("gs://"):
      import tensorflow as tf  # pylint: disable=g-import-not-at-top

      with tf.io.gfile.GFile(image, "rb") as f:
        image_bytes = f.read()
    elif image.startswith("http://") or image.startswith("https://"):
      response = requests.get(image)
      image_bytes = response.content
    else:
      with open(image, "rb") as f:
        image_bytes = f.read()

    base64_image = image_to_base64(Image.open(io.BytesIO(image_bytes)))
    image_uri = f"data:image/jpeg;base64,{base64_image}"
    inlined_images_list.append(image_uri)
  return inlined_images_list


async def benchmark(
    args: argparse.Namespace,
    api_urls: List[str],
    input_requests: List[Tuple[str, int, int]],
    tokenizer: BaseTokenizer,
    prefix: str,
    max_input: int,
    max_output: int,
    concurrent_requests: Optional[int] = None,
    cache_hit_ratio: Optional[float] = None,
):
  """Runs benchmark with asynchronous requests."""
  print(
      f"Running benchmark for {args.backend}, max input: {max_input}, max"
      f" output: {max_output}, concurrent requests: {concurrent_requests},"
      f" request rate: {args.request_rate}, fixed qps: {args.fixed_qps}, cache"
      f" hit rate: {cache_hit_ratio}"
  )
  tasks: List[asyncio.Task] = []
  pbar = tqdm(total=len(input_requests))
  stub = None

  if googleinit:
    from google3.learning.serving.apis import prediction_service_pb2  # pylint: disable=g-import-not-at-top
    from google3.net.rpc2.contrib.smartservice.python import smartservice_util  # pylint: disable=g-import-not-at-top

    global tf, async_stubby, tensor_pb2, predict_pb2
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
    from google3.net.rpc.python.contrib import async_stubby  # pylint: disable=g-import-not-at-top
    from google3.third_party.tensorflow.core.framework import tensor_pb2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    from google3.third_party.tensorflow_serving.apis import predict_pb2  # pylint: disable=g-import-not-at-top
    from google3.learning.deepmind.evergreen.model_access.client.python import model_client  # pylint: disable=g-import-not-at-top

    if args.backend == "servomatic":
      smart_service = smartservice_util.parse(api_urls[0])
      stub = smartservice_util.new_stub(
          prediction_service_pb2.PredictionService, smart_service
      )
    elif args.backend == "evergreen":
      stub = model_client.ModelClient(
          model_url=api_urls[0],
          default_config=model_client.make_generation_config(
              seed=0,
              formatting_options=model_client.FormattingOptions(
                  enable_formatting=False
              ),
              token_generation=model_client.make_token_generation_config(
                  length=max_output,
                  sampling_config=model_client.make_sampling_config(
                      temperature=0,
                  ),
              ),
          ),
      )
    # Ensure stubby is fully initialized. This is required so that first request
    # execution is not delayed.
    await asyncio.sleep(10.0)

  benchmark_start_time = time.perf_counter()
  start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
  sem = (
      asyncio.Semaphore(concurrent_requests)
      if concurrent_requests
      else asyncio.Semaphore(len(input_requests))
  )
  async for request in get_request(
      input_requests,
  ):
    images = []
    if len(request) == 3:
      prompt, prompt_len, output_len = request
    else:
      prompt, prompt_len, output_len, images = request  # pytype: disable=bad-unpacking
    request_extra_body = None
    if args.request_extra_body is not None:
      try:
        request_extra_body = json.loads(args.request_extra_body)
      except json.decoder.JSONDecodeError:
        request_extra_body = args.request_extra_body
    api_url = random.choice(api_urls)
    logging.debug("api url: %s", api_url)
    request_input = RequestFuncInput(
        backend=args.backend,
        api_url=api_url,
        prompt=prompt,
        prompt_len=prompt_len,
        output_len=output_len,
        best_of=args.best_of,
        use_beam_search=args.use_beam_search,
        top_k=args.top_k,
        model=args.model,
        extra_body=request_extra_body,
        images=images
    )
    if args.backend == "chat_completions":
      request_func = send_chat_completions_request
    elif args.backend == "trt_llm_stream":
      request_func = send_trt_llm_stream_request
    elif args.backend == "vllm_stream":
      request_func = send_vllm_stream_request
    elif args.backend == "hex_llm":
      request_func = send_hex_llm_request
    elif args.backend == "servomatic":
      # tokenize prompt before sending request to servomatic.
      request_input.tokenized_prompt = list(
          tokenizer.encode(request_input.prompt)
      )
      request_input.stub = stub
      request_func = send_servomatic_request
    elif args.backend == "evergreen":
      request_input.tokenized_prompt = list(
          tokenizer.encode(request_input.prompt)
      )
      request_input.stub = stub
      request_func = send_evergreen_request
    else:
      request_func = send_request
    task = asyncio.create_task(
        request_func(
            request_input,
            sem,
            pbar,
            args.stream,
        )
    )
    if args.fixed_qps is not None:
      # await here would force task to start when running in fixed_qps mode.
      await asyncio.sleep(1.0 / args.fixed_qps)
    tasks.append(task)
  outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
  duration_sec = time.perf_counter() - benchmark_start_time
  end_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

  if pbar is not None:
    pbar.close()

  metrics, full_results = calculate_metrics(outputs, duration_sec, tokenizer)
  if concurrent_requests:
    full_results = full_results.assign(
        concurrent_requests=concurrent_requests,
    )
  elif args.fixed_qps:
    full_results = full_results.assign(
        fixed_qps=args.fixed_qps,
    )
  else:
    full_results = full_results.assign(
        request_rate=args.request_rate,
    )

  full_results = full_results.assign(
      max_input=max_input,
      max_output=max_output,
  )

  if args.save_full_results:
    f = open(
        os.path.join(
            args.output_dir if args.output_dir else os.getcwd(),
            f"{prefix}_full_results.json",
        ),
        mode="a",
    )
    f.write(full_results.to_json(orient="records", lines=True))
    f.close()

  print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
  print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
  print("{:<40} {:<10.2f}".format("Benchmark duration (s):", duration_sec))
  print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
  print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
  print(
      "{:<40} {:<10}".format(
          "Average input length:", metrics.total_input / metrics.completed
      )
  )
  print(
      "{:<40} {:<10}".format(
          "Average output length:", metrics.total_output / metrics.completed
      )
  )
  print(
      "{:<40} {:<10.3f}".format(
          "Request throughput (req/s):", metrics.request_throughput
      )
  )
  print(
      "{:<40} {:<10.2f}".format(
          "Input token throughput (tok/s):", metrics.input_throughput
      )
  )
  print(
      "{:<40} {:<10.2f}".format(
          "Output token throughput (tok/s):", metrics.output_throughput
      )
  )
  if metrics.mean_ttft_ms:
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    if metrics.mean_tpot_ms:
      print(
          "{s:{c}^{n}}".format(
              s="Time per Output Token (excl. 1st token)", n=50, c="-"
          )
      )
      print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
      print(
          "{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms)
      )
      print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
  if metrics.mean_latency_ms:
    print("{s:{c}^{n}}".format(s="Latencies", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean Latency (ms):", metrics.mean_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median Latency (ms):", metrics.median_latency_ms
        )
    )
    print(
        "{:<40} {:<10.2f}".format("P99 Latency (ms):", metrics.p99_latency_ms)
    )
    print("=" * 50)
  result = {
      "backend": args.backend,
      "start": start_time,
      "end": end_time,
      "duration": duration_sec,
      "completed": metrics.completed,
      "total_input_tokens": metrics.total_input,
      "total_output_tokens": metrics.total_output,
      "request_throughput": metrics.request_throughput,
      "input_throughput": metrics.input_throughput,
      "output_throughput": metrics.output_throughput,
  }
  if metrics.mean_ttft_ms:
    result |= {
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
    }
  return result


def prepend_unique_token(
    input_requests: List[Tuple[str, int, int]], tokenizer: BaseTokenizer
) -> List[Tuple[str, int, int]]:
  """Prepends a unique token to the input requests to avoid cache hit."""
  updated_input_requests = []
  for prompt, _, output_len in input_requests:
    unique_token_id = next_valid_token_id(tokenizer)
    unique_token = tokenizer.decode([unique_token_id])
    # In case the prompt starts with BOS token, we insert the unique token
    # after the BOS token.
    if prompt.startswith(tokenizer.bos_token()):
      updated_prompt = (
          prompt[: len(tokenizer.bos_token())]
          + " "
          + unique_token
          + " "
          + prompt[len(tokenizer.bos_token()) :]
      )
    # Otherwise, we insert the unique token at the beginning of the prompt.
    else:
      updated_prompt = unique_token + " " + prompt
    updated_input_requests.append((
        updated_prompt,
        len(tokenizer.encode(updated_prompt, add_special_tokens=False)),
        output_len,
    ))
  return updated_input_requests


def pad_to_multiple(x: int, multiple: int) -> int:
  assert x > 0
  return x + (-x % multiple)


async def cache_requests(
    args: argparse.Namespace,
    input_requests: List[Tuple[str, int, int]],
    cache_hit_ratio: float,
    tokenizer: BaseTokenizer,
    api_url: str,
):
  """Caches the requests with the given cache hit ratio.

  We warm up the cache by sending the requests with at least cache_hit_ratio of
  the total request length to make sure the following benchmark runs to hit
  at least `cache_hit_ratio`.

  Args:
    args: The benchmark arguments.
    input_requests: The input requests to cache.
    cache_hit_ratio: The cache hit ratio to warm up the cache.
    tokenizer: The tokenizer to encode the prompts.
    api_url: The API URL to send the requests.

  Returns:
    None.
  """
  if cache_hit_ratio <= 0.0:
    return
  print("Warm up cache...")
  pbar = tqdm(total=len(input_requests))
  cached_requests = []
  for prompt, _, _ in input_requests:
    prompt_in_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_in_tokens)
    padded_prompt_len = pad_to_multiple(prompt_len, args.prefill_len_padding)
    cache_prompt_len = math.ceil(padded_prompt_len * cache_hit_ratio)
    padded_cache_prompt_len = pad_to_multiple(
        cache_prompt_len, args.prefill_len_padding
    )
    padded_cache_prompt_len = min(padded_cache_prompt_len, prompt_len)
    cached_prompt = tokenizer.decode(prompt_in_tokens[:padded_cache_prompt_len])
    cached_requests.append((cached_prompt, padded_cache_prompt_len, 1))

  tasks: List[asyncio.Task] = []
  sem = asyncio.Semaphore(len(cached_requests))
  async for request in get_request(
      cached_requests,
  ):

    prompt, prompt_len, output_len = request
    request_input = RequestFuncInput(
        backend=args.backend,
        api_url=api_url,
        prompt=prompt,
        prompt_len=prompt_len,
        output_len=output_len,
        model=args.model,
    )
    if args.backend == "chat_completions":
      request_func = send_chat_completions_request
    else:
      request_func = send_hex_llm_request
    task = asyncio.create_task(
        request_func(
            request_input,
            sem,
            pbar,
            args.stream,
        )
    )
    tasks.append(task)
  await asyncio.gather(*tasks)
  if pbar is not None:
    pbar.close()
  print("Warm up cache done.")


def maybe_update_requests(
    input_requests: List[Tuple[str, int, int]],
    tokenizer: BaseTokenizer,
) -> List[Tuple[str, int, int]]:
  """Updates the requests to avoid cache miss.

  Some tokenizers (e.g. Mistral Tokenizer) will add an extra space after BOS
  token. In other words, decode(encode(x)) != x. This will cause cache miss.
  Since `cache_requests` calls encode and decode for cached prompt. We should
  update the `updated_input_requests` to avoid cache miss.

  Args:
    input_requests: The input requests to update.
    tokenizer: The tokenizer to encode the prompts.

  Returns:
    The updated input requests.
  """
  updated_input_requests = []
  for prompt, _, output_len in input_requests:
    updated_prompt = tokenizer.decode(
        tokenizer.encode(prompt, add_special_tokens=False)
    )
    updated_input_requests.append((
        updated_prompt,
        len(tokenizer.encode(updated_prompt, add_special_tokens=False)),
        output_len,
    ))
  return updated_input_requests


def main(args: argparse.Namespace):
  random.seed(args.seed)
  np.random.seed(args.seed)

  log_levels = {
      "debug": logging.DEBUG,
      "info": logging.INFO,
      "warning": logging.WARNING,
      "error": logging.ERROR,
      "critical": logging.CRITICAL
  }

  # Configure the logging
  logging.basicConfig(level=log_levels[args.verbosity])

  token_id_counter.reset(start=args.token_id_start)
  if args.cache_hit_ratio and args.backend not in [
      "chat_completions",
      "hex_llm",
  ]:
    raise ValueError(
        "Cache hit ratio is only supported for chat_completions and hex_llm"
        " backend."
    )

  endpoint = args.endpoint
  if not args.endpoint:
    if args.backend == "chat_completions":
      endpoint = "v1/chat/completions"
    else:
      endpoint = "generate"

  port_str = ":" + str(args.port) if args.port else ""
  protocol = "" if args.host.startswith("http") else "http://"
  api_url = f"{protocol}{args.host}{port_str}/{endpoint}"

  api_urls = []
  if args.endpoints:
    with open(args.endpoints, "r") as f:
      endpoints = f.readlines()
      for endpoint in endpoints:
        endpoint = endpoint.strip()
        api_url = f"{protocol}{args.host}{port_str}/{endpoint}"
        logging.debug("api url added to list: %s", api_url)
        api_urls.append(f"{api_url}")
  else:
    logging.debug("api url added to list: %s", api_url)
    api_urls.append(api_url)

  if googleinit:
    tokenizer = Llama3Tokenizer(args.tokenizer)
    # Servomatic backend has only BNS address. Evergreen backend only has mBNS.
    api_urls = [args.endpoint]
  else:
    tokenizer = GeneralTokenizer(args.tokenizer, args.trust_remote_code)

  prefix = args.name if args.name else args.backend
  fname = os.path.join(
      args.output_dir if args.output_dir else os.getcwd(),
      f"{prefix}_aggregated_results.json",
  )

  logging.info("preparing requests")
  for max_input in args.max_input_length:
    for max_output in args.max_output_length:
      if args.dataset.endswith("sonnet.txt"):
        min_input_len = int(max_input / 2)
        max_input_len = max_input + min_input_len
        min_output_len = int(max_output / 2)
        max_output_len = max_output + min_output_len
        input_requests = sample_sonnet_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            min_input_len=min_input_len,
            max_input_len=max_input_len,
            min_output_len=min_output_len,
            max_output_len=max_output_len,
            prefix_len=args.sonnet_prefix_len,
            tokenizer=tokenizer,
            fixed_input_length=(
                max_input if args.fixed_input_length else None
            ),
            fixed_output_length=(
                max_output if args.fixed_output_length else None
            ),
        )
        if (
            args.backend == "chat_completions"
            or args.backend == "servomatic"
            or args.backend == "evergreen"
        ):
          input_requests = [
              (prompt, prompt_len, output_len)
              for prompt, _, prompt_len, _, output_len in input_requests
          ]
        else:
          input_requests = [
              (prompt_formatted, prompt_formatted_len, output_len)
              for _, prompt_formatted, _, prompt_formatted_len, output_len in input_requests
          ]
      else:
        input_requests = sample_sharedgpt_requests(
            args.dataset,
            args.num_prompts,
            max_input,
            max_output,
            tokenizer,
            args.use_dummy_text,
            args.only_human_prompts,
            args.fixed_output_length,
        )
      if args.images:
        images = get_images(args.images, args.inline_images)
        input_requests_with_images = []
        for input_request_tuple in input_requests:
          input_requests_with_images.append((
              *input_request_tuple,
              random.sample(images, args.images_per_request),
          ))
        input_requests = input_requests_with_images

      logging.info("staring benchmark")
      c_list = args.c
      if c_list is None:
        c_list = [None]
      cache_hit_ratio_list = (
          args.cache_hit_ratio if args.cache_hit_ratio else [None]
      )
      for concurrent_requests in c_list:
        for cache_hit_ratio in cache_hit_ratio_list:
          updated_input_requests = []
          if cache_hit_ratio is not None:
            # Prepend a unique token to the input requests to avoid cache hit.
            updated_input_requests = prepend_unique_token(
                input_requests, tokenizer
            )
            # Cache the requests to make sure the following benchmark runs to
            # hit >=cache_hit_ratio.
            asyncio.run(
                cache_requests(
                    args,
                    updated_input_requests,
                    cache_hit_ratio,
                    tokenizer,
                    api_url,
                )
            )
            # Some tokenizers (e.g. Mistral Tokenizer) will add an extra space
            # after BOS token. In other words, decode(encode(x)) != x. This will
            # cause cache miss.
            # Since `cache_requests` calls encode and decode for cached prompt.
            # We should update the `updated_input_requests` to avoid cache miss.
            updated_input_requests = maybe_update_requests(
                updated_input_requests, tokenizer
            )
          results = asyncio.run(
              benchmark(
                  args,
                  api_urls,
                  updated_input_requests
                  if updated_input_requests
                  else input_requests,
                  tokenizer,
                  prefix,
                  max_input,
                  max_output,
                  concurrent_requests,
                  cache_hit_ratio,
              )
          )
          print(f"results: {results}")

          bm_configs = dict(vars(args).copy())
          bm_configs.pop("save_full_results")
          bm_configs.pop("c")
          bm_configs.pop("max_input_length")
          bm_configs.pop("max_output_length")
          bm_configs["max_input_len"] = max_input
          bm_configs["max_output_len"] = max_output
          if concurrent_requests is not None:
            bm_configs["concurrent_requests"] = concurrent_requests
            bm_configs.pop("request_rate")
            bm_configs.pop("fixed_qps")
          if cache_hit_ratio is not None:
            bm_configs["cache_hit_ratio"] = cache_hit_ratio
          results = results | bm_configs
          df = pd.DataFrame([results])
          f = open(fname, mode="a")
          f.write(df.to_json(orient="records", lines=True))
          f.close()
  print(f"Saved results to {fname}")


if __name__ == "__main__":
  if googleinit:
    # required for running in google3 environment.
    googleinit.Run(sys.argv[:1])  # pytype: disable=wrong-arg-types

  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--backend",
      type=str,
      default="chat_completions",
      choices=[
          "vllm",
          "tgi",
          "naive_transformers",
          "tensorrt_llm_triton",
          "sax",
          "chat_completions",
          "trt_llm_stream",
          "vllm_stream",
          "servomatic",
          "hex_llm",
          "evergreen",
      ],
  )
  parser.add_argument(
      "--model",
      type=str,
      default="",
      help="Model name to send request to at API server.",
  )
  parser.add_argument("--endpoint", type=str, default=None)
  parser.add_argument("--host", type=str, default="localhost")
  parser.add_argument("--port", type=int, default=None)
  parser.add_argument("--dataset", type=str, help="Path to the dataset.")
  parser.add_argument(
      "--endpoints",
      type=str,
      default=None,
      help="Path to a file containing a list of endpoints.",
  )
  parser.add_argument(
      "--only-human-prompts",
      type=bool,
      default=False,
      help="If true, filter out prompts that are not started by a human.",
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      required=True,
      help="Name or path of the tokenizer.",
  )
  parser.add_argument(
      "--stream",
      type=str2bool,
      default=None,
      help="Whether to uses streaming API.",
  )
  parser.add_argument(
      "--save-full-results",
      type=bool,
      default=False,
      help="Whether to save the full (per request) results.",
  )
  parser.add_argument(
      "--output-dir",
      type=str,
      default=None,
      help=(
          "Directory to the output result file otherwise current directory is"
          " used."
      ),
  )
  parser.add_argument(
      "--best-of",
      type=int,
      default=1,
      help="Generates `best_of` sequences per prompt and returns the best one.",
  )
  parser.add_argument("--use-beam-search", action="store_true")
  parser.add_argument(
      "--num-prompts",
      type=int,
      default=1000,
      help="Number of prompts to process.",
  )

  def _list_of_ints(arg: str) -> List[int]:
    return list(map(int, arg.split(",")))

  parser.add_argument(
      "--max-input-length",
      type=_list_of_ints,
      default=[1024],
      help=(
          "Maximum number of input tokens for filtering the benchmark dataset."
          " This argument can be a list of integers separated by ','."
      ),
  )
  parser.add_argument(
      "--fixed-input-length",
      type=str2bool,
      default=False,
      help="If true, force the input length to be --max-input-length.",
  )
  parser.add_argument(
      "--max-output-length",
      type=_list_of_ints,
      default=[1024],
      help=(
          "Maximum number of input tokens for filtering the benchmark dataset."
          " This argument can be a list of integers separated by ','"
      ),
  )
  parser.add_argument(
      "--fixed-output-length",
      type=str2bool,
      default=False,
      help="If true, force the output length to be --max-output-length.",
  )
  parser.add_argument(
      "--sonnet-prefix-len",
      type=int,
      default=30,
      help="Number of prefix tokens per request, used only for sonnet dataset.",
  )
  parser.add_argument(
      "--top-k",
      type=int,
      default=32000,
      help=(
          "Number of candidate tokens that are considered at each step of the"
          " generation process. 32000 is the vocab_size of Open-LLaMA and"
          " LLaMA2 models."
      ),
  )
  parser.add_argument(
      "--c",
      "--concurrent-requests",
      type=_list_of_ints,
      default=None,
      help=(
          "The number of concurrent requests to send., This argument can be a"
          " list of integers separated by ','"
      ),
  )
  parser.add_argument(
      "--request-rate",
      type=float,
      default=float("inf"),
      help=(
          "If this is inf, all requests are sent at time 0. Otherwise, we take"
          " 1 divided by this argument value to be the parameter of the Poisson"
          " distribution for modeling the request arrival times. Ignored if"
          " --concurrent-requests is set."
      ),
  )
  parser.add_argument(
      "--fixed-qps",
      type=float,
      help=(
          "Number of requests per second sent with equal intervals. If this"
          " argument is set, we ignore request_rate and use a fixed QPS for"
          " sending the requests. Ignored if --concurrent-requests is set."
      ),
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--trust-remote-code",
      action="store_true",
      help="trust remote code from huggingface",
  )
  parser.add_argument(
      "--machine-cost",
      type=float,
      default=None,
      help="Machine cost per hour including accelerators (if any)",
  )
  parser.add_argument(
      "--use-dummy-text",
      action="store_true",
      help=(
          "Whether to use dummy text with length defined by max_input_length"
          " and max_output_length."
      ),
  )

  def _list_of_float(arg: str) -> List[float]:
    return list(map(float, arg.split(",")))

  parser.add_argument(
      "--cache-hit-ratio",
      type=_list_of_float,
      default=[],
      help=(
          "A list of prefix cache ratio the benchmark request is manipulated to"
          " hit."
      ),
  )
  parser.add_argument(
      "--prefill-len-padding",
      type=int,
      default=1,
      help=(
          "Pad the prefill sequence length to be a multiple of this number."
          " This would impact the actual cache hit ratio on some accelerators"
          " like TPU."
      ),
  )
  parser.add_argument(
      "--token-id-start",
      type=int,
      default=0,
      help="Start token id for the unique token id counter.",
  )
  parser.add_argument(
      "--name",
      type=str,
      default="",
      help=(
          "The name of the benchmark. Will be used as the prefix of the saved"
          " results files."
      ),
  )
  # pylint: disable=line-too-long
  parser.add_argument(
      "--request-extra-body",
      type=str,
      default="",
      help=(
          'Extra body to send with request. To disable LLamaGuard, set it to: '
          '\'{"google": { "model_safety_settings": {"enabled": False, "llama_guard_settings": {}}}}\''
      ),
  )
  parser.add_argument(
      "--images",
      type=str,
      default="",
      help=("Image list or use, or gcs glob"),
  )
  parser.add_argument(
      "--inline-images",
      type=str2bool,
      default="false",
      help=("Whether to inline images using base64 encoding"),
  )
  parser.add_argument(
      "--images-per-request",
      type=int,
      default=1,
      help=(
          "Number of images to send in each request,"
          "only used if --images flag is specified."
      ),
  )
  parser.add_argument(
      "-v", "--verbosity", help="Set the logging level (default: %(default)s)",
      default="warning", choices=[
          "debug", "info", "warning", "error", "critical"])

  cmd_args = parser.parse_args()
  main(cmd_args)