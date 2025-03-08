# SPDX-License-Identifier: Apache-2.0
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena

TODO: Implement CustomDataset to parse a JSON file and convert its 
contents into  SampleRequest instances, similar to the approach used
in ShareGPT.
"""

import base64
import io
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: str
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[MultiModalDataDict, dict]] = None
    lora_request: Optional[LoRARequest] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_NUM_REQUESTS = 1000
    DEFAULT_SEED = 0

    # num_requests has default 1000 in both the benchmark_serving.py and
    # benchmark_throughput.py

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        self.dataset_path = dataset_path
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.
        Subclasses must override this method; otherwise,
        NotImplementedError is raised.
        """
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self,
        tokenizer,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Return a tuple (lora_request, tokenizer) for tokenizing requests.  If
        LoRA is enabled, returns the LoRA-specific tokenizer; otherwise, the
        base tokenizer.
        """
        if max_loras is None or lora_path is None:
            return None, tokenizer

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, max_loras)
        lora_request = LoRARequest(
            lora_name=str(lora_id),
            lora_int_id=lora_id,
            lora_path=lora_path_on_disk(lora_path),
        )
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        return lora_request, lora_tokenizer_cache[lora_id]

    @abstractmethod
    def sample(self,
               tokenizer: PreTrainedTokenizerBase) -> list[SampleRequest]:
        """
        Generate sample requests from the dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------

VISION_ARENA_DATASET_PATH = "lmarena-ai/vision-arena-bench-v0.1"


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, AnyTokenizer] = {}


def process_image(image: Any, ) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    For a PIL.Image.Image input:
      - Converts the image to RGB.
      - Saves the image as a JPEG in-memory.
      - Encodes the JPEG data as a base64 string.
      - Returns a dictionary with the image as a base64 data URL.

    For a string input:
      - Treats the string as a URL or file path.
      - Prepends "file://" if the string doesn't start with "http://" or "file://".
      - Returns a dictionary with the image URL.

    Raises:
      ValueError: If the input is neither a PIL.Image.Image nor a string.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

    if isinstance(image, str):
        image_url = (image if image.startswith(
            ("http://", "file://")) else f"file://{image}")
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image or str.")


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 1.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               prefix_len: int = DEFAULT_PREFIX_LEN,
               range_ratio: float = DEFAULT_RANGE_RATIO,
               input_len: int = DEFAULT_INPUT_LEN,
               output_len: int = DEFAULT_OUTPUT_LEN,
               num_requests: int = DEFAULT_NUM_REQUESTS,
               **kwargs) -> list[SampleRequest]:
        prefix_len = (prefix_len
                      if prefix_len is not None else self.DEFAULT_PREFIX_LEN)
        range_ratio = (range_ratio if range_ratio is not None else
                       self.DEFAULT_RANGE_RATIO)
        input_len = (input_len
                     if input_len is not None else self.DEFAULT_INPUT_LEN)
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        num_requests = (num_requests if num_requests is not None else
                        self.DEFAULT_NUM_REQUESTS)

        vocab_size = tokenizer.vocab_size

        prefix_token_ids = (np.random.randint(
            0, vocab_size, size=prefix_len).tolist() if prefix_len > 0 else [])

        input_low = int(input_len * range_ratio)
        output_low = int(output_len * range_ratio)

        input_lens = np.random.randint(input_low,
                                       input_len + 1,
                                       size=num_requests)
        output_lens = np.random.randint(output_low,
                                        output_len + 1,
                                        size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)

        requests = []
        for i in range(num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) %
                         vocab_size).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            total_input_len = prefix_len + int(input_lens[i])
            requests.append(
                SampleRequest(
                    prompt,
                    total_input_len,
                    int(output_lens[i]),
                ))
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.shuffle(self.data)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               lora_path: Optional[str] = None,
               max_loras: Optional[int] = None,
               num_requests: int = DEFAULT_NUM_REQUESTS,
               output_len: Optional[int] = None,
               **kwargs) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt = entry["conversations"][0]["value"]
            completion = entry["conversations"][1]["value"]

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            output_len = (len(completion_ids)
                          if output_len is None else output_len)
            if not is_valid_sequence(
                    prompt_len,
                    output_len,
                    skip_min_output_len_check=output_len is not None
                    and output_len > 0,
            ):
                continue
            samples.append(
                SampleRequest(
                    prompt,
                    prompt_len,
                    output_len,
                    lora_request,
                ))
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(self,
               tokenizer,
               prefix_len: int = DEFAULT_PREFIX_LEN,
               input_len: int = DEFAULT_INPUT_LEN,
               output_len: int = DEFAULT_OUTPUT_LEN,
               num_requests: int = DEFAULT_NUM_REQUESTS,
               return_prompt_formatted: bool = False,
               **kwargs) -> list:
        # Calculate average token length for a poem line.
        prefix_len = (self.DEFAULT_PREFIX_LEN
                      if prefix_len is None else prefix_len)
        input_len = (self.DEFAULT_INPUT_LEN
                     if input_len is None else input_len)
        output_len = (self.DEFAULT_OUTPUT_LEN
                      if output_len is None else output_len)

        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens)
                      for tokens in \
                        tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(base_msg,
                                                 add_generation_prompt=True,
                                                 tokenize=False)
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset}).")

        # Determine how many poem lines to use.
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = round((prefix_len - base_offset) / avg_len)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        for _ in range(num_requests):
            extra_lines = random.choices(self.data,
                                         k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            samples.append(
                SampleRequest(
                    prompt_formatted if return_prompt_formatted else prompt,
                    prompt_len,
                    output_len,
                ))
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self, ):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        self.data = gpt4_df

    def _sample_loaded_data(self, num_requests: int = DEFAULT_NUM_REQUESTS):
        if num_requests <= len(self.data):
            data = self.data.sample(n=num_requests,
                                    random_state=self.random_seed)
        else:
            data = self.data.sample(
                n=num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        return data.values.tolist()

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int = DEFAULT_NUM_REQUESTS,
               max_loras: Optional[int] = None,
               lora_path: Optional[str] = None,
               **kwargs) -> list[SampleRequest]:
        samples = []
        data = self._sample_loaded_data(num_requests=num_requests)
        for i in range(num_requests):
            input_len = int(data[i][2])
            output_len = int(data[i][3])
            lora_req, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                SampleRequest(
                    prompt,
                    input_len,
                    output_len,
                    lora_request=lora_req,
                ))
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Implementation
# -----------------------------------------------------------------------------


class HuggingFaceDataset(BenchmarkDataset):
    """
    Dataset class for processing a HuggingFace dataset with conversation data
    and optional images.
    """
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(
        self,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset

        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided for loading data.")

        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )

        if "conversations" not in self.data.features:
            raise ValueError("HF Dataset must have a 'conversations' column.")

        # Shuffle and filter examples with at least 2 conversations.
        self.data = self.data.shuffle(seed=self.random_seed).filter(
            lambda x: len(x["conversations"]) >= 2)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int = DEFAULT_NUM_REQUESTS,
               lora_path: Optional[str] = None,
               max_loras: Optional[int] = None,
               output_len: Optional[int] = None,
               **kwargs) -> list:
        sampled_requests = []
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer, lora_path=lora_path, max_loras=max_loras)

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(
                    prompt_len, completion_len):
                continue

            mm_content = process_image(
                item["image"]) if "image" in item else None
            sampled_requests.append(
                SampleRequest(
                    prompt,
                    prompt_len,
                    output_len,
                    mm_content,
                    lora_request=lora_request,
                ))
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(BenchmarkDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    DEFAULT_NUM_REQUESTS = 1000

    def __init__(
        self,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset

        if self.dataset_path != VISION_ARENA_DATASET_PATH:
            raise ValueError(f"Only support Vision Arena dataset.\
                    This data path {self.dataset_path} is not valid.")
        if self.dataset_subset is None and self.dataset_split != "train":
            raise ValueError("Dataset split must be 'train'.")

        self.load_data()

    def load_data(self) -> None:
        dataset = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = dataset.shuffle(seed=self.random_seed)

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               output_len: int = DEFAULT_OUTPUT_LEN,
               num_requests: int = DEFAULT_NUM_REQUESTS,
               **kwargs) -> list:
        # TODO (jenniferzhao): Add support for offline benchmark sampling
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0][0]["content"]
            prompt_len = len(tokenizer(prompt).input_ids)
            mm_content = process_image(item["images"][0])
            sampled_requests.append(
                SampleRequest(
                    prompt,
                    prompt_len,
                    output_len,
                    mm_content,
                ))
        return sampled_requests
