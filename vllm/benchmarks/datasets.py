# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""
import ast
import base64
import io
import json
import logging
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from typing import Any, Callable, Optional, Union, cast

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerBase
from typing_extensions import deprecated

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.image import convert_image_mode
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer
from vllm.utils import PlaceholderModule

try:
    from datasets import load_dataset
except ImportError:
    datasets = PlaceholderModule("datasets")
    load_dataset = datasets.placeholder_attr("load_dataset")

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, list[str]]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[
        Union[MultiModalDataDict, dict, list[dict]]
    ] = None
    lora_request: Optional[LoRARequest] = None
    request_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  
        
        Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def apply_multimodal_chat_transformation(
            self,
            prompt: str,
            mm_content: Optional[
                        Union[MultiModalDataDict, dict, list[dict]]
                             ] = None) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            if isinstance(mm_content, list):
                content.extend(cast(list[dict[str, Any]], mm_content))
            elif isinstance(mm_content, dict):
                content.append(mm_content)
            else:
                raise TypeError(  
                    "Could not process multimodal content of type: " +
                    f"{type(mm_content)}"  
                ) 
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Optionally select a random LoRA request and return its associated
        tokenizer.

        This method is used when LoRA parameters are provided.  It randomly
        selects a LoRA based on max_loras and retrieves a cached tokenizer for
        that LoRA if available. Otherwise, it returns the base tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The base tokenizer to use if no
                LoRA is selected.
            max_loras (Optional[int]): The maximum number of LoRAs available.
                If `None`, LoRA is not used.
            lora_path (Optional[str]): Path to the LoRA parameters on disk.
                If `None`, LoRA is not used.

        Returns:
            A tuple with the following elements:
                - A new [LoRARequest][] (or `None` if not applicable).
                - The tokenizer associated with the LoRA request
                  (or the base tokenizer).
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
        # Return lora_request and the cached tokenizer if available; otherwise,
        # return the base tokenizer
        return lora_request, lora_tokenizer_cache[lora_id] or tokenizer

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int, 
               request_id_prefix: str = "") -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
                for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.
            request_id_prefix (str) The prefix of request_id.
            

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str = "",
    ) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
                requests.
            num_requests (int): The target number of requests.
            request_id_prefix (str) The prefix of the request ids.

        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = deepcopy(
                random.choices(requests, k=num_requests - len(requests))
            )
            for i in range(len(additional)):
                req = additional[i]
                req.request_id = request_id_prefix + str(len(requests) + i)
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


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


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and 'bytes' in image:
        image = Image.open(BytesIO(image['bytes']))
    if isinstance(image, Image.Image):
        image = convert_image_mode(image, "RGB")
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

    raise ValueError(f"Invalid image input {image}. Must be a PIL.Image.Image"
                     " or str or dictionary with raw image bytes.")


def process_video(video: Any) -> Mapping[str, Any]:
    """
    Process a single video input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw video bytes: - Expects a dict with a 'bytes' key
       containing raw video data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(video, dict) and 'bytes' in video:
        video_bytes = video['bytes']
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_base64}"
            },
        }

    if isinstance(video, str):
        video_url = (video if video.startswith(
            ("http://", "file://")) else f"file://{video}")
        return {"type": "video_url", "video_url": {"url": video_url}}

    raise ValueError(
        f"Invalid video input {video}. Must be a string of local path/remote url, or a dictionary with raw video bytes in the form of `{{'bytes': raw_video_bytes}}`."  # noqa: E501
    )

# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    """
    Synthetic text-only dataset for serving/throughput benchmarks.

    Strategy:
    - Sample input/output token lengths per request from integer-uniform ranges
      around configured means (controlled by range_ratio).
    - Prepend a fixed random prefix of length prefix_len.
    - Generate the remaining tokens as a reproducible sequence:
      (offset + index + arange(input_len)) % vocab_size.
    - Decode then re-encode/truncate to ensure prompt token counts match.
    - Uses numpy.default_rng seeded with random_seed for reproducible sampling.
    """
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Use numpy's default_rng for deterministic sampling
        # Do not use random.seed() or np.random.seed() elsewhere in this class.
        # This ensures that the RNG is isolated from global RNG state.
        self._rng = np.random.default_rng(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        **kwargs,
    ) -> list[SampleRequest]:

        input_lens, output_lens, offsets = self.get_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        # Generate prefix once
        prefix_token_ids = self.get_prefix(tokenizer, prefix_len)
        vocab_size = tokenizer.vocab_size

        requests = []
        for i in range(num_requests):
            prompt, total_input_len = self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
            )
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    request_id=request_id_prefix + str(i),
                )
            )
        # only used for embeddings benchmark.
        if batchsize > 1:
            batch_requests = []
            # Create batched requests
            for i in range(0, num_requests, batchsize):
                batch = requests[i : i + batchsize]
                batch_requests.append(
                    SampleRequest(
                        prompt=[req.prompt for req in batch],
                        prompt_len=sum(req.prompt_len for req in batch),
                        expected_output_len=0,
                        request_id=request_id_prefix + str(i // batchsize),
                    )
                )
            requests = batch_requests
        return requests

    def get_prefix(
        self, tokenizer: PreTrainedTokenizerBase, prefix_len: int
    ) -> list[int]:
        """
        Get the prefix for the dataset.
        """
        return (
            self._rng.integers(
                0, tokenizer.vocab_size, size=prefix_len).tolist()
            if prefix_len > 0
            else []
        )

    def get_sampling_params(
        self,
        num_requests: int,
        range_ratio: float,
        input_len: int,
        output_len: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the sampling parameters for the dataset.
        """
        # Enforce range_ratio < 1
        if not (0.0 <= range_ratio < 1.0):
            raise ValueError("range_ratio must be in [0, 1).")
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special_tokens)
        # Bounds use floor for low and ceil for high
        input_low = math.floor(real_input_len * (1 - range_ratio))
        input_high = math.ceil(real_input_len * (1 + range_ratio))
        output_low = math.floor(output_len * (1 - range_ratio))
        output_high = math.ceil(output_len * (1 + range_ratio))
        # Ensure the lower bound for output length is at least 1 to
        # prevent sampling 0 tokens.
        output_low = max(output_low, 1)

        if input_low > input_high:
            raise ValueError(
                "Invalid input sampling interval: "
                f"low={input_low} > high={input_high}"
            )
        if output_low > output_high:
            raise ValueError(
                "Invalid output sampling interval: "
                f"low={output_low} > high={output_high}"
            )

        logger.info(
            "Sampling input_len from [%s, %s] and output_len from [%s, %s]",
            input_low,
            input_high,
            output_low,
            output_high,
        )

        input_lens = self._rng.integers(input_low, input_high + 1,
                                           size=num_requests)
        output_lens = self._rng.integers(output_low, output_high + 1,
                                            size=num_requests)
        offsets = self._rng.integers(0, tokenizer.vocab_size, 
                                        size=num_requests)
        return input_lens, output_lens, offsets

    def generate_token_sequence(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        prefix_token_ids: list[int],
        prefix_len: int,
        vocab_size: int,
        input_len: int,
        offset: int,
        index: int,
    ) -> tuple[str, int]:
        """
        Returns (prompt, total_input_len).

        NOTE: After decoding the prompt we have to encode and decode it again.
        This is done because in some cases N consecutive tokens
        give a string tokenized into != N number of tokens.
        For example for GPT2Tokenizer:
        [6880, 6881] -> ['Ġcalls', 'here'] ->
        [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
        To avoid uncontrolled change of the prompt length,
        the encoded sequence is truncated before being decode again.
        """
        # Build the inner sequence by sampling sequentially from the vocab
        inner_seq = ((offset + index + np.arange(input_len)) 
                    % vocab_size).tolist()
        token_sequence = prefix_token_ids + inner_seq

        # Decode, then re-encode and truncate to preserve token count invariants
        prompt = tokenizer.decode(token_sequence)
        total_input_len = prefix_len + int(input_len)

        re_encoded_sequence = tokenizer.encode(
            prompt, add_special_tokens=False)[:total_input_len]
        prompt = tokenizer.decode(re_encoded_sequence)
        total_input_len = len(re_encoded_sequence)

        return prompt, total_input_len


# -----------------------------------------------------------------------------
# MultiModalDataset Implementation
# -----------------------------------------------------------------------------

class RandomMultiModalDataset(RandomDataset):
    """
    Synthetic multimodal dataset (text + images) that extends RandomDataset.

    Status:
    - Images: supported via synthetic RGB data.
    - Video: not yet supported (TODO: implement video generation method).
    - Audio: not yet supported.

    Sampling overview:
    1) Number of items per request is sampled uniformly from the integer range
       [floor(n·(1−r)), ceil(n·(1+r))], where n is the base count and r is
       `num_mm_items_range_ratio` in [0, 1]. r=0 keeps it fixed; r=1 allows 0.
       The maximum is further clamped to the sum of per-modality limits.
    2) Each item’s modality and shape is sampled from `bucket_config`, a dict
       mapping (height, width, num_frames) → probability. We treat 
       `num_frames`=1 as image and and `num_frames` > 1 as video. 
       Entries with zero probability are removed and the rest are renormalized 
       to sum to 1.
    3) Per-modality hard caps are enforced via `limit_mm_per_prompt`.
       When a modality reaches its cap, all of its buckets are excluded and the
       remaining probabilities are renormalized.

    Example bucket configuration:
    {(256, 256, 1): 0.5, (720, 1280, 1): 0.4, (720, 1280, 16): 0.1}
      - Two image buckets (`num_frames`=1) and one video bucket 
      (`num_frames`=16). 
    OBS.: Only image sampling is supported for now.
    """

    IS_MULTIMODAL = True
    # NOTE: video sampling is WIP. Setting it to 0.
    DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 255, "video": 0}

    DEFAULT_BASE_ITEMS_PER_REQUEST = 1
    DEFAULT_NUM_MM_ITEMS_RANGE_RATIO = 0.0
    DEFAULT_MM_ITEM_BUCKET_CONFIG = {
        (256, 256, 1): 0.5,
        (720, 1280, 1): 0.5,
        (720, 1280, 16): 0.0,
    }
    DEFAULT_ENABLE_MULTIMODAL_CHAT = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def generate_synthetic_image(self, width: int, height: int) -> Image.Image:
        """Generate synthetic PIL image with random RGB values.
        
        NOTE: iid pixel sampling results in worst-case compression 
        (good for stressing I/O), but very unlike real photos. 
        We could consider a “low-freq” mode (e.g., noise blur)
        to emulate network realism instead of max stress.
        """
        random_pixels = self._rng.integers(
            0,
            256,
            (height, width, 3),
            dtype=np.uint8,
        )
        return Image.fromarray(random_pixels)

    def generate_synthetic_video(self, width: int, 
                                    height: int, 
                                    num_frames: int) -> Any:
        """Generate synthetic video with random values.
        
        TODO: Finish this method.
        """
        raise NotImplementedError("Video sampling is WIP.")

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        """Map the configuration to the modality."""
        if config[-1] == 1:
            return "image"
        elif config[-1] > 1:
            return "video"
        else:
            raise ValueError(f"Invalid multimodal item configuration: {config}")

    def normalize_bucket_config(self, bucket_config: dict[tuple[int, int, int], 
                                float]) -> dict[tuple[int, int, int], float]:
        """
        Remove zero probability entries
        and normalize the bucket config to sum to 1.
        """
        # Raise error if value is negative
        if any(v < 0 for v in bucket_config.values()):
            raise ValueError("Bucket config values must be non-negative.")
        # Remove zero probability entries
        bucket_config = {k: v for k, v in bucket_config.items() if v > 0}
        # if bucket config is empty, raise error
        if not bucket_config:
            raise ValueError("Got invalid bucket config. "
                             "Bucket config values must be non-zero.")
        # Normalize the remaining bucket config to sum to 1
        total = sum(bucket_config.values())
        return {k: v / total for k, v in bucket_config.items()}


    def generate_mm_item(self, 
                         mm_item_config: tuple[int, int, int],
                         ) -> Mapping[str, Any]:
        """
        Create synthetic images and videos and 
        apply process_image/process_video respectively.
        This follows the OpenAI API chat completions
        https://github.com/openai/openai-python
        """
        
        if self.map_config_to_modality(mm_item_config) == "image":
            return process_image(self.generate_synthetic_image(
                                                            mm_item_config[1],
                                                            mm_item_config[0]))
        elif self.map_config_to_modality(mm_item_config) == "video":
            return process_video(self.generate_synthetic_video(
                                                            mm_item_config[1], 
                                                            mm_item_config[0], 
                                                            mm_item_config[2]))
        else:
            raise ValueError(f"Invalid multimodal item configuration: "
                             f"{mm_item_config}")


    def get_mm_item_sampling_params(
        self,
        base_items_per_request: int,
        num_mm_items_range_ratio: float,
        limit_mm_per_prompt: dict[str, int],
        bucket_config: dict[tuple[int, int, int], float],
    ) -> tuple[int, int, dict[str, int], dict[tuple[int, int, int], float]]:
        """
        Get the sampling parameters for the multimodal items.
        """
        # Enforce num_mm_items_range_ratio <= 1
        if not (0.0 <= num_mm_items_range_ratio <= 1.0):
            raise ValueError("num_mm_items_range_ratio must be in [0, 1].")

        # Ensure modalities to sample are in limit_mm_per_prompt
        for k, v in bucket_config.items():
            # get modality from bucket config
            modality = self.map_config_to_modality(k)
            if modality not in limit_mm_per_prompt:
                raise ValueError(f"Modality {modality} is not in "
                                 f"limit_mm_per_prompt: "
                                 f"{limit_mm_per_prompt.keys()}")

        # Remove zero probability entries 
        # and normalize bucket config to sum to 1
        bucket_config = self.normalize_bucket_config(bucket_config)
        logger.info(
            "Normalized bucket config: %s", bucket_config,
        )
        # Only consider limit per prompt for modalities in bucket config
        allowed_modalities = {self.map_config_to_modality(cfg) 
                              for cfg in bucket_config}
        limit_mm_per_prompt = {
            k: v for k, v in limit_mm_per_prompt.items() 
            if k in allowed_modalities}
        if not limit_mm_per_prompt:
            raise ValueError("No valid limits for modalities present in "
                             "bucket_config.")

        logger.info(
            "Updated mm-limit-per-prompt: %s", limit_mm_per_prompt,
        )

        # Get max and min num mm items and ensure
        # it is at most the sum of limit_mm_per_prompt for all modalities
        max_num_mm_items = min(
            sum(limit_mm_per_prompt.values()), 
            math.ceil(base_items_per_request * (1 + num_mm_items_range_ratio))
        )
        # Ensure min num mm items is at least 0
        min_num_mm_items = max(
            0, 
            math.floor(base_items_per_request * (1 - num_mm_items_range_ratio))
        )
        # Raise error if min num mm items is greater than max num mm items
        if min_num_mm_items > max_num_mm_items:
            raise ValueError(f"Min num mm items is greater than max mm items: "
                             f"{min_num_mm_items} > {max_num_mm_items}")
        
        logger.info(
            "Sampling number of multimodal items from [%s, %s]",
            min_num_mm_items, max_num_mm_items,
        )

        return (
            min_num_mm_items,
            max_num_mm_items,
            limit_mm_per_prompt,
            bucket_config,
        )

    def get_mm_item_iterator(
        self,
        min_num_mm_items: int,
        max_num_mm_items: int,
        bucket_config: dict[tuple[int, int, int], float],
        limit_mm_per_prompt: dict[str, int],
    ) -> Iterator[tuple[int,int, int]]:
        """
        Iterator over the multimodal items for each request
        whose size is between min_num_mm_items and max_num_mm_items.

        Loop over the bucket config and sample a multimodal item.
        Loop until the number of multimodal items sampled is equal to 
        request_num_mm_items or limit of multimodal items per prompt 
        for all modalities is reached.

        Note:
        - This function operates on a per-request shallow copy of
          `bucket_config` (tuple->float). The original dict passed to
          `sample` is not mutated. If this ever changes, a test
          is implemented and will fail.
        """
        # Get the number of multimodal items to sample
        request_num_mm_items = int(
            self._rng.integers(min_num_mm_items, max_num_mm_items + 1)
        ) 
        # If request_num_mm_items is 0, yield an empty iterator
        if request_num_mm_items == 0:
            return
        # Initialize modality counters
        modality_counter = {self.map_config_to_modality(k): 0 
                            for k in bucket_config}
        # Copy the bucket config to avoid modifying the original
        bucket_config_copy = bucket_config.copy()
        # Loop over the number of multimodal items to sample
        while sum(modality_counter.values()) < request_num_mm_items:
            # Sample a multimodal item config
            mm_item_config = self._rng.choice(list(bucket_config_copy.keys()), 
                                                p=list(bucket_config_copy.values()))
            modality = self.map_config_to_modality(mm_item_config)
            # Check that modality count is less than limit per prompt
            if modality_counter[modality] < limit_mm_per_prompt[modality]:
                modality_counter[modality] += 1
                yield (
                    mm_item_config
                )
            else:
                # If the counter is greater than the limit per prompt
                # set all multimodal items of this modality to 0
                for k, v in bucket_config_copy.items():
                    if self.map_config_to_modality(k) == modality:
                        bucket_config_copy[k] = 0
                # If all configs are 0, break the loop
                # This should not happen as request_num_mm_items is at most
                # the sum of limit_mm_per_prompt for all modalities
                if all(v == 0 for v in bucket_config_copy.values()):
                    logger.warning("Exhausted all multimodal items "
                                   "of modality %s",
                                   modality)
                    break
                # Renormalize the bucket config
                bucket_config_copy = self.normalize_bucket_config(
                                        bucket_config_copy)


    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        prefix_len: int = RandomDataset.DEFAULT_PREFIX_LEN,
        range_ratio: float = RandomDataset.DEFAULT_RANGE_RATIO,
        input_len: int = RandomDataset.DEFAULT_INPUT_LEN,
        output_len: int = RandomDataset.DEFAULT_OUTPUT_LEN,
        limit_mm_per_prompt: dict[str, int] = DEFAULT_LIMIT_MM_PER_PROMPT,
        base_items_per_request: int = DEFAULT_BASE_ITEMS_PER_REQUEST,
        num_mm_items_range_ratio: float = DEFAULT_NUM_MM_ITEMS_RANGE_RATIO,
        bucket_config: dict[tuple[int, int, int], float] = 
                                        DEFAULT_MM_ITEM_BUCKET_CONFIG,
        enable_multimodal_chat: bool = DEFAULT_ENABLE_MULTIMODAL_CHAT,
        **kwargs,
    ) -> list[SampleRequest]:

        # NOTE: Video sampling is WIP. Raise error if video is in bucket config
        # and probability is non-zero.
        if any(self.map_config_to_modality(cfg) == "video" and p > 0 
                for cfg, p in bucket_config.items()):
            raise NotImplementedError("Video sampling not implemented; "
                                      "set its probability to 0.")

        # Get the sampling parameters for the dataset
        input_lens, output_lens, offsets = self.get_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        (
            min_num_mm_items,
            max_num_mm_items,
            limit_mm_per_prompt,
            bucket_config,
        ) = self.get_mm_item_sampling_params(
            base_items_per_request,
            num_mm_items_range_ratio,
            limit_mm_per_prompt,
            bucket_config,
        )

        # Generate prefix once
        prefix_token_ids = self.get_prefix(tokenizer, prefix_len)
        vocab_size = tokenizer.vocab_size
        # Add synthetic multimodal items to each request
        mm_requests = []
        for i in range(num_requests):
            prompt, total_input_len = self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
            )
            # Get multimodal item iterator for a given request
            mm_item_iterator = self.get_mm_item_iterator(
                min_num_mm_items,
                max_num_mm_items,
                bucket_config,
                limit_mm_per_prompt,
            )

            mm_content = cast(list[dict[str, Any]], [
                self.generate_mm_item(mm_item_config)
                for mm_item_config in mm_item_iterator
            ])

            if enable_multimodal_chat:
                # NOTE: For now this option is only provided for completeness 
                # given that the serve.py benchmark currently does not use it.
                mm_chat_prompt: Any = prompt
                mm_chat_prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
                sample_request = SampleRequest(
                    prompt=mm_chat_prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    multi_modal_data=None,
                    request_id=request_id_prefix + str(i),
                )
            else:
                sample_request = SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(i),
                )
            mm_requests.append(sample_request)
        return mm_requests

# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

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
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        samples: list = []
        ind = 0
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path)
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = (len(completion_ids)
                              if output_len is None else output_len)
            if not is_valid_sequence(prompt_len,
                                     new_output_len,
                                     skip_min_output_len_check=output_len
                                     is not None):
                continue
            if image_path := entry.get("image"): 
                mm_content = process_image(image_path) 
            elif video_path := entry.get("video"): 
                mm_content = process_video(video_path)
            else: 
                mm_content = None
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    lora_request=lora_request,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                ))
            ind += 1
        self.maybe_oversample_requests(samples, num_requests, request_id_prefix)
        return samples


def add_dataset_parser(parser: FlexibleArgumentParser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=[
            "sharegpt", "burstgpt", "sonnet", "random", "random-mm", "hf", 
            "custom", "prefix_repetition"
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help=
        "Number of output tokens per request, used only for custom dataset.",
    )
    custom_group.add_argument(
        "--custom-skip-chat-template",
        action="store_true",
        help=
        "Skip applying chat template to prompt, used only for custom dataset.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=("Number of fixed prefix tokens before the random context "
              "in a request. "
              "The total input length is the sum of `random-prefix-len` and "
              "a random "
              "context length sampled from [input_len * (1 - range_ratio), "
              "input_len * (1 + range_ratio)]."),
    )
    random_group.add_argument(
        "--random-batch-size",
        type=int,
        default=1,
        help=("Batch size for random sampling. "
              "Only used for embeddings benchmark."),
    )

    # random multimodal dataset options
    random_mm_group = parser.add_argument_group(
        "random multimodal dataset options extended from random dataset")
    random_mm_group.add_argument(
        "--random-mm-base-items-per-request",
        type=int,
        default=RandomMultiModalDataset.DEFAULT_BASE_ITEMS_PER_REQUEST,
        help=(
            "Base number of multimodal items per request for random-mm. "
            "Actual per-request count is sampled around this base using "
            "--random-mm-num-mm-items-range-ratio."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-num-mm-items-range-ratio",
        type=float,
        default=RandomMultiModalDataset.DEFAULT_NUM_MM_ITEMS_RANGE_RATIO,
        help=(
            "Range ratio r in [0, 1] for sampling items per request. "
            "We sample uniformly from the closed integer range "
            "[floor(n*(1-r)), ceil(n*(1+r))] "
            "where n is the base items per request. "
            "r=0 keeps it fixed; r=1 allows 0 items. The maximum is clamped "
            "to the sum of per-modality limits from "
            "--random-mm-limit-mm-per-prompt. "
            "An error is raised if the computed min exceeds the max."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-limit-mm-per-prompt",
        type=json.loads,
        default=RandomMultiModalDataset.DEFAULT_LIMIT_MM_PER_PROMPT,
        help=(
            "Per-modality hard caps for items attached per request, e.g. "
            "'{\"image\": 3, \"video\": 0}'. The sampled per-request item "
            "count is clamped to the sum of these limits. When a modality "
            "reaches its cap, its buckets are excluded and probabilities are "
            "renormalized."
            "OBS.: Only image sampling is supported for now."
        ),
    )

    def _parse_mm_bucket_config(v: object) -> dict[tuple[int, int, int], float]:
        # If already a dict (e.g., programmatic call), normalize keys
        def normalize(d: dict) -> dict[tuple[int, int, int], float]:
            out: dict[tuple[int, int, int], float] = {}
            for k, val in d.items():
                key = k
                if isinstance(key, str):
                    with suppress(Exception):
                        key = ast.literal_eval(key)
                if not (isinstance(key, tuple) and len(key) == 3
                        and all(isinstance(x, int) for x in key)):
                    raise ValueError(
                        f"Invalid bucket key {k!r}. Expected tuple (H, W, T)."
                    )
                out[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
            return out

        if isinstance(v, dict):
            return normalize(v)
        if isinstance(v, str):
            # Python literal (supports tuple keys)
            parsed = ast.literal_eval(v)
            if not isinstance(parsed, dict):
                raise ValueError("Bucket config must parse to a dict.")
            return normalize(parsed)
        raise ValueError("Unsupported value for --random-mm-bucket-config.")

    random_mm_group.add_argument(
        "--random-mm-bucket-config",
        type=_parse_mm_bucket_config,
        default=RandomMultiModalDataset.DEFAULT_MM_ITEM_BUCKET_CONFIG,
        help=(
            "The bucket config is a dictionary mapping a multimodal item"
            "sampling configuration to a probability."
            "Currently allows for 2 modalities: images and videos. "
            "An bucket key is a tuple of (height, width, num_frames)"
            "The value is the probability of sampling that specific item. "
            "Example: "
            "--random-mm-bucket-config "
            "{(256, 256, 1): 0.5, (720, 1280, 1): 0.4, (720, 1280, 16): 0.10} "
            "First item: images with resolution 256x256 w.p. 0.5"
            "Second item: images with resolution 720x1280 w.p. 0.4 "
            "Third item: videos with resolution 720x1280 and 16 frames w.p. 0.1"
            "OBS.: If the probabilities do not sum to 1, they are normalized."
            "OBS bis.: Only image sampling is supported for now."
        ),
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-name",
        type=str,
        default=None,
        help=(
            "Name of the dataset on HuggingFace "
            "(e.g., 'lmarena-ai/VisionArena-Chat'). "
            "Specify this if your dataset-path is a local path."
        ),
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    prefix_repetition_group = parser.add_argument_group(
        "prefix repetition dataset options")
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=256,
        help="Number of prefix tokens per request, used only for prefix "
        "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=256,
        help="Number of suffix tokens per request, used only for prefix "
        "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=10,
        help="Number of prefixes to generate, used only for prefix repetition "
        "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for prefix "
        "repetition dataset.",
    )


def get_samples(args, tokenizer) -> list[SampleRequest]:
    if args.dataset_name == "custom":
        dataset = CustomDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.custom_skip_chat_template,
            request_id_prefix=args.request_id_prefix,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.endpoint_type == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
                request_id_prefix=args.request_id_prefix,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
                request_id_prefix=args.request_id_prefix,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        if (
            args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in VisionArenaDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif (
            args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in InstructCoderDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MTBenchDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in ConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = ConversationDataset
        elif (
            args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in AIMODataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif (
            args.dataset_path
            in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS  # noqa: E501
            or args.hf_name in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in ASRDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = ASRDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MLPerfDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MLPerfDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MLPerfDataset
            args.hf_split = "train"
        else:
            supported_datasets = set([
                dataset_name for cls in HuggingFaceDataset.__subclasses__()
                for dataset_name in cls.SUPPORTED_DATASET_PATHS
            ])
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats.")

        if dataset_class.IS_MULTIMODAL and args.endpoint_type not in [
                "openai-chat",
                "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat
            # endpoint-type.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' endpoint-type.")
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
            hf_name=args.hf_name,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=args.request_id_prefix,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
                request_id_prefix=args.request_id_prefix,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                request_id_prefix=args.request_id_prefix,
            ),
            "random": lambda: RandomDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
                request_id_prefix=args.request_id_prefix,
                batchsize=args.random_batch_size,
            ),
            "random-mm":
            lambda: RandomMultiModalDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                range_ratio=args.random_range_ratio,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                base_items_per_request=args.random_mm_base_items_per_request,
                limit_mm_per_prompt=args.random_mm_limit_mm_per_prompt,
                num_mm_items_range_ratio=args.random_mm_num_mm_items_range_ratio,
                bucket_config=args.random_mm_bucket_config,
                request_id_prefix=args.request_id_prefix,
            ),
            "prefix_repetition":
            lambda: PrefixRepetitionRandomDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.prefix_repetition_prefix_len,
                suffix_len=args.prefix_repetition_suffix_len,
                num_prefixes=args.prefix_repetition_num_prefixes,
                output_len=args.prefix_repetition_output_len,
                request_id_prefix=args.request_id_prefix,
            ),
        }

        try:
            # Enforce endpoint compatibility for multimodal datasets.
            if args.dataset_name == "random-mm" and args.endpoint_type not in [
                    "openai-chat"]:
                raise ValueError(
                    "Multi-modal content (images) is only supported on "
                    "'openai-chat' backend."
                )
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests


# -----------------------------------------------------------------------------
# Custom Dataset Implementation
# -----------------------------------------------------------------------------


class CustomDataset(BenchmarkDataset):
    """
    Implements the Custom dataset.  Loads data from a JSONL file and generates
    sample requests based on conversation turns. E.g.,
    ```
    {"prompt": "What is the capital of India?"}
    {"prompt": "What is the capital of Iran?"}
    {"prompt": "What is the capital of China?"}
    ```
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        # self.data will be a list of dictionaries
        # e.g., [{"prompt": "What is the capital of India?"}, ...]
        # This will be the standardized format which load_data()
        # has to convert into depending on the filetype of dataset_path.
        # sample() will assume this standardized format of self.data
        self.data = []

        # Load the JSONL file
        if self.dataset_path.endswith(".jsonl"):
            jsonl_data = pd.read_json(path_or_buf=self.dataset_path,
                                      lines=True)

            # check if the JSONL file has a 'prompt' column
            if "prompt" not in jsonl_data.columns:
                raise ValueError("JSONL file must contain a 'prompt' column.")

            # Convert each row to a dictionary and append to self.data
            # This will convert the DataFrame to a list of dictionaries
            # where each dictionary corresponds to a row in the DataFrame.
            # This is the standardized format we want for self.data
            for _, row in jsonl_data.iterrows():
                self.data.append(row.to_dict())
        else:
            raise NotImplementedError(
                "Only JSONL format is supported for CustomDataset.")

        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        sampled_requests = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["prompt"]

            # apply template
            if not skip_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": prompt
                    }],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)

        return sampled_requests


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------

@deprecated(
    "SonnetDataset is deprecated and will be removed in a future version.",
)
class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

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

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens)
                      for tokens in tokenized_lines) / len(tokenized_lines)

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
        num_prefix_lines = max(round((prefix_len - base_offset) / avg_len), 0)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        ind = 0
        while len(samples) < num_requests:
            extra_lines = random.choices(self.data,
                                         k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            if prompt_len <= input_len:
                samples.append(
                    SampleRequest(
                        prompt=prompt_formatted
                        if return_prompt_formatted else prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                         request_id=request_id_prefix + str(ind),
                    ))
                ind += 1
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

    def _sample_loaded_data(self, num_requests: int) -> list:
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

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list[SampleRequest]:
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
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    lora_request=lora_req,
                    request_id=request_id_prefix + str(i),
                ))
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: Union[set[str], dict[str, Callable]] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        no_stream: bool = False,
        dataset_subset: Optional[str] = None,
        hf_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_stream = not no_stream
        self.hf_name = hf_name or dataset_path
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=self.load_stream,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Conversation Dataset Implementation
# -----------------------------------------------------------------------------


class ConversationDataset(HuggingFaceDataset):
    """Dataset for conversation data with multimodal support."""
    SUPPORTED_DATASET_PATHS = {
        'lmms-lab/LLaVA-OneVision-Data', 'Aeala/ShareGPT_Vicuna_unfiltered'
    }
    IS_MULTIMODAL = True

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               request_id_prefix: str = "",
               **kwargs) -> list:
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(
            lambda x: len(x["conversations"]) >= 2)
        sampled_requests = []
        ind = 0
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(sampled_requests) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

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
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len and output len
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                ))
            ind += 1
        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat":
        lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1":
        lambda x: x["turns"][0][0]["content"]
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.hf_name)
            if parser_fn is None:
                raise ValueError(f"Unsupported dataset path: {self.hf_name}")
            prompt = parser_fn(item)
            mm_content = process_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(i),
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# Instruct Coder Dataset Implementation
# -----------------------------------------------------------------------------


class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    InstructCoder is the dataset designed for general code editing.  It consists
    of 114,239 instruction-input-output triplets, and covers multiple distinct
    code editing scenario.
    """

    DEFAULT_OUTPUT_LEN = 200  # this is the average default output length
    SUPPORTED_DATASET_PATHS = {
        "likaixin/InstructCoder",
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               request_id_prefix: str = "",
               **kwargs) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = (
                f"{item['input']}\n\n{item['instruction']} Just output "
                "the code, do not include any explanation."
            )

            # apply template
            prompt = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": prompt
                }],
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# MT-Bench Dataset Implementation
# -----------------------------------------------------------------------------


class MTBenchDataset(HuggingFaceDataset):
    """
    MT-Bench Dataset.
    https://huggingface.co/datasets/philschmid/mt-bench

    We create a single turn dataset for MT-Bench.
    This is similar to Spec decoding benchmark setup in vLLM
    https://github.com/vllm-project/vllm/blob/9d98ab5ec/examples/offline_inference/eagle.py#L14-L18
    """  # noqa: E501

    DEFAULT_OUTPUT_LEN = 256  # avg len used in SD bench in vLLM
    SUPPORTED_DATASET_PATHS = {
        "philschmid/mt-bench",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []

        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0]

            # apply template
            prompt = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": prompt
                }],
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# AIMO Dataset Implementation
# -----------------------------------------------------------------------------


class AIMODataset(HuggingFaceDataset):
    """
    Dataset class for processing a AIMO dataset with reasoning questions.
    """
    SUPPORTED_DATASET_PATHS = {
        "AI-MO/aimo-validation-aime", "AI-MO/NuminaMath-1.5",
        "AI-MO/NuminaMath-CoT"
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               request_id_prefix: str = "",
               **kwargs) -> list:
        sampled_requests = []
        ind = 0
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt, completion = item['problem'], item["solution"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len,
                                                        completion_len,
                                                        max_prompt_len=2048,
                                                        max_total_len=32000):
                continue
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=request_id_prefix + str(ind),
                    
                ))
            ind += 1
        self.maybe_oversample_requests(sampled_requests, num_requests,
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# Next Edit Prediction Dataset Implementation
# -----------------------------------------------------------------------------


zeta_prompt = """### Instruction:
You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.

### User Edits:

{}

### User Excerpt:

{}

### Response:

""" # noqa: E501


def _format_zeta_prompt(
        sample: dict,
        original_start_marker: str = "<|editable_region_start|>") -> dict:
    """Format the zeta prompt for the Next Edit Prediction (NEP) dataset.

    This function formats examples from the NEP dataset
    into prompts and expected outputs. It could be
    further extended to support more NEP datasets.

    Args:
        sample: The dataset sample containing events,
            inputs, and outputs.
        original_start_marker: The marker indicating the
            start of the editable region. Defaults to
            "<|editable_region_start|>".

    Returns:
        A dictionary with the formatted prompts and expected outputs.
    """
    events = sample["events"]
    input = sample["input"]
    output = sample["output"]
    prompt = zeta_prompt.format(events, input)

    # following the original implementation, extract the focused region
    # from the raw output
    output_start_index = output.find(original_start_marker)
    output_focused_region = output[output_start_index:]
    expected_output = output_focused_region

    return {"prompt": prompt, "expected_output": expected_output}


class NextEditPredictionDataset(HuggingFaceDataset):
    """
    Dataset class for processing a Next Edit Prediction dataset.
    """

    SUPPORTED_DATASET_PATHS = {
        "zed-industries/zeta",
    }
    MAPPING_PROMPT_FUNCS = {
        "zed-industries/zeta": _format_zeta_prompt,
    }

    def sample(self, tokenizer: PreTrainedTokenizerBase, num_requests: int,
               request_id_prefix: str = "",
               **kwargs):
        formatting_prompt_func = self.MAPPING_PROMPT_FUNCS.get(self.hf_name)
        if formatting_prompt_func is None:
            raise ValueError(f"Unsupported dataset path: {self.hf_name}")
        samples = []
        for i, sample in enumerate(self.data):
            sample = formatting_prompt_func(sample)
            samples.append(
                SampleRequest(
                    prompt=sample["prompt"],
                    prompt_len=len(tokenizer(sample["prompt"]).input_ids),
                    expected_output_len=len(
                        tokenizer(sample["expected_output"]).input_ids),
                    request_id=request_id_prefix + str(i),
                ))
            if len(samples) >= num_requests:
                break
        self.maybe_oversample_requests(samples, num_requests, request_id_prefix)
        return samples


# -----------------------------------------------------------------------------
# ASR Dataset Implementation
# -----------------------------------------------------------------------------


class ASRDataset(HuggingFaceDataset):
    """
    Dataset class for processing a ASR dataset for transcription.
    Tested on the following set:

    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | Dataset        | Domain                                 | Speaking Style           | hf-subset                   |
    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | TED-LIUM       | TED talks                              | Oratory                  | release1, release2, release3|
    |                |                                        |                          | release3-speaker-adaptation |
    | VoxPopuli      | European Parliament                    | Oratory                  | en, de, it, fr,  ...        |
    | LibriSpeech    | Audiobook                              | Narrated                 | "LIUM/tedlium"              |
    | GigaSpeech     | Audiobook, podcast, YouTube            | Narrated, spontaneous    | xs, s, m, l, xl, dev, test  |
    | SPGISpeech     | Financial meetings                     | Oratory, spontaneous     | S, M, L, dev, test          |
    | AMI            | Meetings                               | Spontaneous              | ihm, sdm                    |
    +----------------+----------------------------------------+--------------------------+-----------------------------+

    """  # noqa: E501

    SUPPORTED_DATASET_PATHS = {
        "openslr/librispeech_asr",
        "facebook/voxpopuli",
        "LIUM/tedlium",
        "edinburghcstr/ami",
        "speechcolab/gigaspeech",
        "kensho/spgispeech",
    }

    DEFAULT_OUTPUT_LEN = 128
    IS_MULTIMODAL = True

    # TODO Whisper-specific. Abstract interface when more models are supported.
    TRANSCRIPTION_PREAMBLE = (
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
    skip_long_audios: bool = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        prompt = ASRDataset.TRANSCRIPTION_PREAMBLE
        prompt_len = len(tokenizer(prompt).input_ids)
        sampled_requests = []
        ind = 0
        skipped = 0
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            audio = item["audio"]
            y, sr = audio["array"], audio["sampling_rate"]
            duration_s = librosa.get_duration(y=y, sr=sr)
            # Whisper max supported duration
            if self.skip_long_audios and duration_s > 30:
                skipped += 1
                continue

            mm_content = {"audio": (y, sr)}
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                ))
            ind += 1
        if skipped:
            logger.warning(
                "%d samples discarded from dataset due to"
                " their length being greater than"
                " what Whisper supports.",
                skipped,
            )
        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# MLPerf Dataset Implementation
# -----------------------------------------------------------------------------


class MLPerfDataset(HuggingFaceDataset):
    """
    MLPerf Inference Dataset.

    Dataset on HF:
    https://huggingface.co/datasets/mgoin/mlperf-inference-llama2-data
    https://huggingface.co/datasets/mgoin/mlperf-inference-llama3.1-data

    Each record contains:
      - "system_prompt": system role instruction.
      - "question": user question.
      - "output": reference answer.

    We combine the system prompt and question into a chat-formatted prompt
    (using the tokenizer's chat template) and set the expected output length to
    the tokenized length of the provided reference answer.
    """

    SUPPORTED_DATASET_PATHS = {
        "mgoin/mlperf-inference-llama2-data",
        "mgoin/mlperf-inference-llama3.1-data",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list[SampleRequest]:
        # Force dynamic output length based on reference completion.
        dynamic_output = output_len is None
        sampled_requests: list[SampleRequest] = []
        ind = 0

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break

            system_prompt = item["system_prompt"]
            question = item["question"]
            reference_answer = item["output"]

            # Build chat-style prompt using tokenizer template, if available.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            prompt_formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)

            # Determine output length from reference answer tokens.
            ref_out_len = len(
                tokenizer(reference_answer, add_special_tokens=False).input_ids
            )
            expected_output_len = ref_out_len if dynamic_output else output_len

            # Validate sequence lengths.
            if not is_valid_sequence(prompt_len, expected_output_len):
                continue

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt_formatted,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1

        self.maybe_oversample_requests(sampled_requests, num_requests, 
                                       request_id_prefix)
        return sampled_requests


# -----------------------------------------------------------------------------
# Prefix Repetition Dataset Implementation
# -----------------------------------------------------------------------------


class PrefixRepetitionRandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the repeated prefix 
    # dataset.
    DEFAULT_PREFIX_LEN = 256
    DEFAULT_SUFFIX_LEN = 256
    DEFAULT_NUM_PREFIXES = 10
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        suffix_len: int = DEFAULT_SUFFIX_LEN,
        num_prefixes: int = DEFAULT_NUM_PREFIXES,
        output_len: int = DEFAULT_OUTPUT_LEN,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list[SampleRequest]:
        vocab_size = tokenizer.vocab_size
        prompts_per_prefix = num_requests // num_prefixes
        if prompts_per_prefix == 0:
            raise ValueError(
                f"num_requests ({num_requests}) must be greater than or equal "
                f"to num_prefixes ({num_prefixes})"
            )

        def _generate_exact_length_tokens(target_length: int) -> list[int]:
            """Generate tokens that decode and re-encode to exactly
            target_length."""
            # Generate random tokens
            tokens = np.random.randint(
                0, vocab_size, size=target_length).tolist()
            text = tokenizer.decode(tokens)
            re_encoded = tokenizer.encode(text, add_special_tokens=False)

            if len(re_encoded) == target_length:
                return re_encoded
            elif len(re_encoded) < target_length:
                # Recursively generate additional consistent tokens
                needed = target_length - len(re_encoded)
                extra_tokens = _generate_exact_length_tokens(needed)
                return re_encoded + extra_tokens
            else:
                # Truncate to target length
                return re_encoded[:target_length]

        requests = []
        for _ in range(num_prefixes):
            prefix_tokens = _generate_exact_length_tokens(prefix_len)

            for _ in range(prompts_per_prefix):
                suffix_tokens = _generate_exact_length_tokens(suffix_len)

                combined_tokens = prefix_tokens + suffix_tokens
                prompt = tokenizer.decode(combined_tokens)
                prompt_len = len(combined_tokens)
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                    )
                )

        random.shuffle(requests)
        return requests
