# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import logging
import math
import random
import tempfile
import urllib
from collections.abc import Iterator, Mapping
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.benchmarks.datasets.abstractions import BenchmarkDataset, SampleRequest
from vllm.benchmarks.datasets.utils import (
    apply_multimodal_chat_transformation,
    gen_prompt_decode_to_target_len,
    process_image,
    process_video,
)

logger = logging.getLogger(__name__)


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
        no_oversample: bool = False,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        **kwargs,
    ) -> list[SampleRequest]:
        # validate total input tokens (prefix + sampled) is at least 1.
        num_special = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special)
        min_sampled_input = math.floor(real_input_len * (1.0 - float(range_ratio)))
        min_total_input = int(prefix_len) + min_sampled_input
        if min_total_input < 1:
            raise ValueError(
                "--random-input-len is too small: with tokenizer special "
                f"tokens {num_special} and --random-range-ratio {range_ratio}, "
                "the minimum possible total input tokens (prefix + sampled) is "
                f"{min_total_input}. Increase --random-input-len and/or "
                "--random-prefix-len, or decrease --random-range-ratio so that "
                "prefix_len + floor(max(0, random_input_len - num_special)) "
                "* (1 - range_ratio) >= 1."
            )

        input_lens, output_lens, offsets = self.get_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        vocab_size = tokenizer.vocab_size
        prohibited_tokens = tokenizer.all_special_ids
        all_tokens = np.arange(vocab_size)
        allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))

        # Generate prefix once
        prefix_token_ids = self.get_prefix(allowed_tokens, prefix_len)

        requests = []
        token_mismatch_total = 0
        for i in range(num_requests):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(  # noqa: E501
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
                allowed_tokens=allowed_tokens,
            )
            token_mismatch_total += token_mismatch
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

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )

        return requests

    def get_prefix(
        self,
        allowed_tokens: np.ndarray,
        prefix_len: int,
    ) -> list[int]:
        """
        Get the prefix for the dataset.
        """
        return (
            allowed_tokens[
                self._rng.integers(0, len(allowed_tokens), size=prefix_len)
            ].tolist()
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
        output_high = max(output_high, 1)

        if input_low > input_high:
            raise ValueError(
                f"Invalid input sampling interval: low={input_low} > high={input_high}"
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

        input_lens = self._rng.integers(input_low, input_high + 1, size=num_requests)
        output_lens = self._rng.integers(output_low, output_high + 1, size=num_requests)
        offsets = self._rng.integers(0, tokenizer.vocab_size, size=num_requests)
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
        allowed_tokens: np.ndarray,
    ) -> tuple[str, int, int]:
        """
        Returns (prompt, total_input_len).

        NOTE: After decoding the prompt we have to encode and decode it again.
        This is done because in some cases N consecutive tokens
        give a string tokenized into != N number of tokens.
        For example for GPT2Tokenizer:
        [6880, 6881] -> ['Ġcalls', 'here'] ->
        [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
        To avoid uncontrolled change of the prompt length,
        the encoded sequence is truncated before being decoded again.
        """
        # Build the inner sequence by sampling
        # sequentially from the allowed tokens
        inner_seq = allowed_tokens[
            (offset + index + np.arange(input_len)) % len(allowed_tokens)
        ].tolist()
        token_sequence = prefix_token_ids + inner_seq

        # Decode, then re-encode and truncate to preserve token count invariants
        total_input_len = prefix_len + int(input_len)
        prompt, adjusted_token_sequence, token_mismatch = (
            gen_prompt_decode_to_target_len(
                tokenizer=tokenizer,
                token_sequence=token_sequence,
                target_token_len=total_input_len,
                add_special_tokens=False,
                rng=self._rng,
            )
        )
        total_input_len = len(adjusted_token_sequence)
        return prompt, total_input_len, token_mismatch


class RandomMultiModalDataset(RandomDataset):
    """
    Synthetic multimodal dataset (text + images) that extends RandomDataset.

    Status:
    - Images: supported via synthetic RGB data.
    - Video: supported via synthetic RGB data.
    - Audio: not yet supported.

    Sampling overview:
    1) Number of items per request is sampled uniformly from the integer range
       [floor(n·(1−r)), ceil(n·(1+r))], where n is the base count and r is
       `num_mm_items_range_ratio` in [0, 1]. r=0 keeps it fixed; r=1 allows 0.
       The maximum is further clamped to the sum of per-modality limits.
    2) Each item’s modality and shape is sampled from `bucket_config`, a dict
       mapping (height, width, num_frames) → probability. We treat
       `num_frames`=1 as image and `num_frames` > 1 as video.
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
    DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 255, "video": 1}

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

    def generate_synthetic_video(
        self, width: int, height: int, num_frames: int
    ) -> dict:
        """Generate synthetic video with random values.

        Creates a video with random pixel values, encodes it to MP4 format,
        and returns the content as bytes.
        """
        import cv2

        random_pixels = self._rng.integers(
            0,
            256,
            (num_frames, height, width, 3),
            dtype=np.uint8,
        )

        # Create a temporary video file in memory
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30  # frames per second

        with NamedTemporaryFile(suffix=".mp4", delete_on_close=False) as temp_file:
            temp_path = temp_file.name

            # Create video writer
            video_writer = cv2.VideoWriter(
                temp_path, fourcc=fourcc, fps=fps, frameSize=(width, height)
            )

            if not video_writer.isOpened():
                raise RuntimeError("Failed to create video writer")

            for frame in random_pixels:
                video_writer.write(frame)

            video_writer.release()
            temp_file.close()

            # Read the video file content
            with open(temp_path, "rb") as f:
                video_content = f.read()

            return {"bytes": video_content}

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        """Map the configuration to the modality."""
        if config[-1] == 1:
            return "image"
        elif config[-1] > 1:
            return "video"
        else:
            raise ValueError(f"Invalid multimodal item configuration: {config}")

    def normalize_bucket_config(
        self, bucket_config: dict[tuple[int, int, int], float]
    ) -> dict[tuple[int, int, int], float]:
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
            raise ValueError(
                "Got invalid bucket config. Bucket config values must be non-zero."
            )
        # Normalize the remaining bucket config to sum to 1
        total = sum(bucket_config.values())
        return {k: v / total for k, v in bucket_config.items()}

    def generate_mm_item(
        self,
        mm_item_config: tuple[int, int, int],
    ) -> Mapping[str, Any]:
        """
        Create synthetic images and videos and
        apply process_image/process_video respectively.
        This follows the OpenAI API chat completions
        https://github.com/openai/openai-python
        """

        if self.map_config_to_modality(mm_item_config) == "image":
            return process_image(
                self.generate_synthetic_image(mm_item_config[1], mm_item_config[0])
            )
        elif self.map_config_to_modality(mm_item_config) == "video":
            return process_video(
                self.generate_synthetic_video(
                    mm_item_config[1], mm_item_config[0], mm_item_config[2]
                )
            )
        else:
            raise ValueError(f"Invalid multimodal item configuration: {mm_item_config}")

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
                raise ValueError(
                    f"Modality {modality} is not in "
                    f"limit_mm_per_prompt: "
                    f"{limit_mm_per_prompt.keys()}"
                )

        # Remove zero probability entries
        # and normalize bucket config to sum to 1
        bucket_config = self.normalize_bucket_config(bucket_config)
        logger.info(
            "Normalized bucket config: %s",
            bucket_config,
        )
        # Only consider limit per prompt for modalities in bucket config
        allowed_modalities = {self.map_config_to_modality(cfg) for cfg in bucket_config}
        limit_mm_per_prompt = {
            k: v for k, v in limit_mm_per_prompt.items() if k in allowed_modalities
        }
        if not limit_mm_per_prompt:
            raise ValueError("No valid limits for modalities present in bucket_config.")

        logger.info(
            "Updated mm-limit-per-prompt: %s",
            limit_mm_per_prompt,
        )

        # Get max and min num mm items and ensure
        # it is at most the sum of limit_mm_per_prompt for all modalities
        max_num_mm_items = min(
            sum(limit_mm_per_prompt.values()),
            math.ceil(base_items_per_request * (1 + num_mm_items_range_ratio)),
        )
        # Ensure min num mm items is at least 0
        min_num_mm_items = max(
            0, math.floor(base_items_per_request * (1 - num_mm_items_range_ratio))
        )
        # Raise error if min num mm items is greater than max num mm items
        if min_num_mm_items > max_num_mm_items:
            raise ValueError(
                f"Min num mm items is greater than max mm items: "
                f"{min_num_mm_items} > {max_num_mm_items}"
            )

        logger.info(
            "Sampling number of multimodal items from [%s, %s]",
            min_num_mm_items,
            max_num_mm_items,
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
    ) -> Iterator[tuple[int, int, int]]:
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
        modality_counter = {self.map_config_to_modality(k): 0 for k in bucket_config}
        # Copy the bucket config to avoid modifying the original
        bucket_config_copy = bucket_config.copy()
        # Loop over the number of multimodal items to sample
        while sum(modality_counter.values()) < request_num_mm_items:
            # Sample a multimodal item config
            mm_item_config = self._rng.choice(
                list(bucket_config_copy.keys()), p=list(bucket_config_copy.values())
            )
            modality = self.map_config_to_modality(mm_item_config)
            # Check that modality count is less than limit per prompt
            if modality_counter[modality] < limit_mm_per_prompt[modality]:
                modality_counter[modality] += 1
                yield (mm_item_config)
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
                    logger.warning(
                        "Exhausted all multimodal items of modality %s", modality
                    )
                    break
                # Renormalize the bucket config
                bucket_config_copy = self.normalize_bucket_config(bucket_config_copy)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = RandomDataset.DEFAULT_PREFIX_LEN,
        range_ratio: float = RandomDataset.DEFAULT_RANGE_RATIO,
        input_len: int = RandomDataset.DEFAULT_INPUT_LEN,
        output_len: int = RandomDataset.DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        limit_mm_per_prompt: dict[str, int] = DEFAULT_LIMIT_MM_PER_PROMPT,
        base_items_per_request: int = DEFAULT_BASE_ITEMS_PER_REQUEST,
        num_mm_items_range_ratio: float = DEFAULT_NUM_MM_ITEMS_RANGE_RATIO,
        bucket_config: dict[
            tuple[int, int, int], float
        ] = DEFAULT_MM_ITEM_BUCKET_CONFIG,
        enable_multimodal_chat: bool = DEFAULT_ENABLE_MULTIMODAL_CHAT,
        **kwargs,
    ) -> list[SampleRequest]:
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

        vocab_size = tokenizer.vocab_size
        # Can't use tokenizer.all_special_ids since
        # it returns ONLY ids from special_tokens_map.json
        # We want to exclude placeholder tokens and all
        # tokens that indicate start/end of image as it
        # may break prompt replacement logic.
        prohibited_tokens = list(
            tok_id
            for tok_id, token in tokenizer.added_tokens_decoder.items()
            if token.special
        )
        all_tokens = np.arange(vocab_size)
        allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))
        logger.debug(
            "Sampling from %d out of %d (vocab size)", len(allowed_tokens), vocab_size
        )
        # Generate prefix once
        prefix_token_ids = self.get_prefix(allowed_tokens, prefix_len)
        # Add synthetic multimodal items to each request
        mm_requests = []
        token_mismatch_total = 0
        for i in range(num_requests):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(  # noqa: E501
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
                allowed_tokens=allowed_tokens,
            )
            token_mismatch_total += token_mismatch
            # Get multimodal item iterator for a given request
            mm_item_iterator = self.get_mm_item_iterator(
                min_num_mm_items,
                max_num_mm_items,
                bucket_config,
                limit_mm_per_prompt,
            )

            mm_content = cast(
                list[dict[str, Any]],
                [
                    self.generate_mm_item(mm_item_config)
                    for mm_item_config in mm_item_iterator
                ],
            )

            if enable_multimodal_chat:
                # NOTE: For now this option is only provided for completeness
                # given that the serve.py benchmark currently does not use it.
                mm_chat_prompt: Any = prompt
                mm_chat_prompt = apply_multimodal_chat_transformation(
                    prompt, mm_content
                )
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

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )

        return mm_requests


class RandomDatasetForReranking(RandomDataset):
    """
    Random dataset specialized for the needs of scoring:
    - Batches of inputs
    - Inputs composed of pairs
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = RandomDataset.DEFAULT_PREFIX_LEN,
        range_ratio: float = RandomDataset.DEFAULT_RANGE_RATIO,
        input_len: int = RandomDataset.DEFAULT_INPUT_LEN,
        output_len: int = RandomDataset.DEFAULT_OUTPUT_LEN,
        batchsize: int = 1,
        is_reranker: bool = True,
        **kwargs,
    ) -> list[SampleRequest]:
        n_sep_tokens = int(is_reranker)

        query_len_param = (input_len // 2) - n_sep_tokens if is_reranker else input_len

        query_lens, _, query_offsets = self.get_sampling_params(
            1, range_ratio, query_len_param, 0, tokenizer
        )

        query_len = int(query_lens[0])

        if not is_reranker:
            assert num_requests > 1 and batchsize > 1
            num_requests -= 1
            batchsize -= 1
            doc_len_param = input_len
        else:
            doc_len_param = input_len - query_len - n_sep_tokens

        doc_lens, _, doc_offsets = self.get_sampling_params(
            num_requests, range_ratio, doc_len_param, 0, tokenizer
        )
        vocab_size = tokenizer.vocab_size

        query_prompt, query_input_len, token_mismatch_total = (
            self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=[],
                prefix_len=0,
                vocab_size=vocab_size,
                input_len=query_len,
                offset=int(query_offsets[0]),
                index=0,
            )
        )

        requests = []
        for i in range(num_requests):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(  # noqa: E501
                tokenizer=tokenizer,
                prefix_token_ids=[],
                prefix_len=0,
                vocab_size=vocab_size,
                input_len=int(doc_lens[i]),
                offset=int(doc_offsets[i]),
                index=i + 1,
            )
            token_mismatch_total += token_mismatch
            requests.append((prompt, total_input_len))

        batch_requests = []
        # Create batched requests
        for i in range(0, num_requests, batchsize):
            batch = requests[i : i + batchsize]
            query_contrib = (
                (query_input_len + n_sep_tokens) * len(batch)
                if is_reranker
                else query_input_len
            )
            batch_requests.append(
                SampleRequest(
                    prompt=[query_prompt] + [req[0] for req in batch],
                    prompt_len=query_contrib + sum(req[1] for req in batch),
                    expected_output_len=0,
                    request_id=request_id_prefix + str(i // batchsize),
                )
            )

        if token_mismatch_total != 0:
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                "more" if token_mismatch_total > 0 else "fewer",
            )

        return batch_requests


class TxtSlicesDataset(BenchmarkDataset):
    """
    Implements the TxtSlices dataset. Takes a URL or file path to a text file,
    tokenizes the entire content, and generates sample requests by randomly
    slicing from the tokenized sequence with cycling support.
    """

    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._temp_file = None
        self.load_data()

    def __del__(self):
        """Clean up temporary file if it was created."""
        if self._temp_file is not None:
            import os

            with contextlib.suppress(Exception):
                os.unlink(self._temp_file)

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")

        # Check if dataset_path is a URL
        file_path: str
        if self.dataset_path.startswith(("http://", "https://")):
            # Download to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as f:
                self._temp_file = f.name  # type: ignore
                with urllib.request.urlopen(self.dataset_path) as response:
                    content = response.read().decode("utf-8")
                    f.write(content)
            file_path = self._temp_file  # type: ignore
        else:
            file_path = self.dataset_path

        # Read the file content
        with open(file_path, encoding="utf-8") as f:
            self.text_content = f.read()

    def get_token_ids(self, tokenizer: PreTrainedTokenizerBase):
        tokenized = tokenizer(self.text_content, add_special_tokens=False)
        token_ids = tokenized.input_ids
        total_tokens = len(token_ids)
        return token_ids, total_tokens

    def generate_prompt(
        self,
        tokenizer: PreTrainedTokenizerBase,
        total_tokens: int,
        token_ids: list,
        input_len: int,
        **kwargs,
    ):
        # Randomly select a start position
        start_pos = random.randint(0, total_tokens - 1)

        # Extract tokens with cycling if necessary
        prompt_token_ids = []
        for j in range(input_len):
            token_idx = (start_pos + j) % total_tokens
            prompt_token_ids.append(token_ids[token_idx])

        # Decode the tokens to get the prompt
        prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

        # Get actual token length after re-encoding to ensure accuracy
        actual_prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)

        return prompt, actual_prompt_len

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        # Tokenize the entire text content
        token_ids, total_tokens = self.get_token_ids(tokenizer)

        if total_tokens == 0:
            raise ValueError("The text file is empty or cannot be tokenized.")

        samples = []
        random.seed(self.random_seed)

        for i in range(num_requests):
            prompt, actual_prompt_len = self.generate_prompt(
                tokenizer, total_tokens, token_ids, input_len
            )

            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=actual_prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                )
            )

        return samples
