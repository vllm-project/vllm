# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import io
import random
from collections.abc import Mapping
from functools import cache
from io import BytesIO
from typing import Any, cast

import numpy as np
from PIL import Image

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.image import convert_image_mode
from vllm.tokenizers import TokenizerLike


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
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (
        prompt_too_short or output_too_short or prompt_too_long or combined_too_long
    )


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


def get_random_lora_request(
    max_loras: int | None = None,
    lora_path: str | None = None,
) -> LoRARequest | None:
    """
    Optionally select a random LoRA request.

    This method is used when LoRA parameters are provided.  It randomly
    selects a LoRA based on max_loras.

    Args:
        max_loras (Optional[int]): The maximum number of LoRAs available.
            If `None`, LoRA is not used.
        lora_path (Optional[str]): Path to the LoRA parameters on disk.
            If `None`, LoRA is not used.

    Returns:
        A new [`LoRARequest`][vllm.lora.request.LoRARequest]
        (or `None` if not applicable).
    """
    if max_loras is None or lora_path is None:
        return None

    # Generate a random LoRA ID in the range [1, max_loras].
    lora_id = random.randint(1, max_loras)
    lora_request = LoRARequest(
        lora_name=str(lora_id),
        lora_int_id=lora_id,
        lora_path=lora_path_on_disk(lora_path),
    )
    return lora_request


def apply_multimodal_chat_transformation(
    prompt: str,
    mm_content: MultiModalDataDict | dict | list[dict] | None = None,
) -> list[dict]:
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
                "Could not process multimodal content of type: " + f"{type(mm_content)}"
            )
    return [{"role": "user", "content": content}]


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
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = convert_image_mode(image, "RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = (
            image
            if image.startswith(("http://", "https://", "file://"))
            else f"file://{image}"
        )
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image"
        " or str or dictionary with raw image bytes."
    )


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
    if isinstance(video, dict) and "bytes" in video:
        video_bytes = video["bytes"]
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
        }

    if isinstance(video, str):
        video_url = (
            video
            if video.startswith(("http://", "https://", "file://"))
            else f"file://{video}"
        )
        return {"type": "video_url", "video_url": {"url": video_url}}

    raise ValueError(
        f"Invalid video input {video}. Must be a string of local path/remote url, or a dictionary with raw video bytes in the form of `{{'bytes': raw_video_bytes}}`."  # noqa: E501
    )


def gen_prompt_decode_to_target_len(
    tokenizer: TokenizerLike,
    token_sequence: list[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[str, list[int]]:
    """
    Ensure decoded-then-encoded prompt length matches the target token length.

    This function decodes an initial token sequence to text and re-encodes it
    , iteratively adjusting the token sequence length to match a target.
    This is necessary because some tokenizers do not guarantee a 1:1 mapping
    between consecutive tokens and the decoded-then-encoded sequence length.
    For example, for GPT2Tokenizer:
    [6880, 6881] -> ['Ġcalls', 'here'] ->
    [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']

    Returns a tuple of the final prompt string and the adjusted token sequence.
    """
    remain_num_try = max_retry
    token_mismatch = 0
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if remain_num_try <= 0:
            if len(token_sequence) != target_token_len:
                token_mismatch = len(token_sequence) - target_token_len
            break

        if len(token_sequence) == target_token_len:
            break
        elif len(token_sequence) < target_token_len:
            if rng is not None:
                extra_tokens = rng.integers(
                    0,
                    tokenizer.vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            else:
                extra_tokens = np.random.randint(
                    0,
                    tokenizer.vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            token_sequence.extend(extra_tokens)
        elif len(token_sequence) > target_token_len:
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1

    return prompt, token_sequence, token_mismatch
