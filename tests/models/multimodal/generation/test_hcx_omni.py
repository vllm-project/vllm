# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperCLOVAX-SEED-Omni-8B generation tests.

Tests image-to-text and audio-to-text inference.
Compares vLLM output logprobs against HuggingFace reference.
"""

import numpy as np
import pytest
from PIL import Image
from transformers import AutoModelForCausalLM

from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B"

IMAGE_PROMPT = (
    "<|im_start|>user\n<|IMAGE_PAD|>\n"
    "What color is this image?<|im_end|>\n"
    "<|im_start|>assistant\n"
)

AUDIO_PROMPT = (
    "<|im_start|>user\n<|AUDIO_PAD|>\n"
    "What do you hear?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def run_image_test(
    hf_runner,
    vllm_runner,
    image: Image.Image,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    """Compare vLLM vs HF image-to-text output."""
    # vLLM first (needs fresh CUDA before HF)
    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=2,
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [IMAGE_PROMPT],
            max_tokens,
            num_logprobs=num_logprobs,
            images=[[image]],
        )

    with hf_runner(
        MODEL_NAME,
        dtype=dtype,
        auto_cls=AutoModelForCausalLM,
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            [IMAGE_PROMPT],
            max_tokens,
            num_logprobs=num_logprobs,
            images=[[image]],
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


def run_audio_test(
    hf_runner,
    vllm_runner,
    audio: np.ndarray,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    """Compare vLLM vs HF audio-to-text output."""
    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=2,
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [AUDIO_PROMPT],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[(audio, 16000)],
        )

    with hf_runner(
        MODEL_NAME,
        dtype=dtype,
        auto_cls=AutoModelForCausalLM,
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            [AUDIO_PROMPT],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[(audio, 16000)],
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_image_to_text(
    hf_runner,
    vllm_runner,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")

    image = Image.new("RGB", (224, 224), color=(128, 64, 32))
    run_image_test(
        hf_runner,
        vllm_runner,
        image,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_audio_to_text(
    hf_runner,
    vllm_runner,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")

    sr = 16000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, dtype=np.float32))
    run_audio_test(
        hf_runner,
        vllm_runner,
        audio,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )
