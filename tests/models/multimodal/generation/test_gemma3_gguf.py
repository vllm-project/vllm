# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gemma3 GGUF generation tests covering text-only and multimodal inference.
Tests validate GGUF quantized models against unquantized HF models.
"""

from huggingface_hub import hf_hub_download

from tests.conftest import VllmRunner
from tests.models.utils import check_logprobs_close

# Model configurations
GEMMA3_1B_GGUF_REPO = "google/gemma-3-1b-it-qat-q4_0-gguf"
GEMMA3_1B_GGUF_FILE = "gemma-3-1b-it-q4_0.gguf"
GEMMA3_1B_HF = "google/gemma-3-1b-it"

GEMMA3_4B_GGUF_REPO = "google/gemma-3-4b-it-qat-q4_0-gguf"
GEMMA3_4B_GGUF_FILE = "gemma-3-4b-it-q4_0.gguf"
GEMMA3_4B_HF = "google/gemma-3-4b-it"

MAX_TOKENS = 50
NUM_LOGPROBS = 5


def test_gemma3_1b_gguf_text_only(
    vllm_runner: type[VllmRunner],
    example_prompts,
) -> None:
    """Test text-only generation with Gemma3 1B GGUF (no mmproj)."""
    gguf_path = hf_hub_download(GEMMA3_1B_GGUF_REPO, GEMMA3_1B_GGUF_FILE)

    with vllm_runner(
        gguf_path,
        tokenizer_name=GEMMA3_1B_HF,
        max_model_len=1024,
        enforce_eager=True,
    ) as gguf_model:
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            example_prompts[:1], MAX_TOKENS, NUM_LOGPROBS
        )

    with vllm_runner(
        GEMMA3_1B_HF,
        max_model_len=1024,
        enforce_eager=True,
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(
            example_prompts[:1], MAX_TOKENS, NUM_LOGPROBS
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="hf",
        name_1="gguf",
    )


def test_gemma3_4b_gguf_text_only(
    vllm_runner: type[VllmRunner],
    example_prompts,
) -> None:
    """Test text-only generation with Gemma3 4B GGUF (has mmproj, text prompt).

    Validates that multimodal model handles text-only prompts correctly.
    """
    gguf_path = hf_hub_download(GEMMA3_4B_GGUF_REPO, GEMMA3_4B_GGUF_FILE)

    with vllm_runner(
        gguf_path,
        tokenizer_name=GEMMA3_4B_HF,
        max_model_len=4096,
        enforce_eager=True,
    ) as gguf_model:
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            example_prompts[:1], MAX_TOKENS, NUM_LOGPROBS
        )

    with vllm_runner(
        GEMMA3_4B_HF,
        max_model_len=4096,
        enforce_eager=True,
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(
            example_prompts[:1], MAX_TOKENS, NUM_LOGPROBS
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="hf",
        name_1="gguf",
    )


def test_gemma3_4b_gguf_single_image(
    vllm_runner: type[VllmRunner],
    image_assets,
) -> None:
    """Test single image generation with Gemma3 4B GGUF."""
    gguf_path = hf_hub_download(GEMMA3_4B_GGUF_REPO, GEMMA3_4B_GGUF_FILE)
    # Download mmproj file to enable multimodal
    hf_hub_download(GEMMA3_4B_GGUF_REPO, "mmproj-model-f16-4B.gguf")

    stop_sign = image_assets[0]
    prompt = "<start_of_image>What's the content in the center of the image?"

    with vllm_runner(
        gguf_path,
        tokenizer_name=GEMMA3_4B_HF,
        max_model_len=4096,
        enforce_eager=True,
    ) as gguf_model:
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            [prompt],
            MAX_TOKENS,
            NUM_LOGPROBS,
            images=[[stop_sign.pil_image]],
        )

    with vllm_runner(
        GEMMA3_4B_HF,
        max_model_len=4096,
        enforce_eager=True,
        mm_processor_kwargs={"do_pan_and_scan": True},
    ) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(
            [prompt],
            MAX_TOKENS,
            NUM_LOGPROBS,
            images=[[stop_sign.pil_image]],
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="hf",
        name_1="gguf",
    )
