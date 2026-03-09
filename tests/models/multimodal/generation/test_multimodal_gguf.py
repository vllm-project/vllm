# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from typing import Any, NamedTuple

import pytest
from huggingface_hub import hf_hub_download
from pytest import MarkDecorator
from transformers import AutoModelForImageTextToText

from tests.quantization.utils import is_quant_method_supported
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import rescale_image_size
from vllm.utils.torch_utils import set_default_torch_num_threads

from ....conftest import IMAGE_ASSETS, HfRunner, VllmRunner
from ...utils import check_logprobs_close


class GGUFMMTestConfig(NamedTuple):
    original_model: str
    gguf_repo: str
    gguf_backbone: str
    gguf_mmproj: str
    prompt: list[str]
    image_names: list[str]  # Store names, load PIL images at runtime
    max_model_len: int = 4096
    marks: list[MarkDecorator] = []
    mm_processor_kwargs: dict[str, Any] = {}

    @property
    def gguf_model(self):
        hf_hub_download(self.gguf_repo, filename=self.gguf_mmproj)
        return hf_hub_download(self.gguf_repo, filename=self.gguf_backbone)


# Common prompts aligned with test_common.py "gemma3" entry format
_GEMMA3_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": (
            "<bos><start_of_turn>user\n"
            "<start_of_image>What's the content in the center of the image?"
            "<end_of_turn>\n<start_of_turn>model\n"
        ),
        "cherry_blossom": (
            "<bos><start_of_turn>user\n"
            "<start_of_image>What is the season?"
            "<end_of_turn>\n<start_of_turn>model\n"
        ),
    }
)

# Image asset names - load at runtime to avoid pickle issues with subprocess
_GEMMA3_IMAGE_NAMES = ["stop_sign", "cherry_blossom"]

# Regular multimodal (no pan-and-scan) - uses QAT Q4_0 GGUF
GEMMA3_CONFIG = GGUFMMTestConfig(
    original_model="google/gemma-3-4b-it",
    gguf_repo="google/gemma-3-4b-it-qat-q4_0-gguf",
    gguf_backbone="gemma-3-4b-it-q4_0.gguf",
    gguf_mmproj="mmproj-model-f16-4B.gguf",
    prompt=_GEMMA3_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
    max_model_len=4096,
    marks=[pytest.mark.core_model],
    mm_processor_kwargs={},
)

# Pan-and-scan multimodal - uses unquantized BF16 GGUF
GEMMA3_CONFIG_PAN_AND_SCAN = GGUFMMTestConfig(
    original_model="google/gemma-3-4b-it",
    gguf_repo="unsloth/gemma-3-4b-it-GGUF",
    gguf_backbone="gemma-3-4b-it-BF16.gguf",
    gguf_mmproj="mmproj-BF16.gguf",
    prompt=_GEMMA3_PROMPTS,
    image_names=_GEMMA3_IMAGE_NAMES,
    max_model_len=4096,
    marks=[pytest.mark.core_model],
    mm_processor_kwargs={"do_pan_and_scan": True},
)

MODELS_TO_TEST = [GEMMA3_CONFIG, GEMMA3_CONFIG_PAN_AND_SCAN]


def run_multimodal_gguf_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    # Load images at runtime (inside subprocess) to avoid pickle issues
    images = [ImageAsset(name).pil_image for name in model.image_names]
    size_factors = [0.25, 0.5, 1.0]
    inputs_per_image = [
        (
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        )
        for image, prompt in zip(images, model.prompt)
    ]

    # NOTE: Run vLLM first to avoid CUDA init issues with multiprocessing fork.
    # Run GGUF model via vLLM.
    with (
        set_default_torch_num_threads(1),
        vllm_runner(
            model_name=model.gguf_model,
            enforce_eager=True,
            tokenizer_name=model.original_model,
            dtype=dtype,
            max_model_len=model.max_model_len,
            mm_processor_kwargs=model.mm_processor_kwargs,
        ) as gguf_model,
    ):
        gguf_outputs_per_case = [
            gguf_model.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs_per_image
        ]

    # Then run HfRunner for HuggingFace baseline comparison.
    with hf_runner(
        model.original_model,
        dtype=dtype,
        auto_cls=AutoModelForImageTextToText,
    ) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                images=images,
            )
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, gguf_outputs in zip(hf_outputs_per_case, gguf_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=gguf_outputs,
            name_0="hf",
            name_1="gguf",
        )


@pytest.mark.skipif(
    not is_quant_method_supported("gguf"),
    reason="gguf is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(test_config, marks=test_config.marks)
        for test_config in MODELS_TO_TEST
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [10])
def test_gemma3_mm_gguf(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    run_multimodal_gguf_test(
        hf_runner, vllm_runner, model, dtype, max_tokens, num_logprobs
    )
