# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, NamedTuple

import pytest
from huggingface_hub import hf_hub_download
from pytest import MarkDecorator

from tests.quantization.utils import is_quant_method_supported
from vllm.assets.image import ImageAsset
from vllm.utils.torch_utils import set_default_torch_num_threads

from ....conftest import PromptImageInput, VllmRunner
from ...utils import check_logprobs_close


class GGUFMMTestConfig(NamedTuple):
    original_model: str
    gguf_repo: str
    gguf_backbone: str
    gguf_mmproj: str
    prompt: list[str]
    mm_data: dict[Literal["images"], PromptImageInput]
    max_model_len: int = 4096
    marks: list[MarkDecorator] = []

    @property
    def gguf_model(self):
        hf_hub_download(self.gguf_repo, filename=self.gguf_mmproj)
        return hf_hub_download(self.gguf_repo, filename=self.gguf_backbone)


GEMMA3_CONFIG = GGUFMMTestConfig(
    original_model="google/gemma-3-4b-it",
    gguf_repo="google/gemma-3-4b-it-qat-q4_0-gguf",
    gguf_backbone="gemma-3-4b-it-q4_0.gguf",
    gguf_mmproj="mmproj-model-f16-4B.gguf",
    prompt=["<start_of_image>Describe this image in detail:"],
    mm_data={"images": [ImageAsset("stop_sign").pil_image]},
    marks=[pytest.mark.core_model],
)

MODELS_TO_TEST = [GEMMA3_CONFIG]


def run_multimodal_gguf_test(
    vllm_runner: type[VllmRunner],
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    # Run gguf model.
    with (
        set_default_torch_num_threads(1),
        vllm_runner(
            model_name=model.gguf_model,
            enforce_eager=True,
            tokenizer_name=model.original_model,
            dtype=dtype,
            max_model_len=model.max_model_len,
        ) as gguf_model,
    ):
        gguf_outputs = gguf_model.generate_greedy_logprobs(
            prompts=model.prompt,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            **model.mm_data,
        )

    # Run unquantized model.
    with vllm_runner(
        model_name=model.original_model,
        enforce_eager=True,  # faster tests
        dtype=dtype,
        max_model_len=model.max_model_len,
    ) as original_model:
        original_outputs = original_model.generate_greedy_logprobs(
            prompts=model.prompt,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            **model.mm_data,
        )

    check_logprobs_close(
        outputs_0_lst=original_outputs,
        outputs_1_lst=gguf_outputs,
        name_0="original",
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
def test_models(
    vllm_runner: type[VllmRunner],
    model: GGUFMMTestConfig,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    run_multimodal_gguf_test(vllm_runner, model, dtype, max_tokens, num_logprobs)
