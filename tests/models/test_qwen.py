from typing import Type

import pytest

from ..conftest import HfRunner, VllmRunner
from .utils import check_logprobs_close

models = ["qwen/qwen-vl"]


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("model", models)
def test_text_only_qwen_model(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    example_prompts,
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
):
    # This test checks language inputs only, since the visual component
    # for qwen-vl is still unsupported in VLLM. In the near-future, the
    # implementation and this test will be extended to consider
    # visual inputs as well.
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts,
            max_tokens,
            num_logprobs=num_logprobs,
        )

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts,
            max_tokens,
            num_logprobs=num_logprobs,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
