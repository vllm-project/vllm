# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....conftest import HfRunner, VllmRunner
from ....utils import create_new_process_for_each_test
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: list[tuple[list[str], list[str]]],
    model: str,
    *,
    max_model_len: int,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: str | None = None,
    enforce_eager: bool = True,
) -> None:
    """Inference result should be the same between hf and vllm."""
    with vllm_runner(
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                vllm_prompts,
                max_tokens,
                num_logprobs=num_logprobs,
            )
            for vllm_prompts, _ in inputs
        ]

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                hf_prompts,
                max_tokens,
                num_logprobs=num_logprobs,
            )
            for _, hf_prompts in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.fixture
def input_texts() -> list[tuple[list[str], list[str]]]:
    inputs = [
        (["Translate English to French: Hello world"], ["Translate English to French: Hello world"]),
        (["Summarize: The quick brown fox jumps over the lazy dog."], ["Summarize: The quick brown fox jumps over the lazy dog."]),
    ]
    return inputs


def check_model_available(model: str) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")


@pytest.mark.core_model
@pytest.mark.parametrize("model", ["google/t5gemma-2-270m-270m"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("enforce_eager", [True, False])
@create_new_process_for_each_test()
def test_models(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    num_logprobs: int,
    input_texts,
    enforce_eager: bool,
) -> None:
    check_model_available(model)
    run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        model,
        dtype=dtype,
        max_model_len=512,
        max_tokens=50,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
    )