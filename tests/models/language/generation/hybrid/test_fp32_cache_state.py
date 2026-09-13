# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    FP32_STATE_MODELS,
    MAX_NUM_SEQS,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", FP32_STATE_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "cache_dtype_param", ["mamba_ssm_cache_dtype", "mamba_cache_dtype"]
)
def test_fp32_cache_state(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    cache_dtype_param: str,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    # Leave enough headroom for repeated engine initialization on a
    # 32.5 GiB MIG.
    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=True,
        **{cache_dtype_param: "float32"},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
