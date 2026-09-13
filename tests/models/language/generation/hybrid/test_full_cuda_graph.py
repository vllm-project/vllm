# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    ATTN_BACKEND,
    FULL_CUDA_GRAPH_MODELS,
    MAX_NUM_SEQS,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", FULL_CUDA_GRAPH_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_full_cuda_graph(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
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

    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        attention_backend=ATTN_BACKEND,
        enable_chunked_prefill=True,
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
