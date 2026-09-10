# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    HYBRID_MODELS,
    MAX_NUM_SEQS,
    SSM_MODELS,
)
from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_test

pytestmark = pytest.mark.hybrid_model


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_distributed_correctness(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        model,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
    ) as vllm_model:
        vllm_outputs_tp_1 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        tensor_parallel_size=2,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
    ) as vllm_model:
        vllm_outputs_tp_2 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=vllm_outputs_tp_1,
        outputs_1_lst=vllm_outputs_tp_2,
        name_0="vllm_tp_1",
        name_1="vllm_tp_2",
    )
