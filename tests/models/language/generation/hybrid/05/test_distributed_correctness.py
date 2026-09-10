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

# The "05" directory is not a valid Python identifier, so pytest imports this
# file as a top-level module. The @multi_gpu_test spawn child resolves the
# test callable via importlib.import_module(f.__module__), so point __module__
# at the real package path, which the child can import (its PYTHONPATH
# includes the repo root and importlib accepts numeric path components).
__name__ = "tests.models.language.generation.hybrid.05.test_distributed_correctness"


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
