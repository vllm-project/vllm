# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

from ....utils import large_gpu_test
from ...utils import check_logprobs_close

MODELS = [
    "microsoft/Phi-3.5-MoE-instruct",
]


def test_phimoe_routing_function():
    from vllm.model_executor.models.phimoe import phimoe_routing_function
    test_case = {
        0: {
            "hidden_states":
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8],
                         dtype=torch.float32,
                         requires_grad=False).view(4, 2),
            "gating_output":
            torch.tensor([0.1, 0.2, 0.3, 0.4],
                         dtype=torch.float32,
                         requires_grad=False),
            "topk":
            2,
            "renormalize":
            False,
        },
        1: {
            "hidden_states":
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8],
                         dtype=torch.float32,
                         requires_grad=False).view(4, 2),
            "gating_output":
            torch.tensor([0.4, 0.2, 0.3, 0.4],
                         dtype=torch.float32,
                         requires_grad=False),
            "topk":
            2,
            "renormalize":
            False,
        }
    }

    ground_truth = {
        0: {
            "topk_weights":
            torch.tensor([1., 1.], dtype=torch.float32, requires_grad=False),
            "topk_ids":
            torch.tensor([3, 2], dtype=torch.long, requires_grad=False),
        },
        1: {
            "topk_weights":
            torch.tensor([0.5, 1.], dtype=torch.float32, requires_grad=False),
            "topk_ids":
            torch.tensor([0, 3], dtype=torch.long, requires_grad=False),
        }
    }

    for test_id in test_case:
        topk_weights, topk_ids = phimoe_routing_function(**test_case[test_id])
        assert torch.allclose(topk_weights,
                              ground_truth[test_id]["topk_weights"])
        assert torch.equal(topk_ids, ground_truth[test_id]["topk_ids"])


@pytest.mark.skipif(condition=current_platform.is_cpu(),
                    reason="This test takes a lot time to run on CPU, "
                    "and vllm CI's disk space is not enough for this model.")
@large_gpu_test(min_gb=80)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
