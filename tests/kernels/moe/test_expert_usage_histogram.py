# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.fused_moe.utils import (
    collect_expert_usage_histogram)


@pytest.mark.parametrize("topk_experts,expert_count,topk_ids_dtype",
                         [(8, 264, torch.int32), (4, 32, torch.int32),
                          (1, 1, torch.int64)])
@pytest.mark.parametrize("token_count", [1, 7, 256, 1024])
def test_collect_expert_usage_histogram(topk_experts: int, expert_count: int,
                                        token_count: int,
                                        topk_ids_dtype: torch.dtype):
    device = torch.device('cuda')

    # Make an uniform distribution of expert usage
    topk_ids = torch.stack([torch.arange(topk_experts, dtype=torch.int32)] *
                           token_count)

    topk_ids_gpu = topk_ids.to(device)

    expert_usage_histogram_gpu = torch.zeros(expert_count,
                                             dtype=topk_ids_dtype,
                                             device=device)

    collect_expert_usage_histogram(topk_ids_gpu, expert_usage_histogram_gpu)

    # Every expert is used the same amount, so expecting token_count for
    # each expert set in the topk_ids tensor.
    assert torch.equal(
        expert_usage_histogram_gpu[:topk_experts],
        torch.full([topk_experts],
                   token_count,
                   dtype=topk_ids_dtype,
                   device=device))

    # The rest of the experts weren't used, so they should be zero.
    assert expert_usage_histogram_gpu[topk_experts:].sum() == 0


@pytest.mark.parametrize("topk_experts,expert_count", [(16, 32)])
@pytest.mark.parametrize("token_count", [1])
@pytest.mark.parametrize("seed", [0xDEADBEEF, 0xCAFEBABE])
def test_collect_expert_usage_histogram_random(topk_experts: int,
                                               expert_count: int,
                                               token_count: int, seed: int):
    device = torch.device('cuda')

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Make random distribution of expert usage
    topk_ids_cpu = torch.stack(
        [torch.randperm(topk_experts, generator=generator, dtype=torch.int32)
         ] * token_count)

    # Compute ground truth
    torch_histogram = torch.histogram(topk_ids_cpu.to(torch.float),
                                      bins=expert_count,
                                      range=(0, expert_count - 1))

    # Use our function
    expert_usage_histogram_gpu = torch.zeros(expert_count,
                                             dtype=torch.int32,
                                             device=device)

    topk_ids_gpu = topk_ids_cpu.to(device)

    collect_expert_usage_histogram(topk_ids_gpu, expert_usage_histogram_gpu)

    assert torch.equal(expert_usage_histogram_gpu,
                       torch_histogram.hist.to(torch.int32).to(device))
