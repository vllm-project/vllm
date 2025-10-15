# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests compute_expert_num_tokens kernels
"""

import dataclasses

import pytest
import torch

from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens


@dataclasses.dataclass
class TestTensors:
    topk_ids: torch.Tensor
    expert_map: torch.Tensor | None = None

    def to_device(self, device: str):
        self.topk_ids = self.topk_ids.to(device=device)
        if self.expert_map is not None:
            self.expert_map = self.expert_map.to(device=device)

    @staticmethod
    def make(
        num_tokens: int,
        num_topk: int,
        num_experts: int,
        device: str,
        topk_ids_dtype: torch.dtype,
    ) -> "TestTensors":
        # make topk ids
        topk_ids = torch.empty((num_tokens, num_topk), device=device, dtype=torch.int64)
        for x in range(num_tokens):
            topk_ids[x] = torch.randperm(num_experts)[:num_topk]
        topk_ids = topk_ids.to(dtype=torch.int64)
        return TestTensors(topk_ids=topk_ids)

    def with_ep_rank(
        self, ep_rank: int, num_global_experts: int, num_local_experts: int, device: str
    ):
        # make an expert map
        expert_map = torch.empty((num_global_experts), device=device, dtype=torch.int32)
        expert_map.fill_(-1)
        s = ep_rank * num_local_experts
        e = s + num_local_experts
        expert_map[s:e] = torch.tensor(list(range(num_local_experts)), device=device)

        return TestTensors(topk_ids=self.topk_ids.clone(), expert_map=expert_map)


def ref_impl(tt: TestTensors, expert_num_tokens: torch.Tensor):
    # do the reference in cpu
    tt.to_device("cpu")
    expert_ids, counts = tt.topk_ids.unique(return_counts=True)

    for eid, count in zip(expert_ids, counts):
        if eid != -1 and tt.expert_map is not None:
            eid = tt.expert_map[eid]

        if eid == -1:
            continue

        expert_num_tokens[eid] += count


def do_test_compute_expert_num_tokens(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    ep_size: int,
    topk_ids_dtype: torch.dtype,
):
    assert num_topk <= num_experts

    tt = TestTensors.make(
        num_tokens, num_topk, num_experts, topk_ids_dtype=topk_ids_dtype, device="cpu"
    )

    num_global_experts = num_experts
    assert num_global_experts % ep_size == 0
    num_local_experts = num_global_experts // ep_size
    for ep_rank in range(ep_size):
        tt_rank = tt.with_ep_rank(ep_rank, num_global_experts, num_local_experts, "cpu")

        ref_expert_num_tokens = torch.zeros(
            (num_local_experts), device="cpu", dtype=torch.int32
        )
        ref_impl(tt_rank, ref_expert_num_tokens)
        ref_expert_num_tokens = ref_expert_num_tokens.to("cuda")

        tt_rank.to_device("cuda")
        # Test with expert_map
        triton_expert_num_tokens_w_emap = count_expert_num_tokens(
            tt_rank.topk_ids, num_local_experts, tt_rank.expert_map
        )

        # Test without expert map
        topk_ids = tt_rank.expert_map[tt_rank.topk_ids].to(topk_ids_dtype)
        triton_expert_num_tokens_wo_emap = count_expert_num_tokens(
            topk_ids, num_local_experts, expert_map=None
        )

        torch.testing.assert_close(
            ref_expert_num_tokens, triton_expert_num_tokens_w_emap, atol=0, rtol=0
        )
        torch.testing.assert_close(
            ref_expert_num_tokens, triton_expert_num_tokens_wo_emap, atol=0, rtol=0
        )


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 11, 127, 128, 3333, 7317])
@pytest.mark.parametrize("num_topk", [2, 6, 8])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("topk_ids_dtype", [torch.int64])
def test_compute_expert_num_tokens(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    ep_size: int,
    topk_ids_dtype: torch.dtype,
):
    do_test_compute_expert_num_tokens(
        num_tokens, num_topk, num_experts, ep_size, topk_ids_dtype
    )


@pytest.mark.parametrize("numel", list(range(1, 8192, 111)))
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("ep_size", [2])
@pytest.mark.parametrize("topk_ids_dtype", [torch.int64])
def test_compute_expert_num_tokens_from_numel(
    numel: int, num_experts: int, ep_size: int, topk_ids_dtype: torch.dtype
):
    do_test_compute_expert_num_tokens(
        num_tokens=numel,
        num_topk=1,
        num_experts=num_experts,
        ep_size=ep_size,
        topk_ids_dtype=topk_ids_dtype,
    )
