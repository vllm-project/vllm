# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.forward_context import DPMetadata
from vllm.v1.attention.ops.pcp import maybe_gather_mla_latent_cache_inputs


@pytest.mark.parametrize(
    "pcp_size,sp_size,enable_ep,expected",
    [
        (1, 1, True, [5, 7]),
        (1, 2, True, [3, 3, 4, 4]),
        (2, 1, False, [10, 14]),
        (2, 1, True, [5, 5, 7, 7]),
        (2, 2, True, [3, 3, 3, 3, 4, 4, 4, 4]),
    ],
)
def test_dispatch_sizes_expand_pcp_before_tp(pcp_size, sp_size, enable_ep, expected):
    config = ParallelConfig(
        distributed_executor_backend="mp",
        data_parallel_size=2,
        prefill_context_parallel_size=pcp_size,
        tensor_parallel_size=sp_size,
        enable_expert_parallel=enable_ep,
    )
    metadata = DPMetadata.make(config, 5, torch.tensor([5, 7]))
    with metadata.sp_local_sizes(sp_size, pcp_size=pcp_size, use_ep=enable_ep) as sizes:
        assert sizes == expected
    assert metadata.local_sizes is None
    assert metadata.num_tokens_across_dp_cpu.tolist() == [5, 7]


@pytest.mark.parametrize(
    "dp_size,pcp_size,tp_size,use_ep,is_sp,expected",
    [
        (2, 2, 1, True, False, "ep"),
        (2, 2, 2, True, True, "ep"),
        (2, 2, 2, False, False, "dp"),
        (2, 1, 2, True, False, "dp"),
        (1, 2, 2, True, False, "pcp"),
        (2, 2, 2, True, False, None),
    ],
)
def test_dispatch_reuses_existing_groups(
    dp_size, pcp_size, tp_size, use_ep, is_sp, expected, monkeypatch
):
    groups = {
        "dp": SimpleNamespace(world_size=dp_size),
        "pcp": SimpleNamespace(world_size=pcp_size),
        "ep": object(),
    }
    for name, group in groups.items():
        monkeypatch.setattr(
            f"vllm.distributed.device_communicators.all2all.get_{name}_group",
            lambda group=group: group,
        )
    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    manager.dp_world_size = dp_size
    manager.tp_group = SimpleNamespace(world_size=tp_size)
    manager.use_ep = use_ep
    if expected is None:
        with pytest.raises(AssertionError, match="requires sequence-parallel MoE"):
            manager._get_comm_group(is_sp)
    else:
        assert manager._get_comm_group(is_sp) is groups[expected]


@pytest.mark.parametrize("enable_ep", [False, True])
def test_ag_rs_dispatch_and_combine_use_dp_pcp_sizes(monkeypatch, enable_ep):
    calls = []
    local_tokens = [20, 21] if enable_ep else [20, 21, 22, 23]

    class FakeGroup:
        world_size = 4 if enable_ep else 2
        rank_in_group = 2 if enable_ep else 1

        def all_gatherv(self, tensors, dim, sizes):
            calls.append(("gather", sizes))
            return [torch.tensor([10, 11, 20, 21, 22, 23]) for _ in tensors]

        def reduce_scatterv(self, tensor, dim, sizes):
            calls.append(("scatter", sizes))
            return tensor[2 : 2 + len(local_tokens)]

    config = ParallelConfig(
        distributed_executor_backend="mp",
        data_parallel_size=2,
        data_parallel_rank=1,
        prefill_context_parallel_size=2,
        enable_expert_parallel=enable_ep,
    )
    metadata = DPMetadata.make(config, 2, torch.tensor([1, 2]))
    manager = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    manager.dp_world_size = 2
    manager.use_ep = enable_ep

    manager.tp_group = SimpleNamespace(world_size=1)
    for name in ("dp", "ep"):
        monkeypatch.setattr(
            f"vllm.distributed.device_communicators.all2all.get_{name}_group",
            FakeGroup,
        )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_pcp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.all2all.get_forward_context",
        lambda: SimpleNamespace(dp_metadata=metadata),
    )
    with metadata.sp_local_sizes(1, pcp_size=2, use_ep=enable_ep):
        hidden_states, _, _ = manager.dispatch(
            torch.tensor(local_tokens),
            torch.ones(len(local_tokens)),
            torch.zeros(len(local_tokens)),
        )
        combined = manager.combine(hidden_states)
    assert combined.tolist() == local_tokens
    sizes = [1, 1, 2, 2] if enable_ep else [2, 4]
    assert calls == [("gather", sizes), ("scatter", sizes)]


@pytest.mark.parametrize("slots", [[3, 4], [3, 4, -1, -1, -1, -1]])
def test_decode_cache_write_ignores_dp_padding(slots):
    kv = torch.arange(12).reshape(6, 2)
    pe = torch.arange(6).reshape(6, 1, 1)
    slots = torch.tensor(slots)
    cache_kv, cache_pe, cache_slots = maybe_gather_mla_latent_cache_inputs(
        kv, pe, slots, num_decode_tokens=2, use_pcp=True
    )
    torch.testing.assert_close(cache_kv, kv[:2])
    torch.testing.assert_close(cache_pe, pe[:2])
    torch.testing.assert_close(cache_slots, slots[:2])


def test_expanded_slot_mapping_keeps_pcp_prefill_padding(monkeypatch):
    calls = []

    def all_gather(tensor, dim):
        calls.append(tensor.shape[0])
        return torch.cat((tensor, tensor), dim=dim)

    monkeypatch.setattr(
        "vllm.v1.attention.ops.pcp.get_pcp_group",
        lambda: SimpleNamespace(world_size=2, all_gather=all_gather),
    )
    kv = torch.zeros(3, 2)  # Two decodes followed by one PCP padding row.
    pe = torch.zeros(3, 1, 1)
    slots = torch.tensor([3, 4, 8, 3, 4, -1])
    cache_kv, _, cache_slots = maybe_gather_mla_latent_cache_inputs(
        kv, pe, slots, num_decode_tokens=2, use_pcp=True
    )
    assert calls == [1, 1]
    assert cache_kv.shape == (4, 2)
    assert cache_slots.tolist() == [3, 4, 8, -1]
