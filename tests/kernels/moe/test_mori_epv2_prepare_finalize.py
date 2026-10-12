# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 import (
    MoriEPv2PrepareAndFinalize,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("world_size,lsa_size", [(4, 4), (8, 8), (16, 16), (8, 4)])
def test_mori_epv2_requires_one_lsa_team(monkeypatch, world_size, lsa_size):
    from vllm.distributed.device_communicators.all2all import MoriEPv2All2AllManager
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
    )

    monkeypatch.setattr(
        All2AllManagerBase,
        "__init__",
        lambda self, group: self.__dict__.update(
            rank=0, world_size=world_size, internode=world_size > 4, cpu_group=group
        ),
    )
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **kw: None
    )
    device_comm = Mock(lsa_size=lsa_size)
    comm = Mock()
    comm.create_dev_comm.return_value = device_comm
    communicator = Mock()
    communicator.init.return_value = comm
    monkeypatch.setitem(
        sys.modules,
        "mori.cco",
        SimpleNamespace(
            Communicator=communicator,
            CCODevCommRequirements=SimpleNamespace,
            GDA_CONNECTION_NONE=0,
        ),
    )
    if lsa_size != world_size:
        with pytest.raises(ValueError, match="one LSA team"):
            MoriEPv2All2AllManager(object())
    else:
        manager = MoriEPv2All2AllManager(object())
        manager.destroy()
    device_comm.close.assert_called_once()
    comm.destroy.assert_called_once()


def test_mori_epv2_accepts_non_gated_experts(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    config = make_dummy_moe_config(activation=MoEActivation.SILU_NO_MUL)
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        dp_size=4,
        ep_size=4,
        use_ep=True,
        all2all_backend="mori_epv2",
    )
    config = replace(config, moe_parallel_config=parallel)
    assert config.use_mori_epv2_kernels
    assert not config.rocm_aiter_fmoe_enabled
    with pytest.raises(AssertionError, match="requires AITER"):
        replace(
            config,
            moe_parallel_config=replace(parallel, all2all_backend="mori_low_latency"),
        )


class FakeMoriEPv2Op:
    def __init__(self, recv_count):
        self.routing = object()
        self.recv_count = recv_count
        self.recv_ids = torch.tensor([[2, 3]], dtype=torch.int32).repeat(3, 1)
        self.dispatch_args = None
        self.combine_args = None

    def dispatch(self, *args, **kwargs):
        self.dispatch_args = (args, kwargs)
        return (
            torch.full((3, 4), float("nan"), dtype=torch.bfloat16),
            torch.full((3, 2), float("inf"), dtype=torch.float32),
            None,
            self.recv_ids,
            torch.tensor([self.recv_count], dtype=torch.int32),
            self.routing,
        )

    def combine(self, *args, **kwargs):
        self.combine_args = (args, kwargs)
        return torch.full((1, 4), 7, dtype=torch.bfloat16), None


@pytest.mark.parametrize("recv_count", [0, 1, 3])
def test_mori_epv2_pads_receive_capacity_and_preserves_routing(monkeypatch, recv_count):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op = FakeMoriEPv2Op(recv_count)
    prepare_finalize = MoriEPv2PrepareAndFinalize(op, 16, 4)
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    weights = torch.ones(1, 2, dtype=torch.bfloat16)
    original_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    prepared = prepare_finalize.prepare(
        hidden,
        weights,
        original_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(),
        defer_input_quant=True,
    )
    for actual, active, padding in (
        (prepared[0], float("nan"), 0),
        (prepared[3], torch.tensor([2, 3]), -1),
        (prepared[4], float("inf"), 0),
    ):
        expected = torch.empty_like(actual)
        expected[:recv_count] = active
        expected[recv_count:] = padding
        torch.testing.assert_close(actual, expected, equal_nan=True)
    output = torch.empty_like(hidden)
    prepare_finalize.finalize(
        output,
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )

    assert op.dispatch_args is not None
    assert op.combine_args is not None
    assert op.dispatch_args[1] == {
        "return_routing": True,
        "clone_routing": False,
    }
    assert op.dispatch_args[0][1].dtype == torch.float32
    assert op.dispatch_args[0][3].dtype == torch.int32
    assert op.combine_args[0][2] is op.recv_ids
    assert op.combine_args[1]["routing"] is op.routing
    assert torch.equal(output, torch.full_like(output, 7))


def test_mori_epv2_routes_padding_tokens_to_local_experts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    op = FakeMoriEPv2Op(recv_count=2)
    # Rank 1 of 4 with 2 experts each owns experts 2 and 3.
    prepare_finalize = MoriEPv2PrepareAndFinalize(
        op, 16, 4, num_local_experts=2, rank_expert_offset=2
    )
    # The second token is DP padding: the router marks it with expert id -1.
    topk_ids = torch.tensor([[0, 7], [-1, -1]], dtype=torch.int64)
    prepare_finalize.prepare(
        torch.ones(2, 4, dtype=torch.bfloat16),
        torch.ones(2, 2, dtype=torch.bfloat16),
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(),
    )
    assert op.dispatch_args is not None
    sent = op.dispatch_args[0][3]
    assert sent.dtype == torch.int32
    assert sent.tolist() == [[0, 7], [2, 3]]


def _manager_with_fake_mori(monkeypatch, vmm_gb):
    from vllm.distributed.device_communicators.all2all import MoriEPv2All2AllManager
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
    )

    monkeypatch.setattr(
        All2AllManagerBase,
        "__init__",
        lambda self, group: self.__dict__.update(
            rank=0, world_size=8, internode=False, cpu_group=group
        ),
    )
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **kw: None
    )
    monkeypatch.setenv("VLLM_MORI_EPV2_PER_RANK_VMM_GB", str(vmm_gb))
    comm = Mock()
    comm.create_dev_comm.return_value = Mock(lsa_size=8)
    communicator = Mock()
    communicator.init.return_value = comm
    monkeypatch.setitem(
        sys.modules,
        "mori.cco",
        SimpleNamespace(
            Communicator=communicator,
            CCODevCommRequirements=SimpleNamespace,
            GDA_CONNECTION_NONE=0,
        ),
    )

    def config(**kw):
        return SimpleNamespace(
            **kw,
            scale_dim=0,
            scale_type_size=0,
            token_nbytes=kw["hidden_dim"] * 2,
            combine_token_nbytes=kw["hidden_dim"] * 2,
            effective_max_recv=kw["world_size"] * kw["max_num_inp_token_per_rank"],
            dispatch_dtype=torch.bfloat16,
            combine_dtype=torch.bfloat16,
            schedule=None,
        )

    class FakeOp:  # the handle cache holds ops by weak reference
        def __init__(self, cfg, comm):
            self.cfg = cfg

    monkeypatch.setitem(
        sys.modules,
        "mori.ops.dispatch_combine_v2",
        SimpleNamespace(EpDispatchCombineConfig=config, EpDispatchCombineOp=FakeOp),
    )
    return MoriEPv2All2AllManager(object())


@pytest.mark.parametrize("vmm_gb,fits", [(4, True), (3, False)])
def test_mori_epv2_rejects_arenas_past_the_cco_window(monkeypatch, vmm_gb, fits):
    manager = _manager_with_fake_mori(monkeypatch, vmm_gb)
    # DeepSeek-V4-Pro at EP8 and 16k tokens per rank: about 3.5 GiB of arena.
    args = dict(
        hidden_dim=7168,
        max_num_inp_token_per_rank=16384,
        num_experts_per_rank=48,
        num_experts_per_token=6,
    )
    if fits:
        assert manager.get_handle(args).cfg.hidden_dim == 7168
    else:
        with pytest.raises(ValueError, match="VLLM_MORI_EPV2_PER_RANK_VMM_GB"):
            manager.get_handle(args)


class FakeCapacityOp:
    """MORI EPv2 op whose receive buffers have `capacity` rows, `recv` valid."""

    def __init__(self, capacity, recv):
        self.routing = object()
        self.recv_a1 = torch.full((capacity, 4), float("nan"), dtype=torch.bfloat16)
        self.recv_weights = torch.full((capacity, 2), float("inf"))
        self.recv_ids = torch.full((capacity, 2), 5, dtype=torch.int32)
        self.recv = recv
        self.dispatch_kwargs = None
        self.combine_input: torch.Tensor | None = None

    def dispatch(self, input, weights, scales, indices, *, return_routing=True):
        self.dispatch_kwargs = {"return_routing": return_routing}
        return (
            self.recv_a1,
            self.recv_weights,
            None,
            self.recv_ids,
            torch.tensor([self.recv], dtype=torch.int32),
            self.routing,
        )

    def combine(self, input, weights=None, indices=None, *, routing):
        self.combine_input = input
        return torch.zeros(1, 4, dtype=torch.bfloat16), None


def _prepare_with_dp(monkeypatch, op, num_tokens_across_dp, num_dispatchers=4):
    import vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 as m

    if num_tokens_across_dp is None:
        ctx = SimpleNamespace(dp_metadata=None)
    else:
        ctx = SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(num_tokens_across_dp)
            )
        )
    monkeypatch.setattr(m, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(m, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    prepare_finalize = MoriEPv2PrepareAndFinalize(op, 16, num_dispatchers)
    prepared = prepare_finalize.prepare(
        torch.ones(1, 4, dtype=torch.bfloat16),
        torch.ones(1, 2, dtype=torch.bfloat16),
        torch.tensor([[0, 1]]),
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(),
    )
    prepare_finalize.finalize(
        torch.empty(1, 4, dtype=torch.bfloat16),
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )
    return prepared


@pytest.mark.parametrize(
    "num_dispatchers,num_tokens_across_dp,recv,rows",
    [
        # Sum of DP batches (6) rounded up to 32 rows, of 256 capacity rows.
        (4, [1, 3, 2, 0], 7, 32),
        (4, [8, 8, 8, 8], 32, 32),
        (4, [33, 0, 0, 0], 20, 64),
        (4, [0, 0, 0, 0], 0, 32),
        # TP=2 per DP rank: each TP rank may send its DP batch, 2 * 50 rows.
        (4, [30, 20], 90, 128),
        # Dispatchers not a multiple of DP ranks: dispatchers * largest batch.
        (4, [10, 20, 5], 60, 96),
        # Bound above capacity: keep the full buffer.
        (4, [100, 100, 100, 100], 200, 256),
        # No DP metadata: no host-side bound, keep the full buffer.
        (4, None, 7, 256),
    ],
)
def test_mori_epv2_slices_receive_rows_to_dp_bound(
    monkeypatch, num_dispatchers, num_tokens_across_dp, recv, rows
):
    op = FakeCapacityOp(capacity=256, recv=recv)
    prepared = _prepare_with_dp(
        monkeypatch, op, num_tokens_across_dp, num_dispatchers=num_dispatchers
    )

    a1, _, meta, ids, weights = prepared
    for t, buf in ((a1, op.recv_a1), (ids, op.recv_ids), (weights, op.recv_weights)):
        assert t.shape[0] == rows
        assert t.data_ptr() == buf.data_ptr()  # views, no copy
    assert torch.isnan(a1[:recv]).all() and (a1[recv:] == 0).all()
    assert (ids[:recv] == 5).all() and (ids[recv:] == -1).all()
    assert torch.isinf(weights[:recv]).all() and (weights[recv:] == 0).all()
    # Rows past the bound are left untouched.
    assert torch.isnan(op.recv_a1[rows:]).all()
    assert (op.recv_ids[rows:] == 5).all()
    assert int(meta.expert_num_tokens) == recv
    # combine sees only the bounded expert output.
    assert op.combine_input is not None and op.combine_input.shape[0] == rows
    # MORI main's dispatch has no clone_routing parameter.
    assert op.dispatch_kwargs == {"return_routing": True}
