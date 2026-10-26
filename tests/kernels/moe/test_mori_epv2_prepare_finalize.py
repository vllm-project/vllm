# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MORI EPv2 prepare/finalize and all2all manager, with MORI faked out."""

import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 as mori_epv2
from vllm.config.parallel import resolve_mori_all2all
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 import (
    MoriEPv2PrepareAndFinalize,
)
from vllm.utils.network_utils import get_open_port


class FakeOp:
    """MORI EPv2 op whose receive buffers have `capacity` rows, `recv` valid."""

    def __init__(self, capacity=256, recv=0):
        self.routing = object()
        self.recv_a1 = torch.full((capacity, 4), float("nan"), dtype=torch.bfloat16)
        self.recv_weights = torch.full((capacity, 2), float("inf"))
        self.recv_ids = torch.full((capacity, 2), 5, dtype=torch.int32)
        self.recv = recv
        self.combine_in = torch.empty(capacity, 4, dtype=torch.bfloat16)
        self.sent_ids: torch.Tensor | None = None
        self.combine_input: torch.Tensor | None = None
        self.combine_routing = None

    def combine_in_view(self):
        return self.combine_in

    def dispatch(self, input, weights, scales, indices, *, return_routing=True):
        self.sent_ids = indices
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
        self.combine_routing = routing
        return torch.zeros(1, 4, dtype=torch.bfloat16), None


def _make_pf(op, num_local_experts=2, rank_expert_offset=0):
    return MoriEPv2PrepareAndFinalize(op, 16, 4, num_local_experts, rank_expert_offset)


def _run(monkeypatch, op, num_tokens_across_dp, topk_ids, **pf_kwargs):
    """One prepare + finalize round trip under the given DP token counts."""
    if num_tokens_across_dp is None:
        ctx = SimpleNamespace(dp_metadata=None)
    else:
        counts = torch.tensor(num_tokens_across_dp)
        ctx = SimpleNamespace(
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=counts)
        )
    monkeypatch.setattr(mori_epv2, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mori_epv2, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        mori_epv2, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    prepare_finalize = _make_pf(op, **pf_kwargs)
    num_tokens = topk_ids.shape[0]
    prepared = prepare_finalize.prepare(
        torch.ones(num_tokens, 4, dtype=torch.bfloat16),
        torch.ones(num_tokens, 2, dtype=torch.bfloat16),
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(is_per_tensor=False, a1_scale=None),
    )
    prepare_finalize.finalize(
        torch.empty(num_tokens, 4, dtype=torch.bfloat16),
        prepared[0],
        prepared[4],
        prepared[3],
        False,
        SimpleNamespace(),
    )
    return prepared


@pytest.mark.parametrize(
    "num_tokens_across_dp,recv,rows",
    [
        # Sum of DP batches (6) rounded up to 32 rows, of 256 capacity rows.
        ([1, 3, 2, 0], 7, 32),
        ([33, 0, 0, 0], 20, 64),
        # TP=2 per DP rank: each TP rank may send its DP batch, 2 * 50 rows.
        ([30, 20], 90, 128),
        # Dispatchers not a multiple of DP ranks: dispatchers * largest batch.
        ([10, 20, 5], 60, 96),
        # Bound above capacity: keep the full buffer.
        ([100, 100, 100, 100], 200, 256),
        # No DP metadata, or stale metadata missing this rank's batch: full buffer.
        (None, 7, 256),
        ([0, 3, 2, 0], 5, 256),
    ],
)
def test_receive_rows_are_bounded_by_dp_tokens(
    monkeypatch, num_tokens_across_dp, recv, rows
):
    op = FakeOp(recv=recv)
    a1, _, meta, ids, weights = _run(
        monkeypatch, op, num_tokens_across_dp, torch.tensor([[0, 1]])
    )
    for t, buf in ((a1, op.recv_a1), (ids, op.recv_ids), (weights, op.recv_weights)):
        assert t.shape[0] == rows
        assert t.data_ptr() == buf.data_ptr()  # views, no copy
    # Rows past the receive count are cleared; rows past the bound are untouched.
    assert torch.isnan(a1[:recv]).all() and (a1[recv:] == 0).all()
    assert (ids[:recv] == 5).all() and (ids[recv:] == -1).all()
    assert (weights[recv:] == 0).all()
    assert torch.isnan(op.recv_a1[rows:]).all()
    assert int(meta.expert_num_tokens) == recv
    # combine sees only the bounded expert output, under dispatch's routing.
    assert op.combine_input is not None and op.combine_input.shape[0] == rows
    assert op.combine_routing is op.routing


def test_padding_tokens_are_routed_to_local_experts(monkeypatch):
    op = FakeOp(recv=2)
    # Rank 1 of 4 with 2 experts each owns experts 2 and 3. The second token is
    # DP padding, which the router marks with expert id -1.
    topk_ids = torch.tensor([[0, 7], [-1, -1]])
    _run(monkeypatch, op, None, topk_ids, num_local_experts=2, rank_expert_offset=2)
    assert op.sent_ids is not None and op.sent_ids.dtype == torch.int32
    assert op.sent_ids.tolist() == [[0, 7], [2, 3]]


@pytest.mark.parametrize("world_size,lsa_size", [(8, 8), (8, 4)])
def test_manager_requires_one_lsa_team(monkeypatch, world_size, lsa_size):
    from vllm.distributed.device_communicators import all2all
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
    )

    monkeypatch.setattr(
        All2AllManagerBase,
        "__init__",
        lambda self, group: self.__dict__.update(
            rank=0, world_size=world_size, cpu_group=group
        ),
    )
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **k: None
    )
    device_comm = Mock(lsa_size=lsa_size)
    calls = Mock()  # records the order of op.close and comm.destroy
    comm = calls.comm
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
            all2all.MoriEPv2All2AllManager(object())
    else:
        manager = all2all.MoriEPv2All2AllManager(object())
        op = calls.op
        assert manager.handle_cache.get_or_create({}, lambda: op) is op
        manager.destroy()
        # Ops live in the communicator's window: they must go first.
        order = [c[0] for c in calls.mock_calls]
        assert order.index("op.close") < order.index("comm.destroy")
    device_comm.close.assert_called_once()
    comm.destroy.assert_called_once()


def _unique_id_worker(rank, init_method):
    from vllm.distributed.device_communicators import all2all
    from vllm.distributed.device_communicators.base_device_communicator import (
        All2AllManagerBase,
    )

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    # With PP=2 and DP=2 the EP groups are [0, 2] and [1, 3].
    groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    group = groups[rank % 2]

    def fake_init(self, g):
        self.__dict__.update(rank=g.rank(), world_size=g.size(), cpu_group=g)

    All2AllManagerBase.__init__ = fake_init  # type: ignore[method-assign]
    communicator = Mock()
    communicator.get_unique_id.return_value = f"uid-of-rank-{rank}"
    communicator.init.return_value.create_dev_comm.return_value = Mock(lsa_size=2)
    sys.modules["mori.cco"] = SimpleNamespace(
        Communicator=communicator,
        CCODevCommRequirements=SimpleNamespace,
        GDA_CONNECTION_NONE=0,
    )
    all2all.MoriEPv2All2AllManager(group)
    # Every rank joins the communicator with the ID its group's first rank made.
    assert communicator.init.call_args.args[2] == f"uid-of-rank-{rank % 2}"
    dist.destroy_process_group()


def test_unique_id_comes_from_the_ep_groups_first_rank():
    mp.spawn(_unique_id_worker, args=(f"tcp://127.0.0.1:{get_open_port()}",), nprocs=4)


def test_experts_write_into_combine_staging_only_when_it_fits():
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
        TopKWeightAndReduceNoOP,
    )

    op = FakeOp()
    alloc = _make_pf(op).allocate_fused_expert_output
    device = op.combine_in.device
    out = alloc((32, 4), torch.bfloat16, device, TopKWeightAndReduceNoOP())
    assert out is not None and out.shape == (32, 4)
    assert out.data_ptr() == op.combine_in.data_ptr()
    # Unreduced expert output, other dtypes or too many rows keep the workspace.
    for args in (
        ((32, 4), torch.bfloat16, device, TopKWeightAndReduceContiguous()),
        ((32, 4), torch.float32, device, TopKWeightAndReduceNoOP()),
        ((257, 4), torch.bfloat16, device, TopKWeightAndReduceNoOP()),
    ):
        assert alloc(*args) is None


@pytest.mark.parametrize(
    "in_dtype,quant_dtype,quantizes_itself,ok",
    [
        (torch.bfloat16, None, False, True),
        # e.g. AITER, which quantizes after the all2all.
        (torch.bfloat16, torch.float8_e4m3fn, True, True),
        # e.g. FP8 Triton experts, which expect the dispatch to quantize.
        (torch.bfloat16, torch.float8_e4m3fn, False, False),
        (torch.float16, None, False, False),
    ],
)
def test_experts_it_cannot_feed_are_rejected(
    in_dtype, quant_dtype, quantizes_itself, ok
):
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=in_dtype),
        quant_config=SimpleNamespace(quant_dtype=quant_dtype),
        expects_unquantized_inputs=quantizes_itself,
    )
    prepare_finalize = _make_pf(FakeOp(16))
    if ok:
        prepare_finalize.post_init_setup(experts)
    else:
        with pytest.raises(ValueError, match="VLLM_MORI_EP_V2=0"):
            prepare_finalize.post_init_setup(experts)


def _parallel_config(**overrides):
    fields = dict(
        all2all_backend="mori",
        mori_ep_v2=None,
        tensor_parallel_size=1,
        data_parallel_size=8,
        prefill_context_parallel_size=1,
    )
    return SimpleNamespace(**(fields | overrides))


def _model_config(**overrides):
    return SimpleNamespace(**({"dtype": torch.bfloat16} | overrides))


@pytest.mark.parametrize(
    "parallel,model,moe_backend,env,ep_v2",
    [
        ({}, {}, "aiter", {}, True),
        ({"data_parallel_size": 16}, {}, "aiter", {}, False),
        # The LSA team may span hosts (gfx1250); the manager checks it.
        ({"data_parallel_size": 16, "lsa_spans_hosts": True}, {}, "aiter", {}, True),
        ({}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, False),
        # gfx1250 keeps EPv2 under SDMA.
        ({"sdma_picks_epv1": False}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, True),
        ({}, {}, "aiter_mega_moe", {}, False),
        ({}, {"dtype": torch.float16}, "aiter", {}, False),
        # An explicit VLLM_MORI_EP_V2 wins over the deployment checks.
        ({}, {"dtype": torch.float16}, "aiter", {"VLLM_MORI_EP_V2": "1"}, True),
        ({}, {}, "aiter", {"VLLM_MORI_EP_V2": "0"}, False),
        # Already resolved: kept as is.
        ({"mori_ep_v2": True}, {"dtype": torch.float16}, "aiter", {}, True),
        # Other MoRI backends always run EPv1.
        ({"all2all_backend": "mori_high_throughput"}, {}, "aiter", {}, None),
    ],
)
def test_mori_picks_epv2_where_supported(
    monkeypatch, parallel, model, moe_backend, env, ep_v2
):
    monkeypatch.delenv("VLLM_MORI_EP_V2", raising=False)
    monkeypatch.delenv("MORI_ENABLE_SDMA", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    parallel = dict(parallel)
    lsa_spans_hosts = parallel.pop("lsa_spans_hosts", False)
    sdma_picks_epv1 = parallel.pop("sdma_picks_epv1", True)
    parallel_config = _parallel_config(**parallel)
    model_config = _model_config(**model)
    resolve_mori_all2all(
        parallel_config, model_config, moe_backend, lsa_spans_hosts, sdma_picks_epv1
    )
    assert parallel_config.mori_ep_v2 is ep_v2


@pytest.mark.parametrize(
    "backend,internode,max_tokens,kernel",
    [
        ("mori", False, 4096, "IntraNode"),
        ("mori", True, 256, "InterNodeV1LL"),
        ("mori", True, 257, "InterNodeV1"),
        ("mori_high_throughput", True, 128, "InterNodeV1"),
        ("mori_low_latency", True, 4096, "InterNodeV1LL"),
    ],
)
def test_mori_epv1_kernel_choice(monkeypatch, backend, internode, max_tokens, kernel):
    from vllm.distributed.device_communicators import all2all

    kernel_types = SimpleNamespace(
        IntraNode="IntraNode", InterNodeV1="InterNodeV1", InterNodeV1LL="InterNodeV1LL"
    )
    fake_mori = SimpleNamespace(
        ops=SimpleNamespace(EpDispatchCombineKernelType=kernel_types)
    )
    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(on_gfx942=lambda: False, on_gfx950=lambda: True),
    )
    monkeypatch.delenv(
        "VLLM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", raising=False
    )
    manager = object.__new__(all2all.MoriAll2AllManager)
    manager.internode = internode
    manager._all2all_backend = backend
    kwargs = manager._make_all2all_kwargs(
        rank=0,
        num_ep_ranks=16,
        input_dtype=torch.bfloat16,
        quant_dtype=torch.bfloat16,
        token_hidden_size=7168,
        scale_dim=0,
        scale_type_size=4,
        max_num_tokens_per_dp_rank=max_tokens,
        num_local_experts=16,
        num_experts_per_token=8,
    )
    assert kwargs["kernel_type"] == kernel
