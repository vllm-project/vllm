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
import vllm.utils.import_utils as import_utils
import vllm.v1.worker.ubatching as ubatching
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


def _make_pf(op, num_local_experts=2, rank_expert_offset=0, async_ll=False):
    return MoriEPv2PrepareAndFinalize(
        [op], 16, 4, num_local_experts, rank_expert_offset, async_ll=async_ll
    )


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
    monkeypatch.setattr(all2all, "CudaRTLibrary", Mock)
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
        cfg = dict(
            scale_dim=0,
            scale_type_size=4,
            num_experts_per_token=8,
            token_nbytes=4096,
            combine_token_nbytes=4096,
            effective_max_recv=1024,
            dispatch_dtype=None,
            combine_dtype=None,
            schedule=None,
        )
        configs: list[dict] = []

        def make_config(**kw):
            configs.append(kw)
            return SimpleNamespace(**(cfg | kw))

        monkeypatch.setitem(
            sys.modules,
            "mori.ops.dispatch_combine_v2",
            SimpleNamespace(
                EpDispatchCombineConfig=make_config,
                EpDispatchCombineOp=lambda config, comm: calls.op,
            ),
        )
        manager = all2all.MoriEPv2All2AllManager(object())
        assert manager.get_handle({}) is calls.op
        # A DBO microbatch's op is a separate op; its id is not a MORI setting.
        manager.get_handle({"ubatch_id": 1})
        assert len(manager._ops) == 2
        assert all("ubatch_id" not in kw for kw in configs)
        # The cache holds ops weakly; the manager keeps them alive itself.
        manager.handle_cache._cache.clear()
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
    all2all.CudaRTLibrary = Mock  # type: ignore[misc]
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
    "in_dtype,quant_dtype,quantizes_itself,per_tensor,ok",
    [
        (torch.bfloat16, None, False, False, True),
        # e.g. AITER, which quantizes after the all2all.
        (torch.bfloat16, torch.float8_e4m3fn, True, False, True),
        # AITER cannot quantize the received rows per tensor.
        (torch.bfloat16, torch.float8_e4m3fn, True, True, False),
        # e.g. FP8 Triton experts, which expect the dispatch to quantize.
        (torch.bfloat16, torch.float8_e4m3fn, False, False, False),
        (torch.float16, None, False, False, False),
    ],
)
def test_experts_it_cannot_feed_are_rejected(
    in_dtype, quant_dtype, quantizes_itself, per_tensor, ok
):
    experts = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=in_dtype),
        quant_config=SimpleNamespace(quant_dtype=quant_dtype, is_per_tensor=per_tensor),
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
        data_parallel_size_local=8,
        data_parallel_external_lb=False,
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        prefill_context_parallel_size=1,
    )
    return SimpleNamespace(**(fields | overrides))


def _model_config(quant_cfg=None, **overrides):
    fields = {"dtype": torch.bfloat16, "quantization": None}
    arch = SimpleNamespace(quantization_config=quant_cfg)
    return SimpleNamespace(model_arch_config=arch, **(fields | overrides))


_EXTERNAL_LB = {"data_parallel_size_local": 1, "data_parallel_external_lb": True}
_FP8_BLOCK = {"quantization": "fp8", "quant_cfg": {"weight_block_size": [128, 128]}}
_MODELOPT_FP8 = {"quantization": "modelopt", "quant_cfg": {"quant_algo": "FP8"}}
_MODELOPT_NVFP4 = {"quantization": "modelopt", "quant_cfg": {"quant_algo": "NVFP4"}}


def _ct(strategy):
    act = {"type": "float", "num_bits": 8, "strategy": strategy}
    cfg = {"config_groups": {"group_0": {"input_activations": act}}}
    return {"quantization": "compressed-tensors", "quant_cfg": cfg}


def _quark(qscheme):
    inputs = {"dtype": "fp8_e4m3", "qscheme": qscheme}
    cfg = {"global_quant_config": {"input_tensors": inputs}}
    return {"quantization": "quark", "quant_cfg": cfg}


@pytest.mark.parametrize(
    "parallel,model,moe_backend,env,ep_v2",
    [
        ({}, {}, "aiter", {}, True),
        # EP across hosts: DP over two nodes, even with ep_size <= 8.
        ({"data_parallel_size": 16}, {}, "aiter", {}, False),
        ({"data_parallel_size_local": 4}, {}, "aiter", {}, False),
        # A DP replica (TP/PP) spanning nodes.
        ({"nnodes_within_dp": 2}, {}, "aiter", {}, False),
        # External LB only sees its own rank: fall back to the EP size.
        (_EXTERNAL_LB, {}, "aiter", {}, True),
        (_EXTERNAL_LB | {"data_parallel_size": 16}, {}, "aiter", {}, False),
        # The LSA team may span hosts (gfx1250); the manager checks it.
        ({"data_parallel_size": 16, "lsa_spans_hosts": True}, {}, "aiter", {}, True),
        ({}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, False),
        # A MoRI with EPv2 AsyncLL moves the tokens by SDMA itself.
        ({"has_async_ll": True}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, True),
        # gfx1250 keeps EPv2 under SDMA.
        ({"sdma_moves_tokens": False}, {}, "aiter", {"MORI_ENABLE_SDMA": "1"}, True),
        ({}, {}, "aiter_mega_moe", {}, False),
        ({}, {"dtype": torch.float16}, "aiter", {}, False),
        # An explicit VLLM_MORI_EP_V2 wins over the deployment checks.
        ({}, {"dtype": torch.float16}, "aiter", {"VLLM_MORI_EP_V2": "1"}, True),
        ({}, {}, "aiter", {"VLLM_MORI_EP_V2": "0"}, False),
        # Already resolved: kept as is.
        ({"mori_ep_v2": True}, {"dtype": torch.float16}, "aiter", {}, True),
        # Other MoRI backends always run EPv1.
        ({"all2all_backend": "mori_high_throughput"}, {}, "aiter", {}, None),
        ({"validated_arch": False}, {}, "aiter", {}, False),
        ({"mori_reason": "too old"}, {}, "aiter", {}, False),
        # Per-tensor FP8 activations; block FP8 and NVFP4 are fine.
        ({}, {"quantization": "fp8"}, "aiter", {}, False),
        ({}, _FP8_BLOCK, "aiter", {}, True),
        ({}, _MODELOPT_FP8, "aiter", {}, False),
        ({}, _MODELOPT_NVFP4, "aiter", {}, True),
        ({}, _ct("tensor"), "aiter", {}, False),
        ({}, _ct("group"), "aiter", {}, True),
        ({}, _quark("per_tensor"), "aiter", {}, False),
        ({}, _quark("per_group"), "aiter", {}, True),
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
    sdma_moves_tokens = parallel.pop("sdma_moves_tokens", True)
    validated_arch = parallel.pop("validated_arch", True)
    mori_reason = parallel.pop("mori_reason", None)
    has_async_ll = parallel.pop("has_async_ll", False)
    monkeypatch.setattr(
        import_utils, "mori_epv2_unavailable_reason", lambda: mori_reason
    )
    monkeypatch.setattr(import_utils, "mori_epv2_has_async_ll", lambda: has_async_ll)
    parallel_config = _parallel_config(**parallel)
    model_config = _model_config(**model)
    resolve_mori_all2all(
        parallel_config,
        model_config,
        moe_backend,
        lsa_spans_hosts,
        sdma_moves_tokens,
        validated_arch,
    )
    assert parallel_config.mori_ep_v2 is ep_v2


@pytest.mark.parametrize(
    "ep_v2,sdma,has_async_ll,sdma_moves_tokens,async_ll",
    [
        (None, True, True, True, True),
        (None, False, True, True, False),
        # MoRI without AsyncLL: SDMA selects EPv1.
        (None, True, False, True, False),
        (False, True, True, True, False),
        # gfx1250 keeps native EPv2 under SDMA.
        (None, True, True, False, False),
    ],
)
def test_mori_epv2_runs_async_ll_under_sdma(
    monkeypatch, ep_v2, sdma, has_async_ll, sdma_moves_tokens, async_ll
):
    monkeypatch.delenv("VLLM_MORI_EP_V2", raising=False)
    monkeypatch.setenv("MORI_ENABLE_SDMA", "1" if sdma else "0")
    monkeypatch.setattr(import_utils, "mori_epv2_unavailable_reason", lambda: None)
    monkeypatch.setattr(import_utils, "mori_epv2_has_async_ll", lambda: has_async_ll)
    parallel_config = _parallel_config(mori_ep_v2=ep_v2)
    resolve_mori_all2all(
        parallel_config, _model_config(), "aiter", sdma_moves_tokens=sdma_moves_tokens
    )
    assert parallel_config.mori_epv2_async_ll is async_ll


@pytest.mark.parametrize(
    "modules,version,usable",
    [
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "1.2.4", True),
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "1.2.5.dev3+g1", True),
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "1.2.3.post1", False),
        ({"mori", "mori.cco", "mori.ops.dispatch_combine_v2"}, "unknown", False),
        ({"mori"}, "1.2.4", False),
        (set(), "1.2.4", False),
    ],
)
def test_mori_epv2_needs_a_recent_mori(monkeypatch, modules, version, usable):
    monkeypatch.setattr(import_utils, "_has_module", lambda name: name in modules)
    monkeypatch.setitem(sys.modules, "mori", SimpleNamespace(__version__=version))
    assert (import_utils.mori_epv2_unavailable_reason() is None) is usable


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


class _DBOProbe:
    """Stands in for vLLM's ubatching: which microbatch runs, and the handoffs."""

    def __init__(self, monkeypatch, enabled=True):
        self.ubatch_id = 0
        self.events: list[tuple] = []
        monkeypatch.setattr(ubatching, "dbo_enabled", lambda: enabled)
        monkeypatch.setattr(mori_epv2, "dbo_current_ubatch_id", lambda: self.ubatch_id)
        monkeypatch.setattr(
            ubatching,
            "dbo_yield_and_switch_from_compute_to_comm",
            lambda: self.events.append(("to_comm", self.ubatch_id)),
        )
        monkeypatch.setattr(
            ubatching,
            "dbo_switch_to_compute_sync",
            lambda: self.events.append(("to_compute", self.ubatch_id)),
        )


class _RecordingOp(FakeOp):
    def __init__(self, name, events, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.events = events

    def dispatch(self, *args, **kwargs):
        self.events.append(("dispatch", self.name))
        return super().dispatch(*args, **kwargs)

    def combine(self, *args, **kwargs):
        self.events.append(("combine", self.name))
        return super().combine(*args, **kwargs)


@pytest.mark.parametrize("num_ubatches", [2, 3])
@pytest.mark.parametrize("num_spec_tokens", [0, 1, 3])
def test_later_microbatch_ops_hold_their_share(
    monkeypatch, num_ubatches, num_spec_tokens
):
    import numpy as np

    import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils
    from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices

    spec = SimpleNamespace(num_speculative_tokens=num_spec_tokens)
    monkeypatch.setattr(
        all2all_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(speculative_config=spec if num_spec_tokens else None),
    )
    max_tokens = 64
    bound = all2all_utils._mori_ubatch_max_tokens(max_tokens, num_ubatches)
    assert bound < max_tokens

    q = 1 + num_spec_tokens
    for num_tokens in range(num_ubatches, max_tokens + 1):
        # Split as gpu/ubatch_utils.create_ubatch_slices does.
        split_point = None
        if q > 1 and num_tokens // (q * num_ubatches) > 0:
            split_point = q * (num_tokens // (q * num_ubatches))
        _, slices = maybe_create_ubatch_slices(
            True,
            np.full(num_tokens, 1, dtype=np.int32),
            num_tokens,
            num_tokens,
            num_ubatches,
            split_point=split_point,
        )
        assert slices is not None
        for s in slices[1:]:
            assert s.token_slice.stop - s.token_slice.start <= bound


def test_dbo_microbatches_interleave_on_their_own_ops(monkeypatch):
    dbo = _DBOProbe(monkeypatch)
    ops = [_RecordingOp(f"op{i}", dbo.events, recv=3 + i) for i in range(2)]
    ops[1].routing = object()
    prepare_finalize = MoriEPv2PrepareAndFinalize(ops, 16, 4, 2, 0)
    # Each microbatch's forward context carries its own DP token counts; the
    # second microbatch is larger, so its receive bound is too.
    contexts = [
        SimpleNamespace(
            dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([n] * 4))
        )
        for n in (4, 9)
    ]
    monkeypatch.setattr(mori_epv2, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mori_epv2, "get_forward_context", lambda: contexts[dbo.ubatch_id]
    )
    monkeypatch.setattr(
        mori_epv2, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    quant_config = SimpleNamespace(is_per_tensor=False, a1_scale=None)

    def prepare(ubatch_id):
        dbo.ubatch_id = ubatch_id
        n = (4, 9)[ubatch_id]
        return prepare_finalize.prepare(
            torch.ones(n, 4, dtype=torch.bfloat16),
            torch.ones(n, 2, dtype=torch.bfloat16),
            torch.zeros(n, 2, dtype=torch.int64),
            num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
        )

    def finalize(ubatch_id, prepared):
        dbo.ubatch_id = ubatch_id
        prepare_finalize.finalize(
            torch.empty((4, 9)[ubatch_id], 4, dtype=torch.bfloat16),
            prepared[0],
            prepared[4],
            prepared[3],
            False,
            SimpleNamespace(),
        )

    # The DBO order: both dispatches, then both combines.
    prepared = [prepare(0), prepare(1)]
    finalize(0, prepared[0])
    finalize(1, prepared[1])

    assert dbo.events == [
        ("to_comm", 0),
        ("dispatch", "op0"),
        ("to_compute", 0),
        ("to_comm", 1),
        ("dispatch", "op1"),
        ("to_compute", 1),
        ("to_comm", 0),
        ("combine", "op0"),
        ("to_compute", 0),
        ("to_comm", 1),
        ("combine", "op1"),
        ("to_compute", 1),
    ]
    for i, op in enumerate(ops):
        a1, _, meta, _, _ = prepared[i]
        # Received rows come from this microbatch's op, bounded by its own
        # token counts: 4 * 4 rows -> 32 and 4 * 9 rows -> 64.
        assert a1.data_ptr() == op.recv_a1.data_ptr()
        assert a1.shape[0] == (32, 64)[i]
        assert int(meta.expert_num_tokens) == 3 + i
        # Each combine used the routing of its own microbatch's dispatch.
        assert op.combine_routing is op.routing
        assert op.combine_input is not None and op.combine_input.shape[0] == a1.shape[0]
    assert prepare_finalize._routings == [None, None]


class _AsyncLLOp(_RecordingOp):
    """AsyncLL op: each phase splits into a send and a receive."""

    def dispatch_send(self, *args, **kwargs):
        self.events.append(("dispatch_send", self.name))
        return FakeOp.dispatch(self, *args, **kwargs)

    def dispatch_recv(self):
        self.events.append(("dispatch_recv", self.name))

    def combine_send(self, *args, **kwargs):
        self.events.append(("combine_send", self.name))
        return FakeOp.combine(self, *args, **kwargs)

    def combine_recv(self):
        self.events.append(("combine_recv", self.name))


def test_async_ll_microbatches_receive_in_the_peers_seam(monkeypatch):
    dbo = _DBOProbe(monkeypatch)
    monkeypatch.setattr(
        mori_epv2,
        "dbo_maybe_run_recv_hook",
        lambda: dbo.events.append(("run_peer_hook", dbo.ubatch_id)),
    )
    monkeypatch.setattr(mori_epv2, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ops = [_AsyncLLOp(f"op{i}", dbo.events, recv=2) for i in range(2)]
    ops[1].routing = object()
    prepare_finalize = MoriEPv2PrepareAndFinalize(ops, 16, 4, 2, 0, async_ll=True)
    assert prepare_finalize.supports_async() and prepare_finalize.supports_dbo()
    quant_config = SimpleNamespace(is_per_tensor=False, a1_scale=None)

    def prepare_async(ubatch_id):
        dbo.ubatch_id = ubatch_id
        return prepare_finalize.prepare_async(
            torch.ones(2, 4, dtype=torch.bfloat16),
            torch.ones(2, 2, dtype=torch.bfloat16),
            torch.zeros(2, 2, dtype=torch.int64),
            8,
            None,
            False,
            quant_config,
        )

    def finalize_async(ubatch_id, prepared, output):
        dbo.ubatch_id = ubatch_id
        return prepare_finalize.finalize_async(
            output, prepared[0], prepared[4], prepared[3], False, SimpleNamespace()
        )

    # The modular kernel's DBO order: a microbatch's receive hook runs on the
    # peer, after the peer's compute is queued.
    hooks, receivers = zip(*(prepare_async(i) for i in range(2)))
    prepared = []
    for i in range(2):
        hooks[i]()
        dbo.ubatch_id = i
        prepared.append(receivers[i]())
    outputs = [torch.empty(2, 4, dtype=torch.bfloat16) for _ in range(2)]
    hooks, receivers = zip(
        *(finalize_async(i, prepared[i], outputs[i]) for i in range(2))
    )
    for i in range(2):
        hooks[i]()
        dbo.ubatch_id = i
        receivers[i]()

    for i, op in enumerate(ops):
        # Each microbatch used its own op, in AsyncLL's call order.
        names = [e[0] for e in dbo.events if e[1] == op.name]
        assert names == [
            "dispatch_send",
            "dispatch_recv",
            "combine_send",
            "combine_recv",
        ]
        assert prepared[i][0].data_ptr() == op.recv_a1.data_ptr()
        assert op.combine_routing is op.routing
        assert (outputs[i] == 0).all()
    # No comm-stream handoff: AsyncLL's transfers leave the compute stream.
    assert not any(e[0] in ("to_comm", "to_compute") for e in dbo.events)
    # finalize_async queues the peer's pending receive before its send.
    assert dbo.events.index(("run_peer_hook", 0)) < dbo.events.index(
        ("combine_send", "op0")
    )
    assert prepare_finalize._routings == [None, None]


class _SegmentedOp(FakeOp):
    """AsyncLL op built with MORI's segmented receive layout."""

    def __init__(self, segmented=True, max_recv_per_rank=64, **kwargs):
        super().__init__(**kwargs)
        self.cfg = SimpleNamespace(
            async_ll_segmented=segmented, effective_max_recv_per_rank=max_recv_per_rank
        )
        self.segment_stride: int | None = None

    def dispatch_send(self, *args, segment_stride=None, **kwargs):
        self.segment_stride = segment_stride
        return FakeOp.dispatch(self, *args, **kwargs)

    def dispatch_recv(self):
        pass

    def combine_send(self, *args, **kwargs):
        return FakeOp.combine(self, *args, **kwargs)

    def combine_recv(self):
        pass


def _prepare_segmented(
    monkeypatch, op, num_tokens_across_dp, num_tokens=4, aiter=True, capturing=False
):
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor(num_tokens_across_dp)
        )
    )
    monkeypatch.setattr(mori_epv2, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mori_epv2, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(
        mori_epv2, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    prepare_finalize = _make_pf(op, async_ll=True)
    prepare_finalize._experts_read_recv_rows_only = aiter
    return prepare_finalize.prepare(
        torch.ones(num_tokens, 4, dtype=torch.bfloat16),
        torch.ones(num_tokens, 2, dtype=torch.bfloat16),
        torch.zeros(num_tokens, 2, dtype=torch.int64),
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=SimpleNamespace(is_per_tensor=False, a1_scale=None),
    )


def test_segmented_receive_is_read_in_place(monkeypatch):
    # 4 ranks of up to 31 tokens: stride 32, 4 * 32 = 128 rows for 120 sent.
    op = _SegmentedOp(recv=128)
    a1, _, meta, ids, weights = _prepare_segmented(monkeypatch, op, [30, 31, 29, 30])
    assert op.segment_stride == 32
    # The whole receive is the experts' input: no bound, no row clearing.
    for t, buf in ((a1, op.recv_a1), (ids, op.recv_ids), (weights, op.recv_weights)):
        assert t.data_ptr() == buf.data_ptr() and t.shape[0] == buf.shape[0]
    assert torch.isnan(a1).all() and (ids == 5).all()
    assert int(meta.expert_num_tokens) == 128


@pytest.mark.parametrize(
    "op_kwargs,num_tokens_across_dp,prepare_kwargs",
    [
        # Uneven DP batches: 4 * 32 rows for 35 sent pads past the bound.
        ({}, [32, 1, 1, 1], {}),
        # The stride must fit MORI's per-source receive.
        ({"max_recv_per_rank": 16}, [30, 31, 29, 30], {}),
        # Ops without the layout, captured graphs and experts that would
        # compute the holes keep the compact receive.
        ({"segmented": False}, [30, 31, 29, 30], {}),
        ({}, [30, 31, 29, 30], {"capturing": True}),
        ({}, [30, 31, 29, 30], {"aiter": False}),
    ],
)
def test_compact_receive_where_segments_do_not_apply(
    monkeypatch, op_kwargs, num_tokens_across_dp, prepare_kwargs
):
    op = _SegmentedOp(recv=7, **op_kwargs)
    a1, _, _, ids, _ = _prepare_segmented(
        monkeypatch, op, num_tokens_across_dp, **prepare_kwargs
    )
    assert op.segment_stride is None
    # The compact receive: bounded rows, the ones past the count cleared.
    assert a1.shape[0] < op.recv_a1.shape[0] and (ids[7:] == -1).all()


def test_rank_beyond_the_stride_raises(monkeypatch):
    # Stale DP counts on one rank must not make it fall back alone.
    op = _SegmentedOp(recv=7)
    with pytest.raises(RuntimeError, match="segment stride"):
        _prepare_segmented(monkeypatch, op, [30, 31, 29, 30], num_tokens=40)
