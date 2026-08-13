# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side unit tests for the sharded-RDT backend.

The consumer engine's planning core is pure — it turns the baked plan plus the
driver's gather-group partition into a static `_CallPlan` with no pulls, no
model and no Ray — so it is exercised here on CPU/meta tensors only. These tests
pin the values the engine produces today (chunk boundaries, packed byte offsets,
recorded op chains) so a refactor that changes them fails loudly instead of
silently shipping different bytes.

`tests/distributed/test_weight_transfer.py` covers the trainer (producer) side.
"""

import pytest
import torch

from vllm.distributed.weight_transfer.base import layerwise_groups
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    ALLOWED_OPS,
    SUPPORTED_OPS,
    RdtRouter,
    arena_alloc_bytes,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    _BakedModule,
    _CallPlan,
    _dtype_from_name,
)
from vllm.distributed.weight_transfer.sharded_rdt_lazy import (
    BakeSink,
    LazyRDTTensor,
    PullSink,
    _BakedCopy,
    _UnsupportedLazyOp,
)

META = torch.device("meta")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _FakeLayer:
    """Stand-in for a leaf module. The planner only stores it and keys on
    ``id()``; nothing in the plan path touches its attributes."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __repr__(self) -> str:
        return f"_FakeLayer({self.tag!r})"


def _copy(name, layer_param="weight", *, offset=0, shape=(4,), ops=()):
    """A `_BakedCopy` as the bake would record it: source key + meta dest region."""
    stride = (
        (1,)
        if len(shape) == 1
        else tuple(
            int(torch.empty(shape, device=META).stride()[i]) for i in range(len(shape))
        )
    )
    return _BakedCopy(
        src=(name, tuple(ops)),
        param_name=layer_param,
        offset=offset,
        shape=tuple(shape),
        stride=stride,
    )


def _planner(baked, *, name_meta, live=None, name_ep_rank=None):
    """A planning-only engine.

    `__init__` needs a config/model/device the planner never reads, so build the
    instance without it and set exactly the state `_build_call_plan` consumes:
    the baked plan, the name metadata, the live-name set and the expert stamps
    (``name_ep_rank``: name -> producer EP coordinate; unstamped = -1).

    Unstamped: one producer that owns everything (the degenerate router).
    Stamped: routing needs a rank per coordinate, so a four-producer router
    with coordinates 0-3 stands in — plan-level tests assert chunk structure,
    not owner values.
    """
    eng = object.__new__(ShardedRDTWeightTransferEngine)
    eng._name_to_module = dict(baked)
    eng._name_meta = dict(name_meta)
    eng._live_names = set(live or name_meta)
    eng._name_ep_rank = {n: er for n, er in (name_ep_rank or {}).items() if er >= 0}
    eng._name_group_idx = {}
    if eng._name_ep_rank:
        eng._produce_methods = [object()] * 4
        eng._router = RdtRouter(4, 1, None, 0, producer_ep_ranks=[0, 1, 2, 3])
    else:
        eng._produce_methods = [object()]
        eng._router = RdtRouter(1, 1, None, 0)
    eng._consumer_id = 0
    return eng


def _one_module_per_layer(n_layers, *, dtype="bfloat16", numel=4):
    """`(baked, name_meta, names, group_lens)` for a pre / N-layer / post model
    where every gather group is one fully-baked leaf module."""
    names = ["embed.weight"]
    names += [f"model.layers.{i}.w" for i in range(n_layers)]
    names += ["norm.weight"]
    baked = {}
    for n in names:
        layer = _FakeLayer(n)
        baked[n] = _BakedModule(layer=layer, copies=[_copy(n, shape=(numel,))])
    name_meta = {n: (dtype, [numel]) for n in names}
    group_lens = [1] * len(names)
    return baked, name_meta, names, group_lens


# ---------------------------------------------------------------------------
# LazyRDTTensor: op-chain recording
# ---------------------------------------------------------------------------


class TestLazyOpChains:
    """Every allowlisted op must append itself to the chain and hand back a
    child whose shape/dtype PyTorch itself computed. The chain is the wire
    format the producer replays, so its exact contents are load-bearing."""

    def _lazy(self, shape=(4, 6), dtype=torch.bfloat16, sink=None):
        return LazyRDTTensor(
            name="w", shape=torch.Size(shape), dtype=dtype, device=META, sink=sink
        )

    def test_bare_lazy_has_metadata_but_no_chain(self):
        t = self._lazy()
        assert t.shape == (4, 6)
        assert t.dtype is torch.bfloat16
        assert t._key() == ("w", ())

    @pytest.mark.parametrize(
        "call,expect_op,expect_shape",
        [
            (lambda t: t.narrow(0, 1, 2), ("narrow", (0, 1, 2), ()), (2, 6)),
            (lambda t: t.view(6, 4), ("view", (6, 4), ()), (6, 4)),
            (lambda t: t.reshape(24), ("reshape", (24,), ()), (24,)),
            (lambda t: t[1], ("__getitem__", (1,), ()), (6,)),
            (lambda t: t.unsqueeze(0), ("unsqueeze", (0,), ()), (1, 4, 6)),
            (lambda t: t.transpose(0, 1), ("transpose", (0, 1), ()), (6, 4)),
            (lambda t: t.t(), ("t", (), ()), (6, 4)),
            (lambda t: t.permute(1, 0), ("permute", (1, 0), ()), (6, 4)),
            (lambda t: t.flatten(), ("flatten", (), ()), (24,)),
            (lambda t: t.contiguous(), ("contiguous", (), ()), (4, 6)),
        ],
    )
    def test_single_return_op_appends_one_spec(self, call, expect_op, expect_shape):
        child = call(self._lazy())
        assert isinstance(child, LazyRDTTensor)
        assert child._key() == ("w", (expect_op,))
        assert tuple(child.shape) == expect_shape

    def test_squeeze_records_its_argument(self):
        child = self._lazy(shape=(1, 4)).squeeze(0)
        assert child._key() == ("w", (("squeeze", (0,), ()),))
        assert tuple(child.shape) == (4,)

    def test_chains_compose_in_call_order(self):
        child = self._lazy().t().narrow(0, 0, 3).flatten()
        assert child._key() == (
            "w",
            (
                ("t", (), ()),
                ("narrow", (0, 0, 3), ()),
                ("flatten", (), ()),
            ),
        )
        assert tuple(child.shape) == (12,)

    def test_kwargs_are_frozen_sorted_for_hashability(self):
        child = self._lazy().narrow(dim=0, start=1, length=2)
        (op, args, kwargs) = child._key()[1][0]
        assert (op, args) == ("narrow", ())
        assert kwargs == (("dim", 0), ("length", 2), ("start", 1))
        hash(child._key())  # the chain is used as a dict key on both sides

    @pytest.mark.parametrize("op,n", [("chunk", 2), ("unbind", 4)])
    def test_multi_return_op_emits_one_child_per_output(self, op, n):
        """chunk/unbind hand back a tuple; each child carries the base op plus a
        trailing __getitem__(i) so the producer can index the replayed result."""
        parts = getattr(self._lazy(), op)(*((n,) if op == "chunk" else ()), 0)
        assert isinstance(parts, tuple)
        assert len(parts) == n
        for i, part in enumerate(parts):
            base, index = part._key()[1]
            assert base[0] == op
            assert index == ("__getitem__", (i,), ())

    def test_op_chain_is_hashable_as_a_fetch_key(self):
        keys = {self._lazy().t()._key(), self._lazy().t()._key()}
        assert len(keys) == 1, "equal chains must collapse — they dedup pull keys"


class TestLazyUnsupportedOps:
    """Anything that needs real data must fail loudly at bake time rather than
    silently transferring the wrong bytes."""

    def _lazy(self):
        return LazyRDTTensor(
            name="w",
            shape=torch.Size((4,)),
            dtype=torch.bfloat16,
            device=META,
            sink=None,
        )

    @pytest.mark.parametrize(
        "call",
        [
            lambda t: t.to(torch.float32),
            lambda t: t.float(),
            lambda t: t + 1,
            lambda t: t * 2,
            lambda t: t.sum(),
        ],
        ids=["to", "float", "add", "mul", "sum"],
    )
    def test_data_dependent_ops_raise(self, call):
        with pytest.raises(_UnsupportedLazyOp):
            call(self._lazy())

    def test_error_names_the_weight_and_the_chain(self):
        with pytest.raises(_UnsupportedLazyOp) as exc:
            self._lazy().narrow(0, 0, 2).float()
        msg = str(exc.value)
        assert "'w'" in msg
        assert "narrow" in msg

    def test_unsupported_is_a_notimplementederror(self):
        """Callers distinguish "this backend can't handle the loader" from bugs."""
        assert issubclass(_UnsupportedLazyOp, NotImplementedError)


class TestBakeRecording:
    """During the dry run the lazy's ``copy_`` is the data sink: it records the
    source chain plus the meta destination's strided region and moves nothing."""

    def _recorder_and_lazy(self, shape=(4, 6)):
        rec = BakeSink()
        lazy = LazyRDTTensor(
            name="w",
            shape=torch.Size(shape),
            dtype=torch.bfloat16,
            device=META,
            sink=rec,
        )
        return rec, lazy

    def test_copy_records_the_destination_region(self):
        rec, lazy = self._recorder_and_lazy()
        layer = _FakeLayer("q_proj")
        param = torch.empty((8, 6), dtype=torch.bfloat16, device=META)
        dest = param.narrow(0, 4, 4)  # the second half of a fused param
        rec.current = (layer, "weight")
        dest.copy_(lazy)

        (recorded,) = rec.copies_by_layer[layer]
        assert recorded.src == ("w", ())
        assert recorded.param_name == "weight"
        assert recorded.offset == dest.storage_offset() == 24
        assert recorded.shape == (4, 6)
        assert recorded.stride == (6, 1)

    def test_copy_marks_the_source_name_live(self):
        rec, lazy = self._recorder_and_lazy()
        rec.current = (_FakeLayer("l"), "weight")
        torch.empty((4, 6), dtype=torch.bfloat16, device=META).copy_(lazy)
        assert rec.copied_names == {"w"}

    def test_unattributed_copy_is_live_but_unrecorded(self):
        """A copy_ with no loader stamp cannot be attributed to a param, so its
        module must fall back to the plain load — but the name still moved data."""
        rec, lazy = self._recorder_and_lazy()
        rec.current = None
        torch.empty((4, 6), dtype=torch.bfloat16, device=META).copy_(lazy)
        assert rec.copied_names == {"w"}
        assert dict(rec.copies_by_layer) == {}

    def test_copies_are_grouped_by_module_in_call_order(self):
        rec, _ = self._recorder_and_lazy()
        layer = _FakeLayer("gate_up")
        for i, name in enumerate(("gate", "up")):
            lazy = LazyRDTTensor(
                name=name,
                shape=torch.Size((4,)),
                dtype=torch.bfloat16,
                device=META,
                sink=rec,
            )
            param = torch.empty((8,), dtype=torch.bfloat16, device=META)
            rec.current = (layer, "weight")
            param.narrow(0, 4 * i, 4).copy_(lazy)
        assert [c.src[0] for c in rec.copies_by_layer[layer]] == ["gate", "up"]
        assert [c.offset for c in rec.copies_by_layer[layer]] == [0, 4]

    def test_recording_a_sliced_source(self):
        """The chain on the source and the region on the dest are independent."""
        rec, lazy = self._recorder_and_lazy(shape=(8, 6))
        layer = _FakeLayer("k_proj")
        rec.current = (layer, "weight")
        param = torch.empty((4, 6), dtype=torch.bfloat16, device=META)
        param.copy_(lazy.narrow(0, 2, 4))
        (recorded,) = rec.copies_by_layer[layer]
        assert recorded.src == ("w", (("narrow", (0, 2, 4), ()),))
        assert recorded.shape == (4, 6)


# ---------------------------------------------------------------------------
# Packed layout: the cross-process invariant
# ---------------------------------------------------------------------------


def _producer_pack_offsets(slices):
    """The producer's rule, transcribed from ``rdt_produce_weights_batched``
    (sharded_rdt_trainer.py): 16B-aligned offsets in specs order.

    Kept as an independent implementation on purpose: the consumer computing the
    same offsets is the invariant that makes the packed blob readable, and this
    is the only thing that catches the two rules drifting apart. A divergence is
    silent at runtime — the bytes still arrive, they are just carved up wrong.
    """
    pack_cur = 0
    offsets = []
    for numel, dtype in slices:
        off = (pack_cur + 15) & ~15
        pack_cur = off + numel * dtype.itemsize
        offsets.append(off)
    return offsets, pack_cur


class TestPackedLayout:
    def test_consumer_layout_matches_the_producer_rule(self):
        """Mixed dtypes, sizes that do not land on 16B — i.e. the Kimi-style
        group (fp8 weights + fp32 scales + bf16 norms) that exposes alignment."""
        specs = [
            ("w.fp8", "float8_e4m3fn", [17]),
            ("w.scale", "float32", [3]),
            ("w.norm", "bfloat16", [5]),
            ("w.big", "bfloat16", [1024]),
        ]
        baked, name_meta = {}, {}
        layer = _FakeLayer("mixed")
        copies = []
        for name, dtype_name, shape in specs:
            name_meta[name] = (dtype_name, shape)
            copies.append(_copy(name, shape=tuple(shape)))
        group = _BakedModule(layer=layer, copies=copies)
        for name, _d, _s in specs:
            baked[name] = group

        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan([n for n, _, _ in specs], [len(specs)])
        (chunk,) = plan.chunks

        want_offsets, want_bytes = _producer_pack_offsets(
            [(s[0], _dtype_from_name(d)) for _n, d, s in specs]
        )
        assert [off for off, _dt, _n, _shape in chunk.pack_layout] == want_offsets
        assert chunk.pack_bytes == want_bytes

    def test_every_offset_is_16b_aligned(self):
        specs = [(f"w{i}", "bfloat16", [i * 3 + 1]) for i in range(6)]
        layer = _FakeLayer("l")
        copies = [_copy(n, shape=tuple(s)) for n, _d, s in specs]
        group = _BakedModule(layer=layer, copies=copies)
        eng = _planner(
            {n: group for n, _d, _s in specs},
            name_meta={n: (d, s) for n, d, s in specs},
        )
        (chunk,) = eng._build_call_plan([n for n, _, _ in specs], [len(specs)]).chunks
        assert all(off % 16 == 0 for off, _dt, _n, _s in chunk.pack_layout)

    def test_layout_carries_dtype_numel_and_shape_for_the_view_carve(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(1, numel=6)
        eng = _planner(baked, name_meta=name_meta)
        chunk = eng._build_call_plan(names, group_lens).chunks[0]
        (off, dt, numel, shape) = chunk.pack_layout[0]
        assert (off, dt, numel, shape) == (0, torch.bfloat16, 6, (6,))

    def test_duplicate_source_keys_are_packed_once(self):
        """Two destinations fed by the SAME (name, chain) pull one slice."""
        layer = _FakeLayer("l")
        shared = ("w", ())
        copies = [
            _BakedCopy(src=shared, param_name="a", offset=0, shape=(4,), stride=(1,)),
            _BakedCopy(src=shared, param_name="b", offset=0, shape=(4,), stride=(1,)),
        ]
        eng = _planner(
            {"w": _BakedModule(layer=layer, copies=copies)},
            name_meta={"w": ("bfloat16", [4])},
        )
        (chunk,) = eng._build_call_plan(["w"], [1]).chunks
        assert len(chunk.scatters) == 2
        assert chunk.keys == [shared]
        assert len(chunk.pack_layout) == 1


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------


class TestChunkModuleScatters:
    """The chunk cut: one chunk per distinct producer ep_rank present in the
    copies — ``-1`` (replicated) first, then ascending. Derived purely from the
    bake plus the name stamps, so vLLM's expert placement needs no cases."""

    def test_unstamped_copies_form_one_chunk(self):
        layer = _FakeLayer("l")
        copies = [_copy(f"w{i}", shape=(4,)) for i in range(5)]
        eng = _planner({}, name_meta={f"w{i}": ("bfloat16", [4]) for i in range(5)})
        chunks = eng._chunk_module_scatters([_BakedModule(layer=layer, copies=copies)])
        assert [er for er, _s in chunks] == [-1]
        assert len(chunks[0][1]) == 5

    def test_one_chunk_per_ep_rank_minus_one_first_then_ascending(self):
        layer = _FakeLayer("moe")
        names = ["norm", "e5", "e0", "e2"]
        copies = [_copy(n, shape=(4,)) for n in names]
        eng = _planner(
            {},
            name_meta={n: ("bfloat16", [4]) for n in names},
            name_ep_rank={"e0": 0, "e2": 1, "e5": 2},
        )
        chunks = eng._chunk_module_scatters([_BakedModule(layer=layer, copies=copies)])
        assert [er for er, _s in chunks] == [-1, 0, 1, 2]
        assert [[s.src[0] for s in sc] for _er, sc in chunks] == [
            ["norm"],
            ["e0"],
            ["e2"],
            ["e5"],
        ]

    def test_copy_order_within_a_chunk_is_bake_order(self):
        """Scatter order within a chunk must follow the bake, not the stamp
        sort: the packed layout and the producer's replay agree on keys order."""
        layer = _FakeLayer("moe")
        names = ["e3", "e1", "e2"]  # one stamp, deliberately unsorted names
        copies = [_copy(n, shape=(4,)) for n in names]
        eng = _planner(
            {},
            name_meta={n: ("bfloat16", [4]) for n in names},
            name_ep_rank={n: 0 for n in names},
        )
        ((er, scatters),) = eng._chunk_module_scatters(
            [_BakedModule(layer=layer, copies=copies)]
        )
        assert er == 0
        assert [s.src[0] for s in scatters] == names

    def test_stamps_interleave_across_modules_without_reordering_within(self):
        """Two modules' copies bucket by stamp; within each bucket the order is
        module-then-bake order."""
        a, b = _FakeLayer("a"), _FakeLayer("b")
        mods = [
            _BakedModule(
                layer=a, copies=[_copy("a0", shape=(4,)), _copy("a1", shape=(4,))]
            ),
            _BakedModule(
                layer=b, copies=[_copy("b0", shape=(4,)), _copy("b1", shape=(4,))]
            ),
        ]
        eng = _planner(
            {},
            name_meta={n: ("bfloat16", [4]) for n in ("a0", "a1", "b0", "b1")},
            name_ep_rank={"a0": 0, "a1": 1, "b0": 0, "b1": 1},
        )
        chunks = eng._chunk_module_scatters(mods)
        assert [(er, [s.src[0] for s in sc]) for er, sc in chunks] == [
            (0, ["a0", "b0"]),
            (1, ["a1", "b1"]),
        ]

    def test_scatters_carry_their_own_dtype_and_nbytes(self):
        layer = _FakeLayer("l")
        eng = _planner({}, name_meta={"w": ("float32", [10])})
        ((_er, scatters),) = eng._chunk_module_scatters(
            [_BakedModule(layer=layer, copies=[_copy("w", shape=(10,))])]
        )
        (sc,) = scatters
        assert sc.dtype is torch.float32
        assert sc.nbytes == 40
        assert sc.layer is layer


# ---------------------------------------------------------------------------
# _build_call_plan
# ---------------------------------------------------------------------------


class TestBuildCallPlan:
    def test_one_chunk_per_group_when_unstamped(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(3)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert len(plan.chunks) == len(group_lens) == 5
        assert plan.pre_free == []
        assert plan.residual == []

    def test_stamps_multiply_chunks_within_a_group(self):
        layer = _FakeLayer("l")
        names = [f"w{i}" for i in range(4)]
        copies = [_copy(n, shape=(4,)) for n in names]
        group = _BakedModule(layer=layer, copies=copies)
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            name_ep_rank={"w2": 0, "w3": 1},
        )
        plan = eng._build_call_plan(names, [4])
        assert len(plan.chunks) == 3  # -1, 0, 1

    def test_materialize_fires_on_a_modules_first_chunk_only(self):
        """Empty HF params are allocated once per module, by construction —
        including a FusedMoE-like module whose copies span ep_rank chunks."""
        layer = _FakeLayer("spanning")
        names = [f"w{i}" for i in range(4)]
        group = _BakedModule(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            name_ep_rank={"w0": 0, "w1": 0, "w2": 1, "w3": 1},
        )
        plan = eng._build_call_plan(names, [4])
        assert [c.materialize for c in plan.chunks] == [[layer], []]

    def test_quant_defers_to_a_modules_last_chunk(self):
        layer = _FakeLayer("spanning")
        names = [f"w{i}" for i in range(4)]
        group = _BakedModule(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            name_ep_rank={"w0": 0, "w1": 0, "w2": 1, "w3": 1},
        )
        plan = eng._build_call_plan(names, [4])
        assert [c.quant for c in plan.chunks] == [[], [layer]]

    def test_free_signal_fires_on_each_groups_last_chunk(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.free for c in plan.chunks] == [[0], [1], [2], [3]]

    def test_free_signal_waits_for_a_groups_last_chunk_when_it_is_split(self):
        """The load-bearing case: with the group cut across ep_rank chunks the
        signal must hang off the LAST one. Signaling earlier lets the producers
        drop the gather buffers while a later chunk's RDMA is still reading."""
        layer = _FakeLayer("l")
        names = [f"w{i}" for i in range(4)]
        group = _BakedModule(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            name_ep_rank={"w2": 0, "w3": 1},
        )
        plan = eng._build_call_plan(names, [4])
        assert len(plan.chunks) == 3
        assert [c.free for c in plan.chunks] == [[], [], [0]]

    def test_free_signal_timing_across_two_split_groups(self):
        """Two groups x 2 ep_rank chunks each: signals land on each group's
        last chunk only."""
        names_a = [f"model.layers.0.w{i}" for i in range(4)]
        names_b = [f"model.layers.1.w{i}" for i in range(4)]
        baked, name_meta, stamps = {}, {}, {}
        for names in (names_a, names_b):
            layer = _FakeLayer(names[0])
            group = _BakedModule(
                layer=layer, copies=[_copy(n, shape=(4,)) for n in names]
            )
            for i, n in enumerate(names):
                baked[n] = group
                name_meta[n] = ("bfloat16", [4])
                stamps[n] = i // 2  # two stamps per group
        eng = _planner(baked, name_meta=name_meta, name_ep_rank=stamps)
        plan = eng._build_call_plan(names_a + names_b, [4, 4])
        assert [c.free for c in plan.chunks] == [[], [0], [], [1]]

    def test_a_group_with_nothing_local_is_signaled_at_sync_start(self):
        """The owners still published it, so this consumer must signal it —
        immediately, before the pipeline (signal-before-publish is tolerated),
        never hung off another group's chunk."""
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["model.layers.1.w"]  # nothing baked, and not live => no pull
        eng = _planner(
            baked, name_meta=name_meta, live=set(names) - {"model.layers.1.w"}
        )
        plan = eng._build_call_plan(names, group_lens)
        assert len(plan.chunks) == 3
        assert plan.pre_free == [2]
        assert [c.free for c in plan.chunks] == [[0], [1], [3]]

    def test_leading_groups_with_nothing_local_go_to_pre_free(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["embed.weight"]
        eng = _planner(baked, name_meta=name_meta, live=set(names) - {"embed.weight"})
        plan = eng._build_call_plan(names, group_lens)
        assert plan.pre_free == [0]
        assert len(plan.chunks) == 3

    def test_live_but_unbaked_names_become_residual(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["norm.weight"]  # unbaked but still live => plain load
        eng = _planner(baked, name_meta=name_meta, live=set(names))
        plan = eng._build_call_plan(names, group_lens)
        assert plan.residual == ["norm.weight"]

    def test_names_that_never_copied_are_dropped_entirely(self):
        """Experts owned by another EP rank no-op in their loader; paying
        _load_unbaked for them every sync would be pure waste."""
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["norm.weight"]
        eng = _planner(baked, name_meta=name_meta, live=set(names) - {"norm.weight"})
        plan = eng._build_call_plan(names, group_lens)
        assert plan.residual == []

    def test_the_plan_carries_a_name_to_module_map(self):
        """It selects the owning producer for a residual name's on-demand pull.
        The planner returns it rather than writing engine state, so the plan stays
        a pure function of its inputs."""
        baked, name_meta, names, group_lens = _one_module_per_layer(3)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert plan.name_group_idx == {n: i for i, n in enumerate(names)}

    def test_modules_are_deduped_within_a_group(self):
        """Several names of one fused module map to the same _BakedModule; the
        plan must not chunk it twice."""
        layer = _FakeLayer("fused")
        names = ["qkv.q", "qkv.k", "qkv.v"]
        group = _BakedModule(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
        )
        plan = eng._build_call_plan(names, [3])
        (chunk,) = plan.chunks
        assert chunk.materialize == [layer]
        assert len(chunk.scatters) == 3

    def test_plan_is_a_pure_function_of_its_inputs(self):
        """Called twice with the same arguments it must produce the same plan —
        that is what lets the engine build it once and reuse it every sync."""
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        eng = _planner(baked, name_meta=name_meta)
        a = eng._build_call_plan(names, group_lens)
        b = eng._build_call_plan(names, group_lens)
        assert [c.pack_bytes for c in a.chunks] == [c.pack_bytes for c in b.chunks]
        assert [c.keys for c in a.chunks] == [c.keys for c in b.chunks]
        assert [c.free for c in a.chunks] == [c.free for c in b.chunks]
        assert a.residual == b.residual and a.pre_free == b.pre_free

    def test_single_producer_routes_every_chunk_to_local_owner_zero(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.owner for c in plan.chunks] == [0] * len(plan.chunks)


class TestResidualGuard:
    """Residual (live-but-unbaked) names load AFTER the pipeline — after this
    consumer signaled every group, i.e. after the producers freed them. On an
    expert-sharded model that is a guaranteed stall-watchdog death, so the plan
    build fails at init instead."""

    def _static_plan(self, eng, names, group_lens):
        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            ShardedRDTWeightTransferInitInfo,
        )

        eng._build_static_plan(
            ShardedRDTWeightTransferInitInfo(names=names, group_lens=group_lens)
        )

    def test_residual_on_a_stamped_model_fails_at_init(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["norm.weight"]  # unbaked but live => residual
        eng = _planner(
            baked,
            name_meta=name_meta,
            live=set(names),
            name_ep_rank={"model.layers.0.w": 0},
        )
        with pytest.raises(RuntimeError, match="residual"):
            self._static_plan(eng, names, group_lens)

    def test_residual_on_an_unstamped_model_stays_a_plain_load(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["norm.weight"]
        eng = _planner(baked, name_meta=name_meta, live=set(names))
        self._static_plan(eng, names, group_lens)
        assert eng._cached_plan.residual == ["norm.weight"]


class TestCallPlanRouting:
    """Under partial ownership each chunk must be pulled from a producer that
    actually holds its (group, ep_rank) unit. Consumers bind EVERY producer
    (signals fan out to all owners), so local index == trainer rank."""

    def _routed_planner(
        self,
        group_owners,
        num_consumers,
        consumer_id,
        *,
        producer_ep_ranks=None,
        **planner_kw,
    ):
        from vllm.distributed.weight_transfer.sharded_rdt_common import RdtRouter

        baked, name_meta, names, group_lens = _one_module_per_layer(
            len(group_owners) - 2
        )
        eng = _planner(baked, name_meta=name_meta, **planner_kw)
        n_prod = max(max(o) for o in group_owners) + 1
        router = RdtRouter(
            num_producers=n_prod,
            num_consumers=num_consumers,
            group_owners=group_owners,
            num_groups=len(group_owners),
            producer_ep_ranks=producer_ep_ranks,
        )
        router.validate()
        eng._router = router
        eng._consumer_id = consumer_id
        eng._produce_methods = [object()] * n_prod
        return eng, names, group_lens

    def test_two_stage_ownership_splits_chunks_between_owners(self):
        """Two PP stages, 4 groups: stage 0 owns the first half, stage 1 the
        second. Every chunk goes to an owner of its group."""
        owners = [[0], [0], [1], [1]]
        eng, names, group_lens = self._routed_planner(owners, 1, 0)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.owner for c in plan.chunks] == [0, 0, 1, 1]

    def test_a_multi_owner_group_still_resolves_to_one_producer(self):
        """Every pull unit is served by exactly ONE producer per consumer:
        splitting a pull only multiplies produce calls, since the consumer's NIC
        bounds it."""
        owners = [[0, 1], [0, 1], [0, 1], [0, 1]]
        eng, names, group_lens = self._routed_planner(owners, 1, 0)
        plan = eng._build_call_plan(names, group_lens)
        assert all(isinstance(c.owner, int) for c in plan.chunks)
        assert all(0 <= c.owner < 2 for c in plan.chunks)

    def test_an_expert_chunk_routes_to_the_matching_coordinate_owner(self):
        """PP ∩ EP: 2 stages x ep2, coords [0, 1, 0, 1]. A chunk stamped 1 in a
        stage-1 group must land on the ONE rank in stage 1 with coordinate 1."""
        owners = [[0, 1], [0, 1], [2, 3], [2, 3]]
        # _one_module_per_layer(2): groups = embed, layers.0, layers.1, norm.
        # Stamp group 2's single name with coordinate 1.
        eng, names, group_lens = self._routed_planner(
            owners,
            1,
            0,
            producer_ep_ranks=[0, 1, 0, 1],
            name_ep_rank={"model.layers.1.w": 1},
        )
        plan = eng._build_call_plan(names, group_lens)
        assert plan.chunks[2].owner == 3
        # Unstamped chunks stay within their group's owner set.
        assert plan.chunks[0].owner in (0, 1)
        assert plan.chunks[3].owner in (2, 3)

    def test_an_unowned_pull_unit_fails_at_plan_build(self):
        """A stamped name whose coordinate has no rank inside the group's owner
        set must raise at init (plan build), not hang at first pull."""
        owners = [[0, 1], [0, 1], [0, 1], [0, 1]]
        eng, names, group_lens = self._routed_planner(
            owners,
            1,
            0,
            producer_ep_ranks=[0, 0],
            name_ep_rank={"model.layers.1.w": 1},
        )
        with pytest.raises(ValueError, match="has no owner"):
            eng._build_call_plan(names, group_lens)


class TestSignalCompleteness:
    """Every gather group is signaled exactly once — after its last chunk when
    the worker pulls from it, at sync start otherwise — for ANY expert
    placement. A group signaled twice over-credits the barrier; a group never
    signaled parks a producer credit and hangs end_sync."""

    def _moe_planner(self, worker_experts, *, ep_size=4, n_experts=8, n_layers=2):
        """pre / n_layers MoE layers / post. Each layer: one norm (its own
        module, stamp -1) + n_experts expert names stamped ``e // n_local``.
        The worker's bake covers only ``worker_experts`` (its placement);
        foreign expert names never copied => dropped entirely, as in the real
        bake."""
        n_local = n_experts // ep_size
        names = ["embed.weight"]
        group_lens = [1]
        baked = {
            "embed.weight": _BakedModule(
                layer=_FakeLayer("embed"), copies=[_copy("embed.weight", shape=(4,))]
            )
        }
        name_meta = {"embed.weight": ("bfloat16", [4])}
        stamps: dict = {}
        live = {"embed.weight"}
        for li in range(n_layers):
            norm = f"model.layers.{li}.norm"
            enames = [f"model.layers.{li}.experts.{e}.w" for e in range(n_experts)]
            names += [norm] + enames
            group_lens.append(1 + n_experts)
            name_meta[norm] = ("bfloat16", [4])
            live.add(norm)
            baked[norm] = _BakedModule(
                layer=_FakeLayer(f"norm{li}"), copies=[_copy(norm, shape=(4,))]
            )
            fused = _BakedModule(
                layer=_FakeLayer(f"moe{li}"),
                copies=[
                    _copy(f"model.layers.{li}.experts.{e}.w", shape=(4,))
                    for e in worker_experts
                ],
            )
            for e in range(n_experts):
                n = f"model.layers.{li}.experts.{e}.w"
                name_meta[n] = ("bfloat16", [4])
                stamps[n] = e // n_local
                if e in worker_experts:
                    baked[n] = fused
                    live.add(n)
        names.append("lm_head.weight")
        group_lens.append(1)
        baked["lm_head.weight"] = _BakedModule(
            layer=_FakeLayer("head"), copies=[_copy("lm_head.weight", shape=(4,))]
        )
        name_meta["lm_head.weight"] = ("bfloat16", [4])
        live.add("lm_head.weight")
        eng = _planner(baked, name_meta=name_meta, live=live, name_ep_rank=stamps)
        return eng, names, group_lens

    @staticmethod
    def _signals(plan):
        return sorted(list(plan.pre_free) + [gi for c in plan.chunks for gi in c.free])

    def test_linear_placement_signals_every_group_exactly_once(self):
        eng, names, group_lens = self._moe_planner(worker_experts=[0, 1, 2, 3])
        plan = eng._build_call_plan(names, group_lens)
        assert self._signals(plan) == list(range(len(group_lens)))
        assert plan.residual == []

    def test_round_robin_placement_signals_every_group_exactly_once(self):
        eng, names, group_lens = self._moe_planner(worker_experts=[0, 2, 4, 6])
        plan = eng._build_call_plan(names, group_lens)
        assert self._signals(plan) == list(range(len(group_lens)))
        assert plan.residual == []

    def test_placement_changes_only_the_chunk_count_never_the_signals(self):
        """linear experts 0-3 hit 2 producer coordinates; round_robin 0,2,4,6
        hits all 4. More chunks per group, identical signal set."""
        plans = {}
        for label, experts in (("linear", [0, 1, 2, 3]), ("round_robin", [0, 2, 4, 6])):
            eng, names, group_lens = self._moe_planner(worker_experts=experts)
            plans[label] = eng._build_call_plan(names, group_lens)
        # per MoE group: norm chunk (-1) + one chunk per coordinate present
        assert (
            len(plans["linear"].chunks) == 2 + 2 * 3
        )  # embed+head + 2 layers x (1 + 2)
        assert len(plans["round_robin"].chunks) == 2 + 2 * 5  # 2 layers x (1 + 4)
        assert self._signals(plans["linear"]) == self._signals(plans["round_robin"])

    def test_the_fused_module_materializes_first_and_quants_last_across_chunks(self):
        """The FusedMoE shape: one module's copies span every ep_rank chunk of
        its group; materialize on its first, quant on its last."""
        eng, names, group_lens = self._moe_planner(
            worker_experts=[0, 2, 4, 6], n_layers=1
        )
        plan = eng._build_call_plan(names, group_lens)
        # group 1 = [norm chunk, coord 0..3 chunks] at chunk indices 1..5
        fused_chunks = [
            c
            for c in plan.chunks[1:6]
            if any("experts" in sc.src[0] for sc in c.scatters)
        ]
        assert len(fused_chunks) == 4
        (fused_layer,) = fused_chunks[0].materialize
        assert fused_layer.tag == "moe0"
        assert fused_chunks[0].quant == []
        assert fused_chunks[-1].quant == [fused_layer]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestArenaAllocBytes:
    def test_rounds_up_to_a_coarse_256mb_boundary(self):
        assert arena_alloc_bytes(1) == 256 << 20
        assert arena_alloc_bytes(256 << 20) == 256 << 20
        assert arena_alloc_bytes((256 << 20) + 1) == 512 << 20

    def test_presize_is_a_floor_not_a_cap(self):
        assert arena_alloc_bytes(1, presize=3 << 30) == 3 << 30
        big = 4 << 30
        assert arena_alloc_bytes(big, presize=1 << 30) == big

    def test_never_returns_less_than_requested(self):
        for nbytes in (1, 1 << 20, (700 << 20) + 3):
            assert arena_alloc_bytes(nbytes) >= nbytes


class TestLayerwiseGroups:
    def test_partition_is_pre_then_one_group_per_layer_then_post(self):
        names = [
            "embed.weight",
            "model.layers.0.a",
            "model.layers.0.b",
            "model.layers.1.a",
            "norm.weight",
        ]
        assert layerwise_groups(names) == [
            ["embed.weight"],
            ["model.layers.0.a", "model.layers.0.b"],
            ["model.layers.1.a"],
            ["norm.weight"],
        ]

    def test_flattening_a_partition_returns_the_input_order(self):
        """The partition is by POSITION relative to the first layer name, not by
        name class — so a post-block name that looks like a pre-block name still
        lands in post. Group index therefore means the same thing on every rank."""
        names = [
            "embed.weight",
            "model.layers.0.a",
            "embed.weight.tied",  # after the layers => post, despite the name
        ]
        groups = layerwise_groups(names)
        assert [n for g in groups for n in g] == names

    def test_layers_are_ordered_by_index_not_appearance(self):
        names = ["model.layers.10.a", "model.layers.2.a"]
        assert layerwise_groups(names) == [
            ["model.layers.2.a"],
            ["model.layers.10.a"],
        ]

    def test_no_empty_groups(self):
        for names in ([], ["only.pre"], ["model.layers.0.a"]):
            assert all(g for g in layerwise_groups(names))

    def test_a_model_with_no_layers_is_one_pre_group(self):
        assert layerwise_groups(["a", "b"]) == [["a", "b"]]


class TestDtypeFromName:
    def test_resolves_wire_names(self):
        assert _dtype_from_name("bfloat16") is torch.bfloat16
        assert _dtype_from_name("float8_e4m3fn") is torch.float8_e4m3fn

    def test_rejects_a_non_dtype_attribute(self):
        with pytest.raises(ValueError):
            _dtype_from_name("nn")  # torch.nn exists but is not a dtype

    def test_rejects_an_unknown_name(self):
        with pytest.raises(ValueError):
            _dtype_from_name("float9")


class TestPullSink:
    """The plain-load path: one slice pulled on demand per copy_. Only reached
    for names with no baked plan (attention scales, partial layers)."""

    def _lazy(self, sink, shape=(4,)):
        return LazyRDTTensor(
            name="w",
            shape=torch.Size(shape),
            dtype=torch.bfloat16,
            device=META,
            sink=sink,
        )

    def test_a_meta_destination_pulls_nothing(self):
        """Layerwise reload drives pass 1 against a meta-restored param; there is
        no data to copy yet, but the numel must still count or
        _layerwise_process never fires for the layer."""
        calls = []
        sink = PullSink(lambda keys: calls.append(keys))
        dest = torch.empty((4,), dtype=torch.bfloat16, device=META)
        dest.copy_(self._lazy(sink))
        assert calls == []

    def test_a_real_destination_pulls_and_copies_the_slice(self):
        want = torch.arange(4, dtype=torch.bfloat16)
        blob = want.view(torch.uint8).clone()
        pulled = []

        def _pull(keys):
            pulled.append(keys)
            return blob

        dest = torch.zeros((4,), dtype=torch.bfloat16)
        dest.copy_(self._lazy(PullSink(_pull)))
        assert torch.equal(dest, want)
        assert pulled == [[("w", ())]]

    def test_the_op_chain_is_what_gets_requested(self):
        pulled = []
        blob = torch.zeros(2, dtype=torch.bfloat16).view(torch.uint8).clone()

        def _pull(keys):
            pulled.append(keys)
            return blob

        dest = torch.zeros((2,), dtype=torch.bfloat16)
        dest.copy_(self._lazy(PullSink(_pull)).narrow(0, 1, 2))
        assert pulled == [[("w", (("narrow", (0, 1, 2), ()),))]]

    def test_fetch_reads_back_only_the_slices_bytes(self):
        """The producer answers with one packed blob; a single-key request is
        that slice at offset 0, and trailing arena bytes must be ignored."""
        payload = torch.arange(3, dtype=torch.bfloat16)
        blob = torch.cat(
            [payload.view(torch.uint8), torch.full((32,), 0xFF, dtype=torch.uint8)]
        )
        sink = PullSink(lambda keys: blob)
        got = sink.fetch(("w", ()), torch.Size((3,)), torch.bfloat16)
        assert torch.equal(got, payload)


class TestUnbakedPullRouting:
    """A residual pull must go to the producer that owns the name's group AND
    carry this worker's consumer id.

    The consumer id keys the producer's per-consumer serve rings. Omitting it
    made every worker's residual pull default to consumer 0, so concurrent pulls
    from different workers were served out of one ring and overwrote each other's
    packed blob.
    """

    @pytest.fixture(autouse=True)
    def _no_ray(self, monkeypatch):
        """The sink's closure wraps its call in `ray.get`; the fake producer
        already returns the resolved value, so make `ray.get` the identity."""
        import ray

        monkeypatch.setattr(ray, "get", lambda x: x)

    def _engine(self, consumer_id, name_group_idx, owners_by_group):
        eng = object.__new__(ShardedRDTWeightTransferEngine)
        eng._name_meta = {n: ("bfloat16", [2]) for n in name_group_idx}
        eng._name_ep_rank = {}
        eng._cached_plan = _CallPlan(
            chunks=[], pre_free=[], residual=[], name_group_idx=dict(name_group_idx)
        )
        eng._consumer_id = consumer_id
        n_prod = max(max(o) for o in owners_by_group) + 1
        # num_consumers=8 so the harness's consumer ids (up to 7) are real
        # fleet positions rather than out-of-range indices.
        eng._router = RdtRouter(n_prod, 8, owners_by_group, len(owners_by_group))
        eng._produce_methods = [_RecordingProducer(p) for p in range(n_prod)]
        return eng

    def test_the_pull_carries_this_workers_consumer_id(self):
        eng = self._engine(7, {"w": 0}, [[0]])
        eng._pull_sink_for("w").fetch(("w", ()), torch.Size((2,)), torch.bfloat16)
        assert eng._produce_methods[0].calls == [([("w", ())], 7)]

    def test_two_workers_do_not_share_a_serve_ring(self):
        ids = []
        for consumer_id in (0, 3):
            eng = self._engine(consumer_id, {"w": 0}, [[0]])
            eng._pull_sink_for("w").fetch(("w", ()), torch.Size((2,)), torch.bfloat16)
            ids.append(eng._produce_methods[0].calls[0][1])
        assert ids == [0, 3]

    def test_a_name_is_pulled_from_the_owner_of_its_group(self):
        eng = self._engine(0, {"a": 0, "b": 1}, [[0], [1]])
        for name in ("a", "b"):
            eng._pull_sink_for(name).fetch((name, ()), torch.Size((2,)), torch.bfloat16)
        assert [p.calls for p in eng._produce_methods] == [
            [([("a", ())], 0)],
            [([("b", ())], 0)],
        ]


class _RecordingProducer:
    """Stands in for a Ray actor's bound producer method: records `(keys,
    consumer_id)` and answers the way the real one does, with a one-element list
    holding the packed blob."""

    def __init__(self, rank):
        self.rank = rank
        self.calls = []

    def remote(self, keys, consumer_id=0):
        self.calls.append((list(keys), consumer_id))
        return [torch.zeros(64, dtype=torch.uint8)]


class TestOpAllowlistAgreement:
    """The consumer records op chains from ``SUPPORTED_OPS``; the producer
    replays a chain only if every op is in ``ALLOWED_OPS``. The two must describe
    the same contract, so they are derived from one table.

    They used to be written out twice and had drifted: ``t`` was
    consumer-emittable but producer-rejected, so a loader calling ``.t()`` baked
    successfully at init and then failed at first pull with "disallowed op 't'".
    """

    def test_the_two_allowlists_describe_the_same_contract(self):
        assert set(SUPPORTED_OPS.values()) == set(ALLOWED_OPS)

    def test_every_bake_recordable_op_is_replayable(self):
        """Anything the bake can record must be serveable, or the failure lands a
        whole sync later than the mistake."""
        assert not set(SUPPORTED_OPS.values()) - set(ALLOWED_OPS)

    def test_the_producer_allows_nothing_the_consumer_cannot_emit(self):
        """The allowlist is also a guard against a misbehaving or spoofed
        consumer invoking arbitrary methods on trainer tensors, so it must not be
        wider than the ops the bake can actually produce. ``to`` in particular
        would let a replay change dtype or device."""
        assert not set(ALLOWED_OPS) - set(SUPPORTED_OPS.values())
        assert "to" not in ALLOWED_OPS

    def test_transpose_via_t_is_serveable(self):
        """The specific regression: a chain recorded from ``.t()`` must pass the
        producer's guard."""
        lazy = LazyRDTTensor(
            name="w",
            shape=torch.Size((4, 6)),
            dtype=torch.bfloat16,
            device=META,
            sink=None,
        )
        (op, _args, _kwargs) = lazy.t()._key()[1][0]
        assert op in ALLOWED_OPS
