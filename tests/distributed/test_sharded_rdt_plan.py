# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side unit tests for the sharded-RDT backend.

The consumer engine's planning core is pure — it turns the baked plan plus the
driver's gather-group partition into a static `_CallPlan` with no pulls, no
model and no Ray — so it is exercised here on CPU/meta tensors only. These tests
pin the values the engine produces today (chunk boundaries, packed byte offsets,
recorded op chains) so a refactor that changes them fails loudly instead of
silently shipping different bytes.

`tests/distributed/test_sharded_rdt_trainer.py` covers the trainer (producer)
side; `test_sharded_rdt_producer.py` covers its serve-actor protocol.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.weight_transfer.base import layerwise_groups
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    ALLOWED_OPS,
    SUPPORTED_OPS,
    RdtRouter,
    arena_alloc_bytes,
    assign_producer_indices,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _dtype_from_name,
)
from vllm.distributed.weight_transfer.sharded_rdt_lazy import (
    BakeSink,
    LazyRDTTensor,
    _Scatter,
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


def _module(layer, copies):
    """What the bake records for one leaf module: a scatter list whose entries
    all carry that module as their ``layer``."""
    for c in copies:
        c.layer = layer
    return copies


def _copy(
    name,
    layer_param="weight",
    *,
    offset=0,
    shape=(4,),
    ops=(),
    layer=None,
    dtype=torch.bfloat16,
):
    """A `_Scatter` as the bake would record it: source key, owning layer, and
    the meta destination region."""
    stride = (
        (1,)
        if len(shape) == 1
        else tuple(
            int(torch.empty(shape, device=META).stride()[i]) for i in range(len(shape))
        )
    )
    return _Scatter(
        layer=layer,
        param_name=layer_param,
        src=(name, tuple(ops)),
        offset=offset,
        shape=tuple(shape),
        stride=stride,
        dtype=dtype,
    )


def _planner(baked, *, name_meta, live=None, held_by=None, group_lens=None):
    """A planning-only engine.

    `__init__` needs a config/model/device the planner never reads, so build the
    instance without it and set exactly the state `_build_call_plan` consumes:
    the baked plan, the name metadata, the live-name set, and a bound router.

    ``held_by``: name -> the ranks holding it. Names left out are held by every
    producer (the replicated class 0). Default: one producer holding everything,
    the degenerate router. Stand-in ranks 0-3 let plan tests assert chunk
    structure; owner VALUES are pinned by TestCallPlanRouting.
    """
    names = list(name_meta)
    if held_by:
        n_prod = max(r for owners in held_by.values() for r in owners) + 1
        owner_sets = [list(range(n_prod))]
        seen = {tuple(range(n_prod)): 0}
        classes = []
        for n in names:
            key = tuple(sorted(held_by.get(n, range(n_prod))))
            if key not in seen:
                seen[key] = len(owner_sets)
                owner_sets.append(list(key))
            classes.append(seen[key])
    else:
        n_prod, owner_sets, classes = 1, [[0]], [0] * len(names)
    eng = object.__new__(ShardedRDTWeightTransferEngine)
    eng._name_to_plan = dict(baked)
    eng._name_meta = dict(name_meta)
    eng._live_names = set(live or name_meta)
    router = RdtRouter(
        n_prod, 1, owner_sets, classes, names, group_lens or [1] * len(names)
    )
    router.bind([object()] * n_prod, [object()] * n_prod, 0)
    eng._router = router
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
        baked[n] = _module(layer, [_copy(n, shape=(numel,))])
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

    def test_the_recorded_dtype_is_the_chains_output_not_the_sources(self):
        """``view(dtype)`` is allowlisted, so a chain can reinterpret dtype. The
        producer packs what the replay yields, so the record must carry the
        POST-chain dtype: taking it from the source name's metadata instead sizes
        the slice with the wrong itemsize and shifts every later slice in the
        chunk, carving the packed blob differently on the two sides."""
        rec, lazy = self._recorder_and_lazy(shape=(4,))
        layer = _FakeLayer("reinterpreted")
        rec.current = (layer, "weight")
        viewed = lazy.view(torch.float32)  # 4 x bf16 -> 2 x f32
        param = torch.empty((2,), dtype=torch.float32, device=META)
        param.copy_(viewed)

        (recorded,) = rec.copies_by_layer[layer]
        assert recorded.dtype is torch.float32, "recorded the source dtype"
        assert recorded.shape == (2,)
        # What the source name alone would have said, and what the producer
        # would NOT have sent.
        assert 2 * torch.float32.itemsize == 8 != 2 * torch.bfloat16.itemsize

    def test_a_broadcasting_copy_is_refused_at_bake(self):
        """The packed layout sizes each slice from the DESTINATION shape while
        the producer packs what the chain yields, and nothing downstream can
        catch a mismatch — the consumer's arena view is exactly prod(dest.shape)
        elements, so it reshapes cleanly over bytes laid out at other offsets.
        So the bake refuses rather than recording a slice it cannot carve."""
        rec, lazy = self._recorder_and_lazy(shape=(1, 6))
        rec.current = (_FakeLayer("broadcast"), "weight")
        param = torch.empty((4, 6), dtype=torch.bfloat16, device=META)
        with pytest.raises(_UnsupportedLazyOp, match="broadcasting"):
            param.copy_(lazy)


# ---------------------------------------------------------------------------
# Consumer identity
# ---------------------------------------------------------------------------


class TestConsumerIdentity:
    """Every worker in the fleet needs a DISTINCT id in 0..C-1: it selects the
    worker's producer block and keys the producer's per-consumer serve ring, so
    a collision silently serves two workers out of one ring."""

    def _engine(self, *, dp_index, rank, world_size):
        eng = object.__new__(ShardedRDTWeightTransferEngine)
        eng.parallel_config = SimpleNamespace(
            data_parallel_index=dp_index, rank=rank, world_size=world_size
        )
        return eng

    def _ids(self, *, num_consumers, workers, replica_rank=0, num_replicas=1):
        """Consumer ids for a DP-only engine whose workers are ``dp_index``es."""
        info = ShardedRDTWeightTransferInitInfo(
            num_consumers=num_consumers,
            replica_rank=replica_rank,
            num_replicas=num_replicas,
        )
        out = []
        for dp_index in workers:
            eng = self._engine(dp_index=dp_index, rank=0, world_size=1)
            eng._num_consumers_override = num_consumers
            out.append(eng._resolve_consumer_id(info))
        return out

    def test_one_engine_indexes_by_dp_and_tp(self):
        """dense-via-TP and MoE-via-DP+EP both flatten to 0..C-1."""
        eng = self._engine(dp_index=0, rank=3, world_size=8)
        eng._num_consumers_override = 8
        info = ShardedRDTWeightTransferInitInfo(num_consumers=8)
        assert eng._resolve_consumer_id(info) == 3

        eng = self._engine(dp_index=5, rank=0, world_size=1)
        eng._num_consumers_override = 8
        assert eng._resolve_consumer_id(info) == 5

    def test_independent_engines_offset_into_distinct_ranges(self):
        """A fleet of separate engines restarts _global_worker_index at 0 in
        each, so without the replica offset every engine would claim 0..w-1.
        The driver sets replica_rank on the payload; these are the ids it buys."""
        ids = [
            self._ids(num_consumers=8, workers=range(4), replica_rank=r, num_replicas=2)
            for r in (0, 1)
        ]
        assert ids == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert len(set(ids[0] + ids[1])) == 8, "every worker distinct fleet-wide"

    def test_the_default_is_a_single_engine_with_no_offset(self):
        assert self._ids(num_consumers=4, workers=range(4)) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# RdtRouter: per-name ownership -> one producer per pull
# ---------------------------------------------------------------------------


def _router(
    num_producers,
    num_consumers,
    owner_sets=None,
    name_owner_class=None,
    names=None,
    group_lens=None,
):
    """A router over ``n`` single-name groups unless told otherwise, so a test
    can state expectations in group indices."""
    if names is None:
        n = len(name_owner_class or []) or (len(group_lens or []) or 1)
        names = [f"g{i}" for i in range(n)]
    if group_lens is None:
        group_lens = [1] * len(names)
    return RdtRouter(
        num_producers, num_consumers, owner_sets, name_owner_class, names, group_lens
    )


class TestRdtRouter:
    """Who serves each name.

    A wrong answer is not a wrong number but a hang or a loud misroute: a
    consumer pulling from a producer that never gathered the name trips its
    served-names guard. Freeing does NOT route this way — every consumer signals
    every owner of a group and the producers count to the live total — so these
    pin PULL routing, plus ``group_owners`` for the free fan-out.
    """

    def test_identity_when_fleets_match(self):
        r = _router(8, 8, group_lens=[1] * 6)
        assert all(r.producer_for(3, f"g{g}") == 3 for g in range(6))
        assert all(r.producer_for(c, "g0") == c for c in range(8))

    def test_every_producer_nic_carries_traffic(self):
        """16 producers / 8 consumers: the block rule spreads each consumer's
        pulls over its block, alternating by group, so no producer sits idle.
        Plain `consumer_id % len(owners)` would idle half of them."""
        r = _router(16, 8, group_lens=[1] * 95)
        names = [f"g{i}" for i in range(95)]
        for c in range(8):
            block = assign_producer_indices(16, 8, c)
            assert {r.producer_for(c, n) for n in names} == set(block)
        assert [r.producer_for(0, f"g{g}") for g in range(4)] == [0, 1, 0, 1]
        assert {r.producer_for(c, n) for c in range(8) for n in names} == set(range(16))

    def test_fan_in_shares_one_producer(self):
        r = _router(2, 8, group_lens=[1] * 5)
        names = [f"g{i}" for i in range(5)]
        assert [{r.producer_for(c, n) for n in names} for c in range(8)] == [
            {c // 4} for c in range(8)
        ]

    def test_pipeline_stages_route_to_the_owning_stage(self):
        """2 stages x 8 ranks: groups 0-2 on stage 0, 3-5 on stage 1. Two owner
        sets, one per stage; each consumer reaches both."""
        r = _router(
            16,
            8,
            owner_sets=[list(range(8)), list(range(8, 16))],
            name_owner_class=[0, 0, 0, 1, 1, 1],
        )
        for c in range(8):
            picks = [r.producer_for(c, f"g{g}") for g in range(6)]
            assert picks == [c] * 3 + [c + 8] * 3
        assert r.group_owners(0) == list(range(8))
        assert r.group_owners(3) == list(range(8, 16))
        r.validate()

    def test_expert_names_route_to_the_holding_ranks(self):
        """A name held by only some ranks routes to those; a replicated name
        keeps the full set."""
        r = RdtRouter(
            8,
            4,
            [[0, 1, 2, 3], [2, 3], [4, 5, 6, 7], [6, 7]],
            [0, 1, 2, 3],
            ["repl0", "expert0", "repl1", "expert1"],
            [2, 2],
        )
        r.validate()
        assert r.owners("expert0") == [2, 3]
        assert r.owners("expert1") == [6, 7]
        assert r.owners("repl1") == [4, 5, 6, 7]
        for c in range(4):
            assert r.producer_for(c, "expert0") in (2, 3)
            assert r.producer_for(c, "repl1") in (4, 5, 6, 7)

    def test_consumers_spread_over_an_owner_set(self):
        """Several ranks hold a name (its TP peers); the block rule must spread
        consumers across them, not funnel through one."""
        r = _router(
            4, 4, owner_sets=[[0, 1]], name_owner_class=[0] * 6, group_lens=[1] * 6
        )
        served = {r.producer_for(c, f"g{g}") for c in range(4) for g in range(6)}
        assert served == {0, 1}

    def test_a_group_frees_at_every_owner_of_any_of_its_names(self):
        """The free barrier is per group, so its fan-out is the UNION over the
        group's names: a rank holding none of them is not signalled."""
        r = RdtRouter(4, 2, [[0, 1], [2, 3]], [0, 1, 0], ["a", "b", "c"], [2, 1])
        assert r.group_owners(0) == [0, 1, 2, 3]
        assert r.group_owners(1) == [0, 1]

    def test_an_unowned_name_raises(self):
        r = RdtRouter(2, 2, [[]], [0], ["w"], [1])
        with pytest.raises(ValueError, match="has no owner"):
            r.producer_for(0, "w")

    def test_validate_rejects_an_empty_owner_set(self):
        with pytest.raises(ValueError, match="empty"):
            RdtRouter(
                4, 2, [[0, 1], [], [2, 3]], [0, 1, 2], ["a", "b", "c"], [3]
            ).validate()

    def test_validate_rejects_an_out_of_range_owner(self):
        with pytest.raises(ValueError, match="out of range"):
            RdtRouter(2, 2, [[0, 5]], [0], ["a"], [1]).validate()

    def test_validate_rejects_an_unresolvable_class(self):
        with pytest.raises(ValueError, match="owner class"):
            RdtRouter(4, 2, [[0, 1]], [0, 7], ["a", "b"], [2]).validate()


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
            copies.append(
                _copy(name, shape=tuple(shape), dtype=_dtype_from_name(dtype_name))
            )
        mod_plan = _module(layer, copies)
        for name, _d, _s in specs:
            baked[name] = mod_plan

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
        mod_plan = _module(layer, copies)
        eng = _planner(
            {n: mod_plan for n, _d, _s in specs},
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
            _copy("w", "a", shape=(4,)),
            _copy("w", "b", shape=(4,)),
        ]
        assert copies[0].src == shared and copies[1].src == shared
        eng = _planner(
            {"w": _module(layer, copies)},
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
    """The chunk cut: one chunk per distinct owner class present in the copies,
    ascending by class index. Derived purely from the bake plus the ownership
    table, so vLLM's expert placement needs no cases."""

    def test_unstamped_copies_form_one_chunk(self):
        layer = _FakeLayer("l")
        copies = [_copy(f"w{i}", shape=(4,)) for i in range(5)]
        eng = _planner({}, name_meta={f"w{i}": ("bfloat16", [4]) for i in range(5)})
        chunks = eng._chunk_module_scatters([_module(layer, copies)])
        assert [ci for ci, _s in chunks] == [0]
        assert len(chunks[0][1]) == 5

    def test_one_chunk_per_owner_class_ascending(self):
        """Classes are numbered by first appearance in metadata order, so the
        replicated class (0) leads and the rest follow — the ordering the old
        "-1 first, then ascending stamp" rule produced, now falling out of the
        table instead of a special case."""
        layer = _FakeLayer("moe")
        names = ["norm", "e5", "e0", "e2"]
        copies = [_copy(n, shape=(4,)) for n in names]
        eng = _planner(
            {},
            name_meta={n: ("bfloat16", [4]) for n in names},
            held_by={"e0": [0], "e2": [1], "e5": [2]},
        )
        chunks = eng._chunk_module_scatters([_module(layer, copies)])
        assert [ci for ci, _s in chunks] == [0, 1, 2, 3]
        assert [[s.src[0] for s in sc] for _ci, sc in chunks] == [
            ["norm"],
            ["e5"],
            ["e0"],
            ["e2"],
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
            held_by={n: [0] for n in names},
        )
        ((_ci, scatters),) = eng._chunk_module_scatters([_module(layer, copies)])
        assert [s.src[0] for s in scatters] == names

    def test_stamps_interleave_across_modules_without_reordering_within(self):
        """Two modules' copies bucket by stamp; within each bucket the order is
        module-then-bake order."""
        a, b = _FakeLayer("a"), _FakeLayer("b")
        mods = [
            _module(a, [_copy("a0", shape=(4,)), _copy("a1", shape=(4,))]),
            _module(b, [_copy("b0", shape=(4,)), _copy("b1", shape=(4,))]),
        ]
        eng = _planner(
            {},
            name_meta={n: ("bfloat16", [4]) for n in ("a0", "a1", "b0", "b1")},
            held_by={"a0": [0], "a1": [1], "b0": [0], "b1": [1]},
        )
        chunks = eng._chunk_module_scatters(mods)
        assert [(ci, [s.src[0] for s in sc]) for ci, sc in chunks] == [
            (1, ["a0", "b0"]),
            (2, ["a1", "b1"]),
        ]

    def test_scatters_carry_their_own_dtype(self):
        """dtype rides the record from the bake, where it is the lazy's dtype
        AFTER its op chain — not a plan-time lookup of the source name, which
        would be wrong for any chain that reinterprets dtype."""
        layer = _FakeLayer("l")
        eng = _planner({}, name_meta={"w": ("float32", [10])})
        ((_er, scatters),) = eng._chunk_module_scatters(
            [_module(layer, [_copy("w", shape=(10,), dtype=torch.float32)])]
        )
        (sc,) = scatters
        assert sc.dtype is torch.float32
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

    def test_stamps_multiply_chunks_within_a_group(self):
        layer = _FakeLayer("l")
        names = [f"w{i}" for i in range(4)]
        copies = [_copy(n, shape=(4,)) for n in names]
        mod_plan = _module(layer, copies)
        eng = _planner(
            {n: mod_plan for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            held_by={"w2": [0], "w3": [1]},
        )
        plan = eng._build_call_plan(names, [4])
        assert len(plan.chunks) == 3  # replicated, w2's class, w3's class

    def test_materialize_fires_on_a_modules_first_chunk_only(self):
        """Empty HF params are allocated once per module, by construction —
        including a FusedMoE-like module whose copies span owner-class chunks."""
        layer = _FakeLayer("spanning")
        names = [f"w{i}" for i in range(4)]
        copies = [_copy(n, shape=(4,)) for n in names]
        mod_plan = _module(layer, copies)
        eng = _planner(
            {n: mod_plan for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            held_by={"w0": [0], "w1": [0], "w2": [1], "w3": [1]},
        )
        plan = eng._build_call_plan(names, [4])
        assert [c.materialize for c in plan.chunks] == [[layer], []]

    def test_quant_defers_to_a_modules_last_chunk(self):
        layer = _FakeLayer("spanning")
        names = [f"w{i}" for i in range(4)]
        copies = [_copy(n, shape=(4,)) for n in names]
        mod_plan = _module(layer, copies)
        eng = _planner(
            {n: mod_plan for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            held_by={"w0": [0], "w1": [0], "w2": [1], "w3": [1]},
        )
        plan = eng._build_call_plan(names, [4])
        assert [c.quant for c in plan.chunks] == [[], [layer]]

    def test_free_signal_fires_on_each_groups_last_chunk(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.free for c in plan.chunks] == [[0], [1], [2], [3]]

    def test_free_signal_waits_for_a_groups_last_chunk_when_it_is_split(self):
        """The load-bearing case: with the group cut across owner-class chunks the
        signal must hang off the LAST one. Signaling earlier lets the producers
        drop the gather buffers while a later chunk's RDMA is still reading."""
        layer = _FakeLayer("l")
        names = [f"w{i}" for i in range(4)]
        copies = [_copy(n, shape=(4,)) for n in names]
        mod_plan = _module(layer, copies)
        eng = _planner(
            {n: mod_plan for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            held_by={"w2": [0], "w3": [1]},
        )
        plan = eng._build_call_plan(names, [4])
        assert len(plan.chunks) == 3
        assert [c.free for c in plan.chunks] == [[], [], [0]]

    def test_free_signal_timing_across_two_split_groups(self):
        """Two groups x 2 owner-class chunks each: signals land on each group's
        last chunk only."""
        names_a = [f"model.layers.0.w{i}" for i in range(4)]
        names_b = [f"model.layers.1.w{i}" for i in range(4)]
        baked, name_meta, stamps = {}, {}, {}
        for names in (names_a, names_b):
            layer = _FakeLayer(names[0])
            mod_plan = _module(layer, [_copy(n, shape=(4,)) for n in names])
            for i, n in enumerate(names):
                baked[n] = mod_plan
                name_meta[n] = ("bfloat16", [4])
                stamps[n] = [i // 2]  # two owner classes per group
        eng = _planner(baked, name_meta=name_meta, held_by=stamps)
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

    def test_modules_are_deduped_within_a_group(self):
        """Several names of one fused module map to the same scatter list; the
        plan must not chunk it twice."""
        layer = _FakeLayer("fused")
        names = ["qkv.q", "qkv.k", "qkv.v"]
        copies = [_copy(n, shape=(4,)) for n in names]
        mod_plan = _module(layer, copies)
        eng = _planner(
            {n: mod_plan for n in names},
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
        assert a.pre_free == b.pre_free

    def test_single_producer_routes_every_chunk_to_local_owner_zero(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.owner for c in plan.chunks] == [0] * len(plan.chunks)


class TestUnbakedNamesGuard:
    """A live name with no baked plan has no way to load — there is no fallback,
    and its pull would target groups the pipeline already freed — so the plan
    build fails at init, naming the names."""

    def test_a_live_unbaked_name_fails_the_plan_build(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["norm.weight"]  # unbaked but live
        eng = _planner(baked, name_meta=name_meta, live=set(names))
        with pytest.raises(RuntimeError, match="norm.weight"):
            eng._build_call_plan(names, group_lens)

    def test_never_copied_names_are_dropped_entirely(self):
        """Experts owned by another EP rank no-op in their loader: not an error,
        and not a chunk — their group is pre-freed if nothing else fills it."""
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["norm.weight"]
        eng = _planner(baked, name_meta=name_meta, live=set(names) - {"norm.weight"})
        plan = eng._build_call_plan(names, group_lens)
        assert len(names) - 1 == len(plan.chunks)
        assert plan.pre_free == [len(group_lens) - 1]


class TestCallPlanRouting:
    """Under partial ownership each chunk must be pulled from a producer that
    actually holds every name in it. Consumers bind EVERY producer (signals fan
    out to all owners), so local index == trainer rank."""

    def _routed_planner(
        self, held_by_group, num_consumers, consumer_id, *, held_by=None
    ):
        """``held_by_group[g]`` = the ranks holding group g's names, so a test
        can still speak in groups; ``held_by`` overrides individual names."""
        baked, name_meta, names, group_lens = _one_module_per_layer(
            len(held_by_group) - 2
        )
        per_name = {n: list(held_by_group[gi]) for gi, n in enumerate(names)}
        per_name.update(held_by or {})
        n_prod = max(max(o) for o in per_name.values()) + 1
        owner_sets: list[list[int]] = []
        seen: dict[tuple, int] = {}
        classes = []
        for n in names:
            key = tuple(sorted(per_name[n]))
            if key not in seen:
                seen[key] = len(owner_sets)
                owner_sets.append(list(key))
            classes.append(seen[key])
        eng = _planner(baked, name_meta=name_meta)
        router = RdtRouter(
            n_prod, num_consumers, owner_sets, classes, names, group_lens
        )
        router.validate()
        router.bind([object()] * n_prod, [object()] * n_prod, consumer_id)
        eng._router = router
        return eng, names, group_lens

    def test_two_stage_ownership_splits_chunks_between_owners(self):
        """Two PP stages, 4 groups: stage 0 holds the first half, stage 1 the
        second. Every chunk goes to a holder of its names."""
        eng, names, group_lens = self._routed_planner([[0], [0], [1], [1]], 1, 0)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.owner for c in plan.chunks] == [0, 0, 1, 1]

    def test_a_multi_owner_group_still_resolves_to_one_producer(self):
        """Every chunk is served by exactly ONE producer: splitting a pull only
        multiplies produce calls, since the consumer's NIC bounds it."""
        eng, names, group_lens = self._routed_planner([[0, 1]] * 4, 1, 0)
        plan = eng._build_call_plan(names, group_lens)
        assert all(0 <= c.owner < 2 for c in plan.chunks)

    def test_an_expert_name_routes_to_the_rank_holding_it(self):
        """PP ∩ EP: stage 0 = ranks 0-1, stage 1 = ranks 2-3, and one expert
        name inside a stage-1 group is held by rank 3 alone. The name's own owner
        set is what routes it."""
        eng, names, group_lens = self._routed_planner(
            [[0, 1], [0, 1], [2, 3], [2, 3]],
            1,
            0,
            held_by={"model.layers.1.w": [3]},
        )
        plan = eng._build_call_plan(names, group_lens)
        assert plan.chunks[2].owner == 3
        assert plan.chunks[0].owner in (0, 1)
        assert plan.chunks[3].owner in (2, 3)

    def test_an_unheld_name_is_rejected_at_init(self):
        """A name no rank holds can never be served. The trainer raises when it
        transposes the holdings; the router's validate() is the consumer-side
        backstop, and either way it fails at init rather than at first pull."""
        with pytest.raises(ValueError, match="empty"):
            RdtRouter(2, 1, [[]], [0], ["w"], [1]).validate()


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
            "embed.weight": _module(
                _FakeLayer("embed"), [_copy("embed.weight", shape=(4,))]
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
            baked[norm] = _module(_FakeLayer(f"norm{li}"), [_copy(norm, shape=(4,))])
            fused = _module(
                _FakeLayer(f"moe{li}"),
                [
                    _copy(f"model.layers.{li}.experts.{e}.w", shape=(4,))
                    for e in worker_experts
                ],
            )
            for e in range(n_experts):
                n = f"model.layers.{li}.experts.{e}.w"
                name_meta[n] = ("bfloat16", [4])
                stamps[n] = [e // n_local]  # held by that EP rank alone
                if e in worker_experts:
                    baked[n] = fused
                    live.add(n)
        names.append("lm_head.weight")
        group_lens.append(1)
        baked["lm_head.weight"] = _module(
            _FakeLayer("head"), [_copy("lm_head.weight", shape=(4,))]
        )
        name_meta["lm_head.weight"] = ("bfloat16", [4])
        live.add("lm_head.weight")
        eng = _planner(baked, name_meta=name_meta, live=live, held_by=stamps)
        return eng, names, group_lens

    @staticmethod
    def _signals(plan):
        return sorted(list(plan.pre_free) + [gi for c in plan.chunks for gi in c.free])

    def test_linear_placement_signals_every_group_exactly_once(self):
        eng, names, group_lens = self._moe_planner(worker_experts=[0, 1, 2, 3])
        plan = eng._build_call_plan(names, group_lens)
        assert self._signals(plan) == list(range(len(group_lens)))

    def test_round_robin_placement_signals_every_group_exactly_once(self):
        eng, names, group_lens = self._moe_planner(worker_experts=[0, 2, 4, 6])
        plan = eng._build_call_plan(names, group_lens)
        assert self._signals(plan) == list(range(len(group_lens)))

    def test_placement_changes_only_the_chunk_count_never_the_signals(self):
        """linear experts 0-3 hit 2 owner classes; round_robin 0,2,4,6
        hits all 4. More chunks per group, identical signal set."""
        plans = {}
        for label, experts in (("linear", [0, 1, 2, 3]), ("round_robin", [0, 2, 4, 6])):
            eng, names, group_lens = self._moe_planner(worker_experts=experts)
            plans[label] = eng._build_call_plan(names, group_lens)
        # per MoE group: the replicated chunk + one per owner class present
        assert (
            len(plans["linear"].chunks) == 2 + 2 * 3
        )  # embed+head + 2 layers x (1 + 2)
        assert len(plans["round_robin"].chunks) == 2 + 2 * 5  # 2 layers x (1 + 4)
        assert self._signals(plans["linear"]) == self._signals(plans["round_robin"])

    def test_the_fused_module_materializes_first_and_quants_last_across_chunks(self):
        """The FusedMoE shape: one module's copies span every owner-class chunk of
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


class TestOpAllowlistAgreement:
    """The consumer records op chains from ``SUPPORTED_OPS``; the producer
    replays a chain only if every op is in ``ALLOWED_OPS``. The two must describe
    the same contract, so they are derived from one table.

    A divergence is silent until first pull: an op the consumer can emit but the
    producer rejects bakes fine at init, then fails a whole sync later with
    "disallowed op".
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
