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

from vllm.distributed.weight_transfer.sharded_rdt_common import (
    ALLOWED_OPS,
    RdtRouter,
    arena_alloc_bytes,
    greedy_run_starts,
    layerwise_groups,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    _SUPPORTED_OPS,
    LazyRDTTensor,
    ShardedRDTWeightTransferEngine,
    _BakedCopy,
    _BakedGroup,
    _BakeRecorder,
    _dtype_from_name,
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


def _planner(baked, *, name_meta, live=None, split=1):
    """A planning-only engine.

    `__init__` needs a config/model/device the planner never reads, so build the
    instance without it and set exactly the state `_build_call_plan` consumes:
    the baked plan, the name metadata, the live-name set and the chunk split.
    With a single bound producer `_local_owner_of` short-circuits to 0, but it
    asserts on the router before reaching that shortcut, so a trivial
    gather-to-all router is installed too.
    """
    eng = object.__new__(ShardedRDTWeightTransferEngine)
    eng._name_to_group = dict(baked)
    eng._name_meta = dict(name_meta)
    eng._live_names = set(live or name_meta)
    eng._split = split
    eng._name_group_idx = {}
    eng._produce_methods = [object()]  # len <= 1 => _local_owner_of returns 0
    eng._router = RdtRouter(1, 1, None, 0)
    eng._consumer_id = 0
    eng._local_of_producer = {0: 0}
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
        baked[n] = _BakedGroup(layer=layer, copies=[_copy(n, shape=(numel,))])
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

    def _lazy(self, shape=(4, 6), dtype=torch.bfloat16, ctx=None):
        return LazyRDTTensor(
            name="w", shape=torch.Size(shape), dtype=dtype, device=META, ctx=ctx
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
            ctx=None,
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
        rec = _BakeRecorder()
        lazy = LazyRDTTensor(
            name="w",
            shape=torch.Size(shape),
            dtype=torch.bfloat16,
            device=META,
            ctx=rec,
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
                ctx=rec,
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

    Kept as an independent implementation on purpose — the consumer computing
    the same offsets is the invariant that makes the packed blob readable, and
    the ``pack_check`` diagnostic exists only to detect drift between the two.
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
        group = _BakedGroup(layer=layer, copies=copies)
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
        group = _BakedGroup(layer=layer, copies=copies)
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
            {"w": _BakedGroup(layer=layer, copies=copies)},
            name_meta={"w": ("bfloat16", [4])},
        )
        (chunk,) = eng._build_call_plan(["w"], [1]).chunks
        assert len(chunk.scatters) == 2
        assert chunk.keys == [shared]
        assert len(chunk.pack_layout) == 1


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------


class TestChunkGroupScatters:
    def test_split_one_keeps_the_group_whole(self):
        layer = _FakeLayer("l")
        copies = [_copy(f"w{i}", shape=(4,)) for i in range(5)]
        eng = _planner(
            {f"w{i}": _BakedGroup(layer=layer, copies=copies) for i in range(5)},
            name_meta={f"w{i}": ("bfloat16", [4]) for i in range(5)},
            split=1,
        )
        chunks = eng._chunk_group_scatters([_BakedGroup(layer=layer, copies=copies)])
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_split_three_cuts_byte_balanced_runs(self):
        layer = _FakeLayer("l")
        copies = [_copy(f"w{i}", shape=(4,)) for i in range(6)]
        eng = _planner(
            {}, name_meta={f"w{i}": ("bfloat16", [4]) for i in range(6)}, split=3
        )
        chunks = eng._chunk_group_scatters([_BakedGroup(layer=layer, copies=copies)])
        assert [len(c) for c in chunks] == [2, 2, 2]

    def test_a_single_oversized_copy_becomes_its_own_chunk(self):
        """Copies are atomic, so an untied lm_head cannot be subdivided — it
        just makes its run oversized. This is what arena_presize_gb is for."""
        layer = _FakeLayer("l")
        copies = [_copy("small", shape=(1,)), _copy("huge", shape=(1000,))]
        eng = _planner(
            {},
            name_meta={"small": ("bfloat16", [1]), "huge": ("bfloat16", [1000])},
            split=2,
        )
        chunks = eng._chunk_group_scatters([_BakedGroup(layer=layer, copies=copies)])
        assert [[s.src[0] for s in c] for c in chunks] == [["small"], ["huge"]]

    def test_a_single_copy_never_splits(self):
        layer = _FakeLayer("l")
        eng = _planner({}, name_meta={"w": ("bfloat16", [8])}, split=4)
        chunks = eng._chunk_group_scatters(
            [_BakedGroup(layer=layer, copies=[_copy("w", shape=(8,))])]
        )
        assert len(chunks) == 1

    def test_scatters_carry_their_own_dtype_and_nbytes(self):
        layer = _FakeLayer("l")
        eng = _planner({}, name_meta={"w": ("float32", [10])})
        (scatters,) = eng._chunk_group_scatters(
            [_BakedGroup(layer=layer, copies=[_copy("w", shape=(10,))])]
        )
        (sc,) = scatters
        assert sc.dtype is torch.float32
        assert sc.nbytes == 40
        assert sc.layer is layer


# ---------------------------------------------------------------------------
# _build_call_plan
# ---------------------------------------------------------------------------


class TestBuildCallPlan:
    def test_one_chunk_per_group_at_split_one(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(3)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert len(plan.chunks) == len(group_lens) == 5
        assert plan.pre_free == []
        assert plan.residual == []

    def test_split_multiplies_chunks_within_a_group(self):
        layer = _FakeLayer("l")
        names = [f"w{i}" for i in range(4)]
        copies = [_copy(n, shape=(4,)) for n in names]
        group = _BakedGroup(layer=layer, copies=copies)
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            split=2,
        )
        plan = eng._build_call_plan(names, [4])
        assert len(plan.chunks) == 2

    def test_materialize_fires_on_a_modules_first_chunk_only(self):
        """Empty HF params are allocated once per module, by construction."""
        layer = _FakeLayer("spanning")
        names = [f"w{i}" for i in range(4)]
        group = _BakedGroup(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            split=2,
        )
        plan = eng._build_call_plan(names, [4])
        assert [c.materialize for c in plan.chunks] == [[layer], []]

    def test_quant_defers_to_a_modules_last_chunk(self):
        layer = _FakeLayer("spanning")
        names = [f"w{i}" for i in range(4)]
        group = _BakedGroup(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            split=2,
        )
        plan = eng._build_call_plan(names, [4])
        assert [c.quant for c in plan.chunks] == [[], [layer]]

    def test_free_fires_on_each_groups_last_chunk(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        eng = _planner(baked, name_meta=name_meta)
        plan = eng._build_call_plan(names, group_lens)
        assert [c.free for c in plan.chunks] == [
            [(0, ["embed.weight"])],
            [(1, ["model.layers.0.w"])],
            [(2, ["model.layers.1.w"])],
            [(3, ["norm.weight"])],
        ]

    def test_free_waits_for_a_groups_last_chunk_when_it_is_split(self):
        """The load-bearing case: with the group split across chunks the free
        must hang off the LAST one. Freeing earlier lets the trainer drop the
        gather buffers while a later chunk's RDMA is still reading them."""
        layer = _FakeLayer("l")
        names = [f"w{i}" for i in range(4)]
        group = _BakedGroup(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
        eng = _planner(
            {n: group for n in names},
            name_meta={n: ("bfloat16", [4]) for n in names},
            split=2,
        )
        plan = eng._build_call_plan(names, [4])
        assert len(plan.chunks) == 2
        assert [c.free for c in plan.chunks] == [[], [(0, names)]]

    def test_free_timing_across_two_split_groups(self):
        """Two groups x 2 chunks each: frees land on chunks 1 and 3 only."""
        names_a = [f"model.layers.0.w{i}" for i in range(4)]
        names_b = [f"model.layers.1.w{i}" for i in range(4)]
        baked, name_meta = {}, {}
        for names in (names_a, names_b):
            layer = _FakeLayer(names[0])
            group = _BakedGroup(
                layer=layer, copies=[_copy(n, shape=(4,)) for n in names]
            )
            for n in names:
                baked[n] = group
                name_meta[n] = ("bfloat16", [4])
        eng = _planner(baked, name_meta=name_meta, split=2)
        plan = eng._build_call_plan(names_a + names_b, [4, 4])
        assert [c.free for c in plan.chunks] == [
            [],
            [(0, names_a)],
            [],
            [(1, names_b)],
        ]

    def test_a_group_with_nothing_local_is_freed_after_the_previous_chunk(self):
        """The trainer gathered it under the lockstep plan, so it must still be
        freed — hung off the last chunk this worker does pull."""
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["model.layers.1.w"]  # nothing baked, and not live => no pull
        eng = _planner(
            baked, name_meta=name_meta, live=set(names) - {"model.layers.1.w"}
        )
        plan = eng._build_call_plan(names, group_lens)
        assert len(plan.chunks) == 3
        assert plan.chunks[1].free == [
            (1, ["model.layers.0.w"]),
            (2, ["model.layers.1.w"]),
        ]

    def test_leading_groups_with_nothing_local_go_to_pre_free(self):
        """No chunk exists yet to hang the free on, so it fires before the
        pipeline starts."""
        baked, name_meta, names, group_lens = _one_module_per_layer(2)
        del baked["embed.weight"]
        eng = _planner(baked, name_meta=name_meta, live=set(names) - {"embed.weight"})
        plan = eng._build_call_plan(names, group_lens)
        assert plan.pre_free == [(0, ["embed.weight"])]
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

    def test_name_to_group_index_covers_every_name(self):
        baked, name_meta, names, group_lens = _one_module_per_layer(3)
        eng = _planner(baked, name_meta=name_meta)
        eng._build_call_plan(names, group_lens)
        assert eng._name_group_idx == {n: i for i, n in enumerate(names)}

    def test_modules_are_deduped_within_a_group(self):
        """Several names of one fused module map to the same _BakedGroup; the
        plan must not chunk it twice."""
        layer = _FakeLayer("fused")
        names = ["qkv.q", "qkv.k", "qkv.v"]
        group = _BakedGroup(layer=layer, copies=[_copy(n, shape=(4,)) for n in names])
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
        for chunk in plan.chunks:
            (owner, run_keys, start, end) = chunk.subpulls[0]
            assert owner == 0
            assert run_keys == chunk.keys
            assert (start, end) == (0, chunk.pack_bytes)


class TestCallPlanRouting:
    """Under partial ownership each chunk must be pulled from — and each group
    freed at — the one producer that gathered it."""

    def _routed_planner(self, group_owners, num_consumers, consumer_id, n_local):
        from vllm.distributed.weight_transfer.sharded_rdt_common import RdtRouter

        baked, name_meta, names, group_lens = _one_module_per_layer(
            len(group_owners) - 2
        )
        eng = _planner(baked, name_meta=name_meta)
        router = RdtRouter(
            num_producers=max(max(o) for o in group_owners) + 1,
            num_consumers=num_consumers,
            group_owners=group_owners,
            num_groups=len(group_owners),
        )
        router.validate()
        eng._router = router
        eng._consumer_id = consumer_id
        eng._produce_methods = [object()] * n_local
        eng._local_of_producer = {
            p: i for i, p in enumerate(router.bound_producers(consumer_id))
        }
        return eng, names, group_lens

    def test_two_stage_ownership_splits_chunks_between_owners(self):
        """Two PP stages, 4 groups: stage 0 owns the first half, stage 1 the
        second. Every chunk goes to an owner of its group."""
        owners = [[0], [0], [1], [1]]
        eng, names, group_lens = self._routed_planner(owners, 1, 0, 2)
        plan = eng._build_call_plan(names, group_lens)
        locals_ = [c.subpulls[0][0] for c in plan.chunks]
        assert locals_ == [0, 0, 1, 1]

    def test_every_group_is_served_by_exactly_one_owner(self):
        owners = [[0, 1], [0, 1], [0, 1], [0, 1]]
        eng, names, group_lens = self._routed_planner(owners, 1, 0, 2)
        plan = eng._build_call_plan(names, group_lens)
        for chunk in plan.chunks:
            assert len(chunk.subpulls) == 1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestGreedyRunStarts:
    def test_equal_weights_split_evenly(self):
        assert greedy_run_starts([1] * 6, 3) == [0, 2, 4]

    def test_never_emits_more_than_n_runs(self):
        assert len(greedy_run_starts([1] * 100, 4)) == 4

    def test_first_start_is_always_zero(self):
        assert greedy_run_starts([5, 1, 1], 3)[0] == 0

    def test_an_item_heavier_than_the_target_makes_its_run_oversized(self):
        # target = ceil(102/2) = 51; the 100 cannot be subdivided.
        assert greedy_run_starts([1, 100, 1], 2) == [0, 1]

    def test_single_run_when_n_is_one(self):
        assert greedy_run_starts([3, 4, 5], 1) == [0]

    def test_empty_weights(self):
        assert greedy_run_starts([], 3) == [0]


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
    """The consumer records op chains from ``_SUPPORTED_OPS``; the producer
    replays them only if every op is in ``ALLOWED_OPS``. The two must describe
    the same contract or a loader bakes at init and fails at first pull."""

    @pytest.mark.xfail(
        strict=True,
        reason="known defect: 't' is consumer-emittable but producer-rejected, "
        "and 'to'/'split'/'select' are producer-allowed but unreachable",
    )
    def test_the_two_allowlists_describe_the_same_contract(self):
        assert set(_SUPPORTED_OPS.values()) == set(ALLOWED_OPS)

    def test_consumer_emittable_ops_are_all_producer_allowed(self):
        """The direction that matters: anything the bake can record must be
        replayable. Currently violated by ``t``."""
        unserveable = sorted(set(_SUPPORTED_OPS.values()) - set(ALLOWED_OPS))
        assert unserveable == ["t"], (
            "the set of bake-recordable-but-unserveable ops changed; a loader "
            f"using any of {unserveable} bakes at init and fails at first pull"
        )

    def test_producer_allows_nothing_that_needs_data(self):
        """``to`` would let a replay change dtype or device — exactly what the
        bake refuses. It is unreachable today only because the consumer cannot
        emit it."""
        assert "to" in ALLOWED_OPS
        assert "to" not in set(_SUPPORTED_OPS.values())
