# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side unit tests for the sharded-RDT backend.

The consumer engine's planning core is pure — it turns the baked plan plus the
driver's gather-group partition into a static `_CallPlan` with no pulls, no
model and no Ray — so most of it is exercised here on CPU/meta tensors only.
These tests pin the values the engine produces today (chunk boundaries, packed
byte offsets, recorded op chains) so a refactor that changes them fails loudly
instead of silently shipping different bytes.

`TestBakeOnARealModel` at the end is the exception: it needs a GPU, because the
bake only exists to be driven through a real `model.load_weights`, and the parts
that matter there — that the monkeypatched loaders are put back, and that the
recorded op chains reproduce what the loaders did — cannot be faked.

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
    assign_producer_indices,
    buffer_alloc_bytes,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
    ShardedRDTWeightTransferInitInfo,
    _dtype_from_name,
)
from vllm.distributed.weight_transfer.sharded_rdt_fake import (
    BakeSink,
    FakeRDTTensor,
    _Scatter,
    _UnsupportedFakeOp,
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
# FakeRDTTensor: op-chain recording
# ---------------------------------------------------------------------------


class TestFakeOpChains:
    """Every allowlisted op must append itself to the chain and hand back a
    child whose shape/dtype PyTorch itself computed. The chain is the wire
    format the producer replays, so its exact contents are load-bearing."""

    def _fake(self, shape=(4, 6), dtype=torch.bfloat16, sink=None):
        return FakeRDTTensor(
            name="w", shape=torch.Size(shape), dtype=dtype, device=META, sink=sink
        )

    def test_bare_fake_has_metadata_but_no_chain(self):
        t = self._fake()
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
        child = call(self._fake())
        assert isinstance(child, FakeRDTTensor)
        assert child._key() == ("w", (expect_op,))
        assert tuple(child.shape) == expect_shape

    def test_squeeze_records_its_argument(self):
        child = self._fake(shape=(1, 4)).squeeze(0)
        assert child._key() == ("w", (("squeeze", (0,), ()),))
        assert tuple(child.shape) == (4,)

    def test_chains_compose_in_call_order(self):
        child = self._fake().t().narrow(0, 0, 3).flatten()
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
        child = self._fake().narrow(dim=0, start=1, length=2)
        (op, args, kwargs) = child._key()[1][0]
        assert (op, args) == ("narrow", ())
        assert kwargs == (("dim", 0), ("length", 2), ("start", 1))
        hash(child._key())  # the chain is used as a dict key on both sides

    @pytest.mark.parametrize("op,n", [("chunk", 2), ("unbind", 4)])
    def test_multi_return_op_emits_one_child_per_output(self, op, n):
        """chunk/unbind hand back a tuple; each child carries the base op plus a
        trailing __getitem__(i) so the producer can index the replayed result."""
        parts = getattr(self._fake(), op)(*((n,) if op == "chunk" else ()), 0)
        assert isinstance(parts, tuple)
        assert len(parts) == n
        for i, part in enumerate(parts):
            base, index = part._key()[1]
            assert base[0] == op
            assert index == ("__getitem__", (i,), ())

    def test_op_chain_is_hashable_as_a_fetch_key(self):
        keys = {self._fake().t()._key(), self._fake().t()._key()}
        assert len(keys) == 1, "equal chains must collapse — they dedup pull keys"


class TestFakeUnsupportedOps:
    """Anything that needs real data must fail loudly at bake time rather than
    silently transferring the wrong bytes."""

    def _fake(self):
        return FakeRDTTensor(
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
        with pytest.raises(_UnsupportedFakeOp):
            call(self._fake())

    def test_error_names_the_weight_and_the_chain(self):
        with pytest.raises(_UnsupportedFakeOp) as exc:
            self._fake().narrow(0, 0, 2).float()
        msg = str(exc.value)
        assert "'w'" in msg
        assert "narrow" in msg

    def test_unsupported_is_a_notimplementederror(self):
        """Callers distinguish "this backend can't handle the loader" from bugs."""
        assert issubclass(_UnsupportedFakeOp, NotImplementedError)


class TestBakeRecording:
    """During the dry run the fake's ``copy_`` is the data sink: it records the
    source chain plus the meta destination's strided region and moves nothing."""

    def _recorder_and_fake(self, shape=(4, 6)):
        rec = BakeSink()
        fake = FakeRDTTensor(
            name="w",
            shape=torch.Size(shape),
            dtype=torch.bfloat16,
            device=META,
            sink=rec,
        )
        return rec, fake

    def test_copy_records_the_destination_region(self):
        rec, fake = self._recorder_and_fake()
        layer = _FakeLayer("q_proj")
        param = torch.empty((8, 6), dtype=torch.bfloat16, device=META)
        dest = param.narrow(0, 4, 4)  # the second half of a fused param
        rec.current = (layer, "weight")
        dest.copy_(fake)

        (recorded,) = rec.copies_by_layer[layer]
        assert recorded.src == ("w", ())
        assert recorded.param_name == "weight"
        assert recorded.offset == dest.storage_offset() == 24
        assert recorded.shape == (4, 6)
        assert recorded.stride == (6, 1)

    def test_copy_marks_the_source_name_live(self):
        rec, fake = self._recorder_and_fake()
        rec.current = (_FakeLayer("l"), "weight")
        torch.empty((4, 6), dtype=torch.bfloat16, device=META).copy_(fake)
        assert rec.copied_names == {"w"}

    def test_unattributed_copy_is_live_but_unrecorded(self):
        """A copy_ with no loader stamp cannot be attributed to a param, so its
        module must fall back to the plain load — but the name still moved data."""
        rec, fake = self._recorder_and_fake()
        rec.current = None
        torch.empty((4, 6), dtype=torch.bfloat16, device=META).copy_(fake)
        assert rec.copied_names == {"w"}
        assert dict(rec.copies_by_layer) == {}

    def test_copies_are_grouped_by_module_in_call_order(self):
        rec, _ = self._recorder_and_fake()
        layer = _FakeLayer("gate_up")
        for i, name in enumerate(("gate", "up")):
            fake = FakeRDTTensor(
                name=name,
                shape=torch.Size((4,)),
                dtype=torch.bfloat16,
                device=META,
                sink=rec,
            )
            param = torch.empty((8,), dtype=torch.bfloat16, device=META)
            rec.current = (layer, "weight")
            param.narrow(0, 4 * i, 4).copy_(fake)
        assert [c.src[0] for c in rec.copies_by_layer[layer]] == ["gate", "up"]
        assert [c.offset for c in rec.copies_by_layer[layer]] == [0, 4]

    def test_recording_a_sliced_source(self):
        """The chain on the source and the region on the dest are independent."""
        rec, fake = self._recorder_and_fake(shape=(8, 6))
        layer = _FakeLayer("k_proj")
        rec.current = (layer, "weight")
        param = torch.empty((4, 6), dtype=torch.bfloat16, device=META)
        param.copy_(fake.narrow(0, 2, 4))
        (recorded,) = rec.copies_by_layer[layer]
        assert recorded.src == ("w", (("narrow", (0, 2, 4), ()),))
        assert recorded.shape == (4, 6)

    def test_the_recorded_dtype_is_the_chains_output_not_the_sources(self):
        """``view(dtype)`` is allowlisted, so a chain can reinterpret dtype. The
        producer packs what the replay yields, so the record must carry the
        POST-chain dtype: taking it from the source name's metadata instead sizes
        the slice with the wrong itemsize and shifts every later slice in the
        chunk, carving the packed blob differently on the two sides."""
        rec, fake = self._recorder_and_fake(shape=(4,))
        layer = _FakeLayer("reinterpreted")
        rec.current = (layer, "weight")
        viewed = fake.view(torch.float32)  # 4 x bf16 -> 2 x f32
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
        catch a mismatch — the consumer's buffer view is exactly prod(dest.shape)
        elements, so it reshapes cleanly over bytes laid out at other offsets.
        So the bake refuses rather than recording a slice it cannot carve."""
        rec, fake = self._recorder_and_fake(shape=(1, 6))
        rec.current = (_FakeLayer("broadcast"), "weight")
        param = torch.empty((4, 6), dtype=torch.bfloat16, device=META)
        with pytest.raises(_UnsupportedFakeOp, match="broadcasting"):
            param.copy_(fake)


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


class TestReplicaOverlay:
    """[RDT-SHARE-SLOTS] Several inference deployments carve the same blocks.

    The producer can serve R deployments out of ONE slot only if the R copies of
    a worker meet on the same producer asking for the same bytes. Identical
    plans give the second half for free; this is the first half. Carving over
    the whole fleet instead spreads the multi-owner (replicated) names of
    different deployments onto different producers, so a producer ends up
    holding a ring per distinct worker index and nothing can merge.
    """

    def _fleet(
        self,
        num_producers,
        workers,
        replicas,
        owner_sets=None,
        name_owner_class=None,
        n_names=6,
    ):
        names = [f"g{i}" for i in range(n_names)]
        return (
            RdtRouter(
                num_producers,
                workers * replicas,
                owner_sets,
                name_owner_class,
                names,
                [1] * n_names,
                workers_per_replica=workers,
            ),
            names,
        )

    def test_every_replica_of_a_worker_resolves_the_same_producer(self):
        r, names = self._fleet(16, 8, 4)
        for worker in range(8):
            for name in names:
                picks = {r.producer_for(worker + rep * 8, name) for rep in range(4)}
                assert len(picks) == 1, (
                    f"worker {worker} split across producers {picks}"
                )

    def test_adding_a_deployment_does_not_move_the_first_one(self):
        """The strongest form: an existing consumer's routes are byte-identical
        before and after another deployment is provisioned."""
        one, names = self._fleet(16, 8, 1)
        two, _ = self._fleet(16, 8, 2)
        for consumer_id in range(8):
            assert [one.producer_for(consumer_id, n) for n in names] == [
                two.producer_for(consumer_id, n) for n in names
            ]

    def test_one_deployment_is_the_historical_carve(self):
        """Default (no width) and width == the whole fleet must agree with the
        plain block rule, so a single-deployment fleet is untouched."""
        plain = _router(16, 8, group_lens=[1] * 6)
        explicit, names = self._fleet(16, 8, 1)
        for consumer_id in range(8):
            block = assign_producer_indices(16, 8, consumer_id)
            for name in names:
                assert plain.producer_for(consumer_id, name) == explicit.producer_for(
                    consumer_id, name
                )
                assert explicit.producer_for(consumer_id, name) in block

    def test_the_overlay_holds_for_partially_owned_names(self):
        """Pipeline stages and experts too: two owner sets, one narrow. The
        narrow one already forced agreement (one owner, no block to carve); the
        wide one is what the carve decides."""
        r, names = self._fleet(
            8,
            4,
            2,
            owner_sets=[[0, 1, 2, 3], [5]],
            name_owner_class=[0, 1, 0, 1, 0, 1],
        )
        for worker in range(4):
            for name in names:
                assert r.producer_for(worker, name) == r.producer_for(worker + 4, name)
        assert {r.producer_for(c, "g1") for c in range(8)} == {5}

    def test_a_width_that_does_not_divide_the_fleet_raises(self):
        """A uniform fleet is asserted upstream in ``get_world_size``; if it ever
        is not, the overlay would map two workers of one deployment onto the same
        block index, so refuse rather than serve the wrong bytes."""
        with pytest.raises(ValueError, match="does not divide"):
            RdtRouter(4, 6, None, None, ["a"], [1], workers_per_replica=4)

    def test_pulls_still_carry_the_global_consumer_id(self):
        """The carve uses the index WITHIN a deployment, but the wire must keep
        the fleet-global id: it is what lets the producer tell the sharers of a
        slot apart, and count their arrivals separately."""

        class _Method:
            def __init__(self):
                self.calls = []

            def remote(self, keys, consumer_id, seq):
                self.calls.append((keys, consumer_id, seq))

        r, _ = self._fleet(4, 2, 2)
        methods = [_Method() for _ in range(4)]
        r.bind([object()] * 4, methods, consumer_id=3)  # replica 1, worker 1
        r.pull(2, ["k"], 7)

        assert methods[2].calls == [(["k"], 3, 7)]


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
        """dtype rides the record from the bake, where it is the fake's dtype
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


class TestBufferAllocBytes:
    def test_rounds_up_to_a_coarse_256mb_boundary(self):
        assert buffer_alloc_bytes(1) == 256 << 20
        assert buffer_alloc_bytes(256 << 20) == 256 << 20
        assert buffer_alloc_bytes((256 << 20) + 1) == 512 << 20

    def test_presize_is_a_floor_not_a_cap(self):
        assert buffer_alloc_bytes(1, presize=3 << 30) == 3 << 30
        big = 4 << 30
        assert buffer_alloc_bytes(big, presize=1 << 30) == big

    def test_never_returns_less_than_requested(self):
        for nbytes in (1, 1 << 20, (700 << 20) + 3):
            assert buffer_alloc_bytes(nbytes) >= nbytes


class TestLayerwiseGroups:
    def test_partition_is_one_group_per_layer_between_unindexed_blocks(self):
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
        """Un-indexed names group by POSITION, not by name class — so one that
        looks like a pre-block name still lands after the layers it follows.
        Group index therefore means the same thing on every rank."""
        names = [
            "embed.weight",
            "model.layers.0.a",
            "embed.weight.tied",  # after the layers => its own trailing block
        ]
        groups = layerwise_groups(names)
        assert [n for g in groups for n in g] == names

    def test_layers_are_ordered_by_index_not_appearance(self):
        names = ["model.layers.10.a", "model.layers.2.a"]
        assert layerwise_groups(names) == [
            ["model.layers.2.a"],
            ["model.layers.10.a"],
        ]

    def test_stacks_keep_prefix_appearance_order_and_sort_within(self):
        """Two stacks stay in the order they appear — the vision tower before the
        text stack — while each sorts by index internally."""
        names = [
            "visual.blocks.1.w",
            "visual.blocks.0.w",
            "model.language_model.layers.1.w",
            "model.language_model.layers.0.w",
        ]
        assert layerwise_groups(names) == [
            ["visual.blocks.0.w"],
            ["visual.blocks.1.w"],
            ["model.language_model.layers.0.w"],
            ["model.language_model.layers.1.w"],
        ]

    def test_the_post_block_lands_last_however_early_it_arrives(self):
        """Megatron-Bridge streams the last pipeline stage's output block before
        its layers. Sweeping un-indexed names after the first layer into a
        trailing group is what lets the gather loop walk groups ascending."""
        names = ["model.layers.0.w", "model.norm.weight", "model.layers.1.w"]
        assert layerwise_groups(names) == [
            ["model.layers.0.w"],
            ["model.layers.1.w"],
            ["model.norm.weight"],
        ]

    def test_names_sharing_a_layer_coalesce_when_the_source_interleaves(self):
        """A raw-checkpoint source may yield in shard-packing order. Coalescing on
        the key keeps that from shattering one layer into many groups."""
        names = [
            "model.layers.0.a",
            "model.layers.1.a",
            "model.layers.0.b",
        ]
        assert layerwise_groups(names) == [
            ["model.layers.0.a", "model.layers.0.b"],
            ["model.layers.1.a"],
        ]

    def test_no_empty_groups(self):
        for names in ([], ["only.pre"], ["model.layers.0.a"]):
            assert all(g for g in layerwise_groups(names))

    def test_a_model_with_no_layers_is_one_pre_group(self):
        assert layerwise_groups(["a", "b"]) == [["a", "b"]]

    @pytest.mark.parametrize(
        "prefix",
        [
            "model.layers.",  # Llama, Qwen3, GLM-4.5, DeepSeek, Mixtral
            "model.language_model.layers.",  # Qwen3.5/3.6, Qwen2.5-VL, Gemma3
            "language_model.layers.",  # Kimi-K2.5, Nemotron-VL
            "thinker.model.layers.",  # Qwen2.5/3-Omni
            "transformer.h.",  # GPT-2, GPT-J, Bloom, Falcon
            "transformer.blocks.",  # MPT, DBRX
            "transformer.encoder.layers.",  # ChatGLM
            "backbone.layers.",  # Mamba, Mamba2
            "gpt_neox.layers.",  # GPT-NeoX
            "model.decoder.layers.",  # OPT
            "model.h.",  # Exaone
            "model.layers.layers.",  # Plamo3
            "bert.encoder.layer.",  # BERT, RoBERTa
            "visual.blocks.",  # Qwen-VL vision tower
            "vision_tower.vision_model.encoder.layers.",  # SigLIP/CLIP towers
            "audio_tower.layers.",  # Whisper-style audio towers
        ],
    )
    def test_every_supported_naming_convention_partitions_per_layer(self, prefix):
        """One decoder layer per group under any prefix, with no per-architecture
        table. A literal `model.layers.` match silently yields one whole-model
        group for most of these."""
        names = [f"{prefix}{i}.self_attn.q_proj.weight" for i in range(3)]
        assert layerwise_groups(names) == [[n] for n in names]

    def test_a_moe_layer_stays_whole(self):
        """The OUTERMOST index wins, so per-expert names group by layer. Splitting
        on the expert would cut FusedMoE — one leaf module — across groups."""
        names = [
            "model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.mlp.experts.1.w1.weight",
            "model.layers.1.mlp.experts.0.w1.weight",
        ]
        assert layerwise_groups(names) == [
            [
                "model.layers.0.mlp.experts.0.w1.weight",
                "model.layers.0.mlp.experts.1.w1.weight",
            ],
            ["model.layers.1.mlp.experts.0.w1.weight"],
        ]

    def test_a_vlm_partitions_the_vision_tower_and_the_text_stack(self):
        """Qwen-VL layout. Today's literal `model.layers.` match puts every one of
        these in ONE group — the whole model — because the text stack ships as
        `model.language_model.layers.`."""
        names = [
            "visual.patch_embed.proj.weight",
            "visual.blocks.0.attn.qkv.weight",
            "visual.blocks.1.attn.qkv.weight",
            "visual.merger.mlp.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ]
        assert layerwise_groups(names) == [
            ["visual.patch_embed.proj.weight"],
            ["visual.blocks.0.attn.qkv.weight"],
            ["visual.blocks.1.attn.qkv.weight"],
            ["model.language_model.layers.0.self_attn.q_proj.weight"],
            ["visual.merger.mlp.weight", "lm_head.weight"],
        ]

    def test_two_stacks_sharing_an_index_do_not_merge(self):
        """The key carries the prefix, so vision block 0 and text layer 0 are
        different groups even though both are index 0."""
        names = [
            "visual.blocks.0.attn.qkv.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ]
        assert layerwise_groups(names) == [[n] for n in names]


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
        fake = FakeRDTTensor(
            name="w",
            shape=torch.Size((4, 6)),
            dtype=torch.bfloat16,
            device=META,
            sink=None,
        )
        (op, _args, _kwargs) = fake.t()._key()[1][0]
        assert op in ALLOWED_OPS


# ---------------------------------------------------------------------------
# Bake against a real model (GPU)
# ---------------------------------------------------------------------------

TINY_QWEN = "Qwen/Qwen3-0.6B"

# One small model per attention/FFN shape the bake has to survive. The point is
# the checkpoint->module mapping, which differs structurally between them: GQA
# fuses Q/K/V of unequal head counts into one param, MLA replaces them with the
# compressed kv_a/kv_b pair, and MoE turns per-expert checkpoint entries into
# stacked expert params.
#
# Keep them small: `load_format="dummy"` skips the download but not the
# allocation, so every parameter is still built on the device at full size.
BAKE_MODELS = [
    pytest.param(TINY_QWEN, id="gqa-dense"),
    pytest.param("TitanML/tiny-mixtral", id="moe"),
    # MLA (kv_lora_rank set, so `ModelConfig.use_mla`) plus routed experts. Only
    # the STRUCTURE is under test -- `_bake_probe` synthesizes the checkpoint
    # from the safetensors headers and never reads a weight value -- so a random
    # 0.06B model does the job, as long as it has the MLA head shape the prefill
    # backends accept: (qk_nope=128, qk_rope=64, v=128).
    #
    # Not an fp8 checkpoint: `process_weights_after_loading` swaps the vLLM
    # parameter subclasses out on those, and the plain second `load_weights`
    # this test compares against then cannot run.
    pytest.param("hmellor/tiny-random-DeepseekV2ForCausalLM", id="mla-moe"),
]


class TestRequiresTheRayExecutor:
    """The engine's data plane is Ray's, so a non-Ray executor cannot work. The
    check is at construction because the alternative is an opaque failure during
    the first handshake, long after the misconfiguration."""

    def _construct(self, backend):
        cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(distributed_executor_backend=backend)
        )
        eng = object.__new__(ShardedRDTWeightTransferEngine)
        # Only the guard is under test, so stand in for the base __init__.
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                ShardedRDTWeightTransferEngine.__bases__[0],
                "__init__",
                lambda self, *a, **k: None,
            )
            ShardedRDTWeightTransferEngine.__init__(eng, None, cfg, META, None)
        return eng

    @pytest.mark.parametrize("backend", ["uni", "mp", "external_launcher"])
    def test_a_non_ray_executor_is_refused(self, backend):
        with pytest.raises(ValueError, match="requires distributed_executor"):
            self._construct(backend)

    def test_ray_is_accepted(self):
        self._construct("ray")

    def test_a_custom_executor_class_is_left_alone(self):
        """A `type[Executor]` override is deliberate and unjudgeable here, so it
        must not be rejected for merely not being the string "ray"."""

        class _CustomExecutor:
            pass

        self._construct(_CustomExecutor)


def _bake_probe(self, model_name):  # noqa: PLR0915  (runs in the vLLM worker)
    """Drive `_bake` against the worker's real model and check the three
    properties. Returns a small summary; every assertion fires here, in the
    worker, so a failure surfaces as the RPC's exception.

    Self-contained on purpose: this is pickled to the worker, so it imports what
    it needs rather than closing over this module's globals.
    """
    import torch
    from huggingface_hub import get_safetensors_metadata

    from vllm.distributed.weight_transfer.sharded_rdt_common import ALLOWED_OPS
    from vllm.distributed.weight_transfer.sharded_rdt_engine import (
        ShardedRDTWeightTransferEngine,
        ShardedRDTWeightTransferInitInfo,
    )
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

    model = self.model_runner.model
    device = next(model.parameters()).device

    # ---- the checkpoint, exactly as a trainer would describe and serve it ----
    # Only the safetensors HEADERS are fetched (a few KB); the values are
    # synthesized. Nothing here depends on real weights -- both the reference
    # load and the replay read this same dict, so the comparison is unaffected,
    # and it keeps a multi-GB download out of CI.
    st_dtype = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "I64": torch.int64,
        "I32": torch.int32,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    meta = get_safetensors_metadata(model_name)
    specs = {}
    for file_meta in meta.files_metadata.values():
        for name, t in file_meta.tensors.items():
            specs[name] = (st_dtype[t.dtype], tuple(t.shape))

    gen = torch.Generator(device="cpu").manual_seed(0)
    ckpt: dict[str, torch.Tensor] = {}
    for name, (dt, shape) in specs.items():
        if dt.is_floating_point:
            # Small magnitudes so any narrowing cast stays finite.
            v = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
            ckpt[name] = v.to(dtype=dt)
        else:
            ckpt[name] = torch.zeros(shape, dtype=dt)
    names = sorted(ckpt)
    dtype_names = [str(ckpt[n].dtype).split(".")[-1] for n in names]
    shapes = [list(ckpt[n].shape) for n in names]

    def _feed():
        return iter([(n, ckpt[n]) for n in names])

    # ---- reference: what the model's OWN loader produces from that checkpoint --
    # Poison first, so "which params did load_weights actually write?" is
    # answerable. Some params have no checkpoint source at all -- derived ones,
    # or ones process_weights_after_loading builds -- and those keep their NaNs
    # and are legitimately outside the plan. That set is what the plan must
    # cover, not every parameter of the model.
    def _poison():
        with torch.no_grad():
            for prm in model.parameters():
                if prm.is_floating_point():
                    prm.fill_(float("nan"))
                else:
                    prm.zero_()

    _poison()
    model.load_weights(_feed())
    loaded = {
        n
        for n, prm in model.named_parameters()
        if not (prm.is_floating_point() and torch.isnan(prm).any())
    }
    # On the host: the weights themselves already fill the GPU.
    reference = {
        n: prm.detach().to("cpu", copy=True)
        for n, prm in model.named_parameters()
        if n in loaded
    }

    # ---- pre-bake state, for the restoration check --------------------------
    param_name_of = {}
    for mod_name, mod in model.named_modules():
        for pname in get_layer_tensors(mod):
            param_name_of[(id(mod), pname)] = f"{mod_name}.{pname}".lstrip(".")
    loaders_before = {}
    for mod in model.modules():
        for pname, tensor in get_layer_tensors(mod).items():
            loaders_before[(id(mod), pname)] = getattr(tensor, "weight_loader", None)

    # ---- bake ---------------------------------------------------------------
    eng = object.__new__(ShardedRDTWeightTransferEngine)
    eng.model = model
    eng.device = device
    eng._name_to_plan = {}
    eng._name_meta = {}
    eng._live_names = set()
    eng._bake(
        ShardedRDTWeightTransferInitInfo(
            names=names, dtype_names=dtype_names, shapes=shapes
        )
    )

    # (1) the bake ran and produced a plan
    assert eng._name_to_plan, "bake produced no plan"
    assert eng._live_names, "bake recorded no live names"
    n_modules = len({id(v) for v in eng._name_to_plan.values()})

    # (2) every intermediate change to model state was undone
    for mod in model.modules():
        for pname, tensor in get_layer_tensors(mod).items():
            key = (id(mod), pname)
            loader = getattr(tensor, "weight_loader", None)
            assert not hasattr(loader, "_rdt_stamp_inner"), (
                f"recording stamp left on {param_name_of.get(key, key)}"
            )
            assert loader is loaders_before[key], (
                f"weight_loader not restored on {param_name_of.get(key, key)}"
            )
            assert not tensor.is_meta, (
                f"{param_name_of.get(key, key)} left on meta after the dry run"
            )
    for pname, p in model.named_parameters():
        assert torch.equal(p.detach().cpu(), reference[pname]), f"bake mutated {pname}"

    # (3) replaying the recorded op chains reproduces that same load
    scatters = {id(v): v for v in eng._name_to_plan.values()}

    # No deferred-attention layer may be a plan target. The engine's quant
    # thread calls `info.reset()` on every module it processes, which makes
    # finalize_layerwise_reload SKIP that module -- and finalize is the only
    # thing that calls the LAYER's process_weights_after_loading, which is where
    # MLA derives W_UK_T/W_UV from kv_b_proj. An attention layer in the plan
    # would therefore leave those derived weights stale after a sync, silently.
    from vllm.model_executor.layers.attention import is_deferred_attention_layer

    attn_in_plan = sorted(
        {
            type(c.layer).__name__
            for plan in scatters.values()
            for c in plan
            if is_deferred_attention_layer(c.layer)
        }
    )
    assert not attn_in_plan, (
        f"plan targets deferred-attention layer(s) {attn_in_plan}; the quant "
        f"thread would reset them and finalize would skip their "
        f"process_weights_after_loading, staling any derived weights"
    )

    touched = set()
    _poison()
    with torch.no_grad():
        for plan in scatters.values():
            for c in plan:
                src_name, chain = c.src
                t = ckpt[src_name]
                for op, args, kw in chain:
                    assert op in ALLOWED_OPS, f"disallowed op {op!r}"
                    t = getattr(t, op)(*args, **dict(kw))
                dest = getattr(c.layer, c.param_name)
                dest.as_strided(c.shape, c.stride, c.offset).copy_(t)
                touched.add(param_name_of[(id(c.layer), c.param_name)])

    assert loaded, "the reference load wrote nothing"
    # The plan owns everything with a checkpoint source. Anything else the model
    # ends up holding is DERIVED -- under MLA, `process_weights_after_loading`
    # splits kv_b_proj into W_UK_T/W_UV via `replace_parameter` -- and belongs to
    # finalize_layerwise_reload, which `finish_weight_update` runs after the
    # replay. So the uncovered set must contain only params owned by a module
    # that post-processes; a plain loadable weight going missing still fails.
    post_processed = {
        id(mod)
        for mod in model.modules()
        if hasattr(mod, "process_weights_after_loading")
    }
    modules_by_name = dict(model.named_modules())
    uncovered = sorted(loaded - touched)
    derived: list[str] = []
    unexplained: list[str] = []
    for n in uncovered:
        mod_name = n.rsplit(".", 1)[0]
        mod = modules_by_name.get(mod_name)
        is_derived = mod is not None and id(mod) in post_processed
        (derived if is_derived else unexplained).append(n)
    assert not unexplained, (
        f"the plan misses {len(unexplained)} param(s) load_weights wrote that "
        f"nothing derives: {unexplained[:5]}"
    )
    mismatched = [
        n
        for n, prm in model.named_parameters()
        if n in loaded and not torch.equal(prm.detach().cpu(), reference[n])
    ]
    assert not mismatched, f"replay differs from load_weights for: {mismatched[:5]}"

    return {
        "names": len(names),
        "baked": len(eng._name_to_plan),
        "live": len(eng._live_names),
        "modules": n_modules,
        "loaded_params": len(loaded),
        "derived_not_in_plan": len(derived),
        "replayed_params": len(touched),
        "total_params": sum(1 for _ in model.named_parameters()),
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the bake drives a real model on GPU"
)
class TestBakeOnARealModel:
    """`_bake` against a real vLLM model, which the CPU tests above cannot reach:
    they drive the planner with hand-built scatters, so nothing there exercises
    the dry run through a real `load_weights` — the part that monkeypatches the
    model and has to put it back.
    """

    @pytest.mark.parametrize("model", BAKE_MODELS)
    def test_bake_records_a_replayable_plan_and_restores_the_model(
        self, vllm_runner, monkeypatch, model
    ):
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(
            model, load_format="dummy", enforce_eager=True, max_model_len=1024
        ) as llm:
            (summary,) = llm.llm.collective_rpc(_bake_probe, args=(model,))
        print(f"bake summary [{model}]: {summary}")
        # Non-vacuity: the checks above are only worth something if the plan
        # covers the model. Fusion means far fewer leaf modules than names --
        # QKV and gate/up for GQA, the stacked experts for MoE.
        assert summary["baked"] >= 0.95 * summary["names"], summary
        assert summary["live"] >= 0.95 * summary["names"], summary
        assert 0 < summary["modules"] < summary["baked"], summary
        # The probe already asserts the plan covers every param load_weights
        # wrote; this just keeps that from being vacuously few.
        assert summary["loaded_params"] > 0.5 * summary["total_params"], summary
