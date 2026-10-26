# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode segment follows vLLM's KV cache re-initialization."""

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context
from vllm.models.glm53.nvidia import decode as glm53_decode
from vllm.models.glm53.nvidia import draft as glm53_draft
from vllm.models.glm53.nvidia import kv as glm53_kv
from vllm.models.glm53.nvidia import model as glm53_model
from vllm.models.glm53.nvidia import switch


class _Segment:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _decode(geometry) -> glm53_decode.SpecializedDecode:
    return glm53_decode.SpecializedDecode(
        SimpleNamespace(), SimpleNamespace(geometry=geometry)
    )


def _tier(segment=None, table=None, padded=None, tensors=None, wide=False):
    return glm53_decode.TierState(
        segment,
        tensors or {},
        table,
        padded,
        token_ids=1,
        positions=2,
        captures=2,
        wide=wide,
    )


def test_close_drops_the_kv_bound_state_and_keeps_the_adopted_buffers(
    monkeypatch, geometry
):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    decode = _decode(geometry)
    segment, embedding = _Segment(), object()
    decode._embedding = embedding
    decode._kv = object()
    decode._tiers = {
        switch.Build(4): _tier(
            segment, torch.zeros(2), torch.zeros(4), {"x": torch.zeros(1)}
        )
    }
    assert decode.built and decode.captures == 2
    decode.close()
    assert segment.closed == 1 and not decode.built
    assert decode._kv is None and decode.captures == 0
    assert decode._embedding is embedding
    decode.close()
    assert segment.closed == 1


@pytest.mark.parametrize("installed", [True, False])
def test_model_state_drops_the_segment_at_every_kv_initialization(installed):
    closes: list[int] = []
    segment = SimpleNamespace(close=lambda: closes.append(1)) if installed else None
    state = object.__new__(glm53_model.Glm53ModelState)
    state.model = SimpleNamespace(model=SimpleNamespace(glm53_segment=segment))
    state.initialize_kv_cache(None, None)  # the profiling cache
    state.initialize_kv_cache(None, None)  # the real cache
    assert len(closes) == (2 if installed else 0)


def test_capture_check_compares_the_bound_layer0_cache(monkeypatch, geometry):
    built = torch.zeros(2, 64, 576, dtype=torch.uint8)
    layer = SimpleNamespace(kv_cache=[built])
    context = SimpleNamespace(
        no_compile_layers={"model.layers.0.self_attn.attn": layer}
    )
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: context)
    decode = _decode(geometry)
    tier = _tier()
    assert not decode._binds_live_cache(tier)
    tier.tensors = {"layers.0.attention.kv_cache": built}
    assert decode._binds_live_cache(tier)
    layer.kv_cache = [torch.zeros(4, 64, 576, dtype=torch.uint8)]
    assert not decode._binds_live_cache(tier)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_rebuild_reuses_the_adopted_embedding_arena(monkeypatch, geometry):
    from vllm.models.glm53.nvidia import arena as glm53_arena

    adopted: list[object] = []
    monkeypatch.setattr(
        glm53_decode,
        "adopt_embedding",
        lambda weight, geometry, group: adopted.append(weight),
    )
    monkeypatch.setattr(glm53_decode.providers, "program", lambda *args: [])
    monkeypatch.setattr(glm53_arena, "Glm53Placement", lambda *args: SimpleNamespace())
    arena_cls = type(
        "Arena", (glm53_arena.PeerArena,), {"__init__": lambda *args: None}
    )
    monkeypatch.setattr(glm53_arena, "PeerArena", arena_cls)
    monkeypatch.setattr(glm53_decode.glm53_launch, "bind", lambda launches, device: [])
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    tensors = {"x": torch.zeros(1, device="meta")}
    first: glm53_arena.PeerArena = object.__new__(arena_cls)
    segment = glm53_decode.Glm53Segment(geometry, 0, tensors, None, embedding=first)
    assert segment.embedding_arena is first and adopted == []
    weight = object()
    glm53_decode.Glm53Segment(geometry, 0, tensors, None, embedding=weight)
    assert adopted == [weight]


def test_request_blocks_stay_on_the_device(monkeypatch, geometry):
    decode = _decode(geometry)
    table = torch.tensor([7, 3, 0], dtype=torch.int32)
    decode._tiers = {switch.Build(4): _tier(table=table)}
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda self: pytest.fail("host copy of the block table")
    )
    assert decode._request_blocks("r") is table


def test_row_tier_requests_read_their_own_slot(geometry):
    rows = torch.arange(12, dtype=torch.int32).view(3, 4)
    tables = SimpleNamespace(block_tables=[SimpleNamespace(gpu=rows)])
    decode = glm53_decode.SpecializedDecode(
        SimpleNamespace(),
        SimpleNamespace(geometry=geometry),
        {8: 8},
        rows_per_request=4,
        row_tiers=True,
    )
    decode._block_tables = tables
    decode.request_added("a", 5, slot=2)
    decode.request_added("b", 7, slot=0)
    assert decode._request_blocks("a").tolist() == rows[2].tolist()
    assert decode._request_blocks("b").tolist() == rows[0].tolist()
    decode.request_removed("a")
    assert "a" not in decode._slots


def test_target_close_closes_the_installed_drafter(monkeypatch, geometry):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    drafter = _Segment()
    monkeypatch.setattr(glm53_draft, "_CURRENT", drafter)
    _decode(geometry).close()
    assert drafter.closed == 1


def _draft_tier(requests, segment):
    return glm53_draft.DraftTier(
        requests,
        segment,
        {"token_ids": torch.zeros(4 * requests)},
        torch.zeros(2),
        1,
        torch.zeros(1),
        torch.zeros(1),
        None,
        {"prefill": 1, "decode": 2},
    )


def test_drafter_close_forgets_its_captures_and_bound_tensors(monkeypatch, geometry):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2, 4)
    )
    assert draft.sizes == (1, 2, 4)
    segments = [_Segment(), _Segment()]
    draft._tiers = {1: _draft_tier(1, segments[0]), 4: _draft_tier(4, segments[1])}
    draft._positions = torch.zeros(4)
    draft._sequence_captures, draft._decoding, draft._warmed = 2, True, True
    draft._batch, draft._tokens = ("a",), torch.zeros(1)
    assert draft.claims_draft_prefill(3) and draft.claims_draft_prefill(1)
    draft.close()
    assert [segment.closed for segment in segments] == [1, 1] and not draft.built
    assert not draft.claims_draft_prefill(1) and draft._sequence_captures == 0
    assert not draft._decoding and not draft._warmed and draft._tokens is None
    assert draft._batch == () and draft._positions is None
    draft.decode_phase(True)  # a new capture sequence starts at proposal 0
    draft.decode_phase(False)
    assert not draft.claims_proposals()


def test_the_draft_decode_phase_ends_after_whole_graphs_of_proposals(geometry):
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2, 4)
    )
    draft.decode_phase(True)
    # Three sizes captured one after the other: two proposals each.
    draft._sequence_captures = 6
    draft.decode_phase(False)
    assert draft._sequence_captures == 0
    draft.decode_phase(True)
    draft._sequence_captures = 3  # a graph that holds one proposal of two
    with pytest.raises(RuntimeError, match="holds one proposal"):
        draft.decode_phase(False)


def test_a_batch_runs_the_smallest_drafter_tier_that_holds_it(geometry):
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2, 4)
    )
    assert draft.sizes == (1, 2, 4)
    assert [draft.tier_of(b) for b in (1, 2, 3, 4, 5)] == [1, 2, 4, 4, None]
    # No drafter tier above the deployment's batch.
    small = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2)
    )
    assert small.sizes == (1, 2) and small.tier_of(3) is None


def test_the_drafter_steps_the_tier_of_the_batch(monkeypatch, geometry):
    steps: list[tuple[int, int]] = []
    resets: list[int] = []

    class _Tiered(_Segment):
        def __init__(self, requests):
            super().__init__()
            self.requests = requests

        def step(self, live):
            steps.append((self.requests, live))

        def reset_seeds(self):
            resets.append(self.requests)

    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2, 4)
    )
    draft._tiers = {b: _draft_tier(b, _Tiered(b)) for b in (1, 2, 4)}
    start = torch.zeros(4, dtype=torch.int32)
    draft.step(["a", "b", "c"], start)
    draft.step(["a", "b", "c"], start)
    draft.step(["a", "b"], start)
    assert steps == [(4, 3), (4, 3), (2, 2)]
    assert resets == [4, 2]  # seeds follow the batch: reset when it changes


def test_a_step_past_the_largest_tier_runs_vllms_drafter(monkeypatch, geometry):
    resets: list[int] = []

    class _Tiered(_Segment):
        def __init__(self, requests):
            super().__init__()
            self.requests = requests

        def step(self, live):
            pass

        def reset_seeds(self):
            resets.append(self.requests)

    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2, 4)
    )
    draft._tiers = {b: _draft_tier(b, _Tiered(b)) for b in (1, 2, 4)}
    start = torch.zeros(9, dtype=torch.int32)
    draft.step(["a", "b", "c", "d"], start)
    draft.step(list("abcdef"), start)  # vLLM's drafter
    assert not draft.claims_proposals() and not draft.claims_draft_prefill(6)
    draft.step(["a", "b", "c", "d"], start)  # same four requests: fresh seeds
    assert resets == [4, 4]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for rows, lean in ((1, True), (2, True), (4, True), (8, False), (3, False)):
        draft._eager_rows = rows
        assert draft.claims_capture() == lean
    draft.decode_phase(True)
    draft._eager_rows = 4
    assert draft.claims_capture()
    with pytest.raises(RuntimeError, match="claimed lean"):
        draft.forward(SimpleNamespace(is_cuda=True, shape=(8,)), None, None)


def test_a_replayed_step_past_the_drafter_tiers_completes_its_index_keys(
    monkeypatch, geometry
):
    completed: list[int] = []

    class _Kv:
        pending = True

        def complete_pending(self, blocks_of, group):
            completed.append(len(completed))

    import vllm.distributed as distributed

    monkeypatch.setattr(distributed, "get_tp_group", lambda: None, raising=False)
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=3, tiers=(1, 2, 4)
    )
    draft._tiers = {b: _draft_tier(b, _Segment()) for b in (1, 2, 4)}
    monkeypatch.setattr(glm53_draft, "_CURRENT", draft)
    decode = _decode(geometry)
    decode._kv = _Kv()
    decode._complete_for_stock_drafter(5)  # no CuTe step yet: nothing pending of ours
    decode._complete_for_stock_drafter(4)
    decode._complete_for_stock_drafter(3)
    assert completed == []
    decode._complete_for_stock_drafter(5)  # first vLLM-drafter step after CuTe steps
    decode._complete_for_stock_drafter(6)
    assert completed == [0]
    decode._complete_for_stock_drafter(2)
    decode._complete_for_stock_drafter(8)
    assert completed == [0, 1]


def test_drafter_segment_close_gives_its_placement_back():
    events: list[str] = []
    segment = object.__new__(glm53_draft.Glm53DraftSegment)
    segment.arena = SimpleNamespace(close=lambda: events.append("arena"))
    segment.placement = SimpleNamespace(release=lambda: events.append("placement"))
    segment.close()
    assert events == ["arena", "placement"]


def test_a_capture_size_between_two_tiers_is_served_by_the_next_tier_up(geometry):
    def build(claims):
        return glm53_decode.SpecializedDecode(
            SimpleNamespace(),
            SimpleNamespace(geometry=geometry),
            claims,
            rows_per_request=4,
            row_tiers=True,
        )

    assert build({16: 16, 12: 16}).claims == {
        16: switch.Build(16),
        12: switch.Build(16),
    }
    for claims in ({16: 16, 10: 16}, {16: 16, 20: 16}, {16: 12}):
        with pytest.raises(ValueError, match="cannot serve"):
            build(claims)


def test_the_drafter_shares_the_tensors_of_the_target_tier_of_its_rows(geometry):
    decode = _decode(geometry)
    names = ("rope_table", glm53_decode.glm53_launch.BLOCK_TABLE)
    tensors = {name: torch.zeros(i + 1) for i, name in enumerate(names)}
    decode._tiers = {switch.Build(8): _tier(tensors=tensors)}
    assert decode.has_tier(8) and not decode.has_tier(16)
    assert decode.shared_tensors(8) == tensors and decode.shared_tensors(16) == {}
    assert decode.shared_tensors() == tensors  # the only tier


def test_a_drafter_tier_shares_the_tensors_of_its_own_build(geometry):
    decode = _decode(geometry)
    names = ("rope_table", glm53_decode.glm53_launch.BLOCK_TABLE)
    q4, q8 = switch.Build(8, 4), switch.Build(8, 8)
    two, four = ({name: torch.zeros(n) for name in names} for n in (1, 3))
    decode._tiers = {q4: _tier(tensors=two), q8: _tier(tensors=four)}
    assert decode.shared_tensors(q8) == four and decode.shared_tensors(q4) == two
    assert decode.shared_tensors(8) == two  # by row tier: the q4 build
    assert not decode.has_tier(switch.Build(8)) and decode.has_tier(q8)
    # MTP 1: two requests run the q4 MoE build of tier 8, three and four the q8 one.
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=1, tiers=(1, 2, 4)
    )
    assert [draft.target_rows(b) for b in (1, 2, 4)] == [switch.Build(4), q4, q8]
    assert not any(draft._foreign_target(b) for b in (1, 2, 4))


@pytest.mark.parametrize(
    ("drafts", "windows"), [(1, (1,)), (2, (1,)), (3, ()), (4, (0, 1)), (7, (0, 1))]
)
def test_the_catchup_stages_one_padded_window_below_mtp_3(geometry, drafts, windows):
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=drafts, tiers=(1,)
    )
    assert draft.windows == windows
    assert draft._vllm_name("token_ids") == ("vllm." * bool(windows)) + "token_ids"
    draft.decode_phase(True)
    draft.decode_phase(False)  # K = 1 captures no proposal


def test_a_step_owns_its_attention_metadata_when_its_drafter_tier_replays(
    monkeypatch, geometry
):
    asked: list[int] = []

    class _Drafter:
        def claims_draft_prefill(self, requests):
            asked.append(requests)
            return requests in (3, 4)

    monkeypatch.setattr(glm53_draft, "_CURRENT", _Drafter())
    decode = glm53_decode.SpecializedDecode(
        SimpleNamespace(),
        SimpleNamespace(geometry=geometry),
        {16: 16, 12: 16},
        rows_per_request=4,
        row_tiers=True,
    )
    decode._tiers = {switch.Build(16): _tier()}
    assert decode.owns_attn_metadata(16) and decode.owns_attn_metadata(12)
    assert not decode.owns_attn_metadata(8)  # not a claimed size
    assert asked == [4, 3]


def test_a_wide_tier_completes_the_rows_of_narrow_steps_first(monkeypatch, geometry):
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: "group")
    calls: list[list[str]] = []
    kv = SimpleNamespace(pending=glm53_kv.PendingRows(), narrow=glm53_kv.PendingRows())

    def complete_narrow(request_ids, block_ids_of, group):
        calls.append(list(request_ids))
        assert group == "group"
        for request_id in request_ids:
            kv.narrow.take(request_id)

    kv.complete_narrow = complete_narrow
    decode = glm53_decode.SpecializedDecode(
        SimpleNamespace(),
        SimpleNamespace(geometry=geometry),
        {8: 8, 16: 16},
        rows_per_request=4,
        row_tiers=True,
    )
    segment = SimpleNamespace(
        row_table=SimpleNamespace(set_uniform_step=lambda *args: None),
        reset_seeds=lambda: None,
        geometry=SimpleNamespace(max_requests=2, rows=8),
    )
    decode._kv = kv
    decode._tiers = {
        switch.Build(8): _tier(segment),
        switch.Build(16): _tier(segment, wide=True),
    }
    for request_id in ("a", "b"):
        decode.request_added(request_id, 100)
    decode.step(["a", "b"], [100, 100], [4, 4], 8)  # the narrow tier
    assert kv.narrow.has("a") and kv.pending.has("b") and not calls
    decode.step(["a", "b"], [104, 104], [4, 4], 8)
    decode.step(["a", "b", "c", "d"], [108, 108, 50, 50], [4, 4, 4, 4], 16)  # wide
    assert calls == [["a", "b", "c", "d"]]
    assert not kv.narrow.has("a") and kv.pending.has("a")
    decode.step(["a", "b", "c", "d"], [112, 112, 54, 54], [4, 4, 4, 4], 16)
    assert not kv.narrow  # wide steps record nothing narrow


def test_an_eager_forward_of_a_claimed_uniform_step_counts_as_stock(
    monkeypatch, geometry
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    decode = glm53_decode.SpecializedDecode(
        SimpleNamespace(), SimpleNamespace(geometry=geometry), {4: 4}
    )
    decode._tiers = {switch.Build(4): _tier(SimpleNamespace(reset_seeds=lambda: None))}
    decode.request_added("a", 100)
    input_ids = SimpleNamespace(is_cuda=True, shape=(4,))
    decode.step(["a"], [100], [4], 4)
    assert decode.forward(input_ids, None) is None
    assert (decode.uniform_steps, decode.unclaimed_steps) == (1, 1)
    assert decode.forward(input_ids, None) is None  # a dummy run: no step
    assert decode.unclaimed_steps == 1


def test_at_mtp_1_the_sampled_rows_come_from_the_sampled_token_counts(geometry):
    draft = glm53_draft.Glm53Draft(
        SimpleNamespace(), geometry, {}, drafts=1, tiers=(1, 2, 4)
    )
    assert draft._sampled is None  # no CUDA forms: nothing to write
    draft._first_rows = torch.arange(4) * 2 - 1
    draft._sampled = torch.zeros(4, dtype=torch.int64)
    draft.observe_sampled(torch.tensor([1, 2, 2], dtype=torch.int32))
    assert draft._sampled.tolist() == [0, 3, 5, 0]  # rows 2b + accepted
    draft.observe_sampled(0)  # a pipeline-parallel step without counts
    assert draft._sampled.tolist() == [0, 3, 5, 0]
