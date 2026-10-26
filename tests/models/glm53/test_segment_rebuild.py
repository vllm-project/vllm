# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode segment follows vLLM's KV cache re-initialization."""

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context
from vllm.models.glm53.nvidia import decode as glm53_decode
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
