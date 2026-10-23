# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode segment follows vLLM's KV cache re-initialization."""

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context
from vllm.models.glm53.nvidia import decode as glm53_decode
from vllm.models.glm53.nvidia import model as glm53_model


class _Segment:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _decode(geometry) -> glm53_decode.SpecializedDecode:
    return glm53_decode.SpecializedDecode(
        SimpleNamespace(), SimpleNamespace(geometry=geometry), claim=4
    )


def test_close_drops_the_kv_bound_state_and_keeps_the_adopted_buffers(
    monkeypatch, geometry
):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    decode = _decode(geometry)
    segment, embedding = _Segment(), object()
    decode._segment, decode._embedding = segment, embedding
    decode._kv = object()
    decode._tensors = {"layers.0.attention.kv_cache": torch.zeros(1)}
    decode._table, decode._padded = torch.zeros(2), torch.zeros(4)
    decode.captures, decode._token_ids, decode._positions = 2, 1, 2
    decode.close()
    assert segment.closed == 1 and not decode.built
    assert decode._kv is None and decode._tensors == {}
    assert decode.captures == 0 and decode._token_ids == 0 and decode._positions is None
    assert decode._table is None and decode._padded is None
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
    assert not decode._binds_live_cache()
    decode._tensors = {"layers.0.attention.kv_cache": built}
    assert decode._binds_live_cache()
    layer.kv_cache = [torch.zeros(4, 64, 576, dtype=torch.uint8)]
    assert not decode._binds_live_cache()


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
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    tensors = {"x": torch.zeros(1, device="meta")}
    first = object.__new__(arena_cls)
    segment = glm53_decode.Glm53Segment(geometry, 0, tensors, None, embedding=first)
    assert segment.embedding_arena is first and adopted == []
    weight = object()
    glm53_decode.Glm53Segment(geometry, 0, tensors, None, embedding=weight)
    assert adopted == [weight]


def test_request_blocks_stay_on_the_device(monkeypatch, geometry):
    decode = _decode(geometry)
    decode._table = torch.tensor([7, 3, 0], dtype=torch.int32)
    monkeypatch.setattr(
        torch.Tensor, "tolist", lambda self: pytest.fail("host copy of the block table")
    )
    assert decode._request_blocks("r") is decode._table
