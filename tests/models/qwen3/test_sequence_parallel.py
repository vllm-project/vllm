# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3Model's sequence-parallel (+ async-TP) wiring: the
``mark_sp_region`` boundaries set up in ``__init__`` and the shard/gather
done around ``forward``. Not an end-to-end correctness check of the
comm+GEMM kernels themselves -- see
``tests/models/common/ops/test_sequence_parallel.py`` for that.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.models.qwen3.nvidia import model as qwen3_model


class _StubLayer:
    """Stand-in decoder layer exposing just the four linears
    ``Qwen3Model.__init__`` wires SP regions around."""

    def __init__(self) -> None:
        self.self_attn = SimpleNamespace(qkv_proj=object(), o_proj=object())
        self.mlp = SimpleNamespace(gate_up_proj=object(), down_proj=object())


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
    parallel_config: ParallelConfig,
    *,
    num_layers: int = 2,
    pp_world_size: int = 1,
) -> tuple[qwen3_model.Qwen3Model, list[_StubLayer]]:
    """Build a ``Qwen3Model`` with ``Qwen2Model.__init__`` stubbed out to
    just install ``layers``/``start_layer``/``end_layer``, so the real SP
    wiring in ``Qwen3Model.__init__`` runs against stand-in layers instead
    of a full model."""
    layers = [_StubLayer() for _ in range(num_layers)]

    def fake_qwen2_init(self, *, vllm_config, prefix="", decoder_layer_type=None):
        nn.Module.__init__(self)
        self.layers = layers
        self.start_layer = 0
        self.end_layer = num_layers

    monkeypatch.setattr(qwen3_model.Qwen2Model, "__init__", fake_qwen2_init)
    monkeypatch.setattr(
        qwen3_model,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=pp_world_size, is_last_rank=True),
    )

    vllm_config = VllmConfig(parallel_config=parallel_config)
    model = qwen3_model.Qwen3Model(vllm_config=vllm_config)
    return model, layers


def test_wires_mark_sp_region_per_layer_boundary(monkeypatch: pytest.MonkeyPatch):
    """Each layer has two SP regions -- [prev down_proj -> qkv_proj] and
    [o_proj -> gate_up_proj] -- and the final region ends at the gathered
    norm (``exit_before=None``), not at another layer's input."""
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        qwen3_model,
        "mark_sp_region",
        lambda entry_after, exit_before, pc: calls.append((entry_after, exit_before)),
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=2, enable_sequence_parallel=True
    )
    _, layers = _make_model(monkeypatch, parallel_config, num_layers=2)

    expected = []
    prev_down_proj = None
    for layer in layers:
        expected.append((prev_down_proj, layer.self_attn.qkv_proj))
        expected.append((layer.self_attn.o_proj, layer.mlp.gate_up_proj))
        prev_down_proj = layer.mlp.down_proj
    expected.append((prev_down_proj, None))

    assert calls == expected


def test_skips_mark_sp_region_when_sequence_parallel_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(qwen3_model, "mark_sp_region", lambda *a, **k: calls.append(a))
    parallel_config = ParallelConfig(
        tensor_parallel_size=2, enable_sequence_parallel=False
    )
    _make_model(monkeypatch, parallel_config)

    assert calls == []


def test_rejects_sequence_parallel_with_pipeline_parallelism(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(qwen3_model, "mark_sp_region", lambda *a, **k: None)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2, enable_sequence_parallel=True
    )
    with pytest.raises(AssertionError, match="pipeline parallelism"):
        _make_model(monkeypatch, parallel_config, pp_world_size=2)


def test_forward_skips_shard_gather_when_sp_inactive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qwen3_model, "mark_sp_region", lambda *a, **k: None)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2, enable_sequence_parallel=True
    )
    model, _ = _make_model(monkeypatch, parallel_config)

    monkeypatch.setattr(qwen3_model, "is_sp_active", lambda pc: False)
    sentinel = object()
    base_forward = lambda self, *a, **k: sentinel  # noqa: E731
    monkeypatch.setattr(qwen3_model.Qwen2Model, "forward", base_forward)
    for name in ("sp_shard", "sp_all_gather"):

        def _fail(x, name=name):
            pytest.fail(f"{name} must not run while SP is inactive")

        monkeypatch.setattr(qwen3_model, name, _fail)

    out = model.forward(input_ids=torch.zeros(3, dtype=torch.long), positions=None)

    assert out is sentinel


def test_forward_shards_embeds_and_gathers_single_output_when_sp_active(
    monkeypatch: pytest.MonkeyPatch,
):
    parallel_config = ParallelConfig(
        tensor_parallel_size=2, enable_sequence_parallel=True
    )
    monkeypatch.setattr(qwen3_model, "mark_sp_region", lambda *a, **k: None)
    model, _ = _make_model(monkeypatch, parallel_config)
    monkeypatch.setattr(qwen3_model, "is_sp_active", lambda pc: True)

    embeds = torch.arange(12.0).view(4, 3)
    model.embed_input_ids = lambda input_ids: embeds

    shard_calls, gather_calls = [], []

    def fake_sp_shard(x):
        shard_calls.append(x)
        return x[:2]  # pretend tp_size=2 shards 4 -> 2

    monkeypatch.setattr(qwen3_model, "sp_shard", fake_sp_shard)

    def fake_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        return inputs_embeds * 10

    monkeypatch.setattr(qwen3_model.Qwen2Model, "forward", fake_forward)

    def fake_sp_all_gather(x):
        gather_calls.append(x)
        return torch.cat([x, x])  # un-shard back to 4

    monkeypatch.setattr(qwen3_model, "sp_all_gather", fake_sp_all_gather)

    positions = torch.arange(4)
    out = model.forward(input_ids=torch.ones(4, dtype=torch.long), positions=positions)

    assert len(shard_calls) == 1
    torch.testing.assert_close(shard_calls[0], embeds)
    assert len(gather_calls) == 1
    torch.testing.assert_close(gather_calls[0], embeds[:2] * 10)
    torch.testing.assert_close(out, torch.cat([embeds[:2] * 10, embeds[:2] * 10]))


def test_forward_gathers_every_aux_hidden_state_when_sp_active(
    monkeypatch: pytest.MonkeyPatch,
):
    parallel_config = ParallelConfig(
        tensor_parallel_size=2, enable_sequence_parallel=True
    )
    monkeypatch.setattr(qwen3_model, "mark_sp_region", lambda *a, **k: None)
    model, _ = _make_model(monkeypatch, parallel_config)
    monkeypatch.setattr(qwen3_model, "is_sp_active", lambda pc: True)
    monkeypatch.setattr(qwen3_model, "sp_shard", lambda x: x)

    hidden = torch.ones(3, 2)
    aux = [torch.full((3, 2), 2.0), torch.full((3, 2), 3.0)]
    monkeypatch.setattr(
        qwen3_model.Qwen2Model, "forward", lambda self, *a, **k: (hidden, aux)
    )
    monkeypatch.setattr(qwen3_model, "sp_all_gather", lambda x: x)

    positions = torch.arange(3)
    out_hidden, out_aux = model.forward(
        input_ids=None, positions=positions, inputs_embeds=torch.zeros(3, 2)
    )

    torch.testing.assert_close(out_hidden, hidden)
    assert [a.tolist() for a in out_aux] == [a.tolist() for a in aux]
