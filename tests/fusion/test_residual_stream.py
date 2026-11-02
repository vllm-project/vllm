# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the model-level ``ResidualStream`` state machine.

One ``ResidualStream`` is created per model forward and threaded through every
decoder layer, so correctness hinges on its pending/record/prepare transitions.
These tests pin those transitions plus the distribution-state gating
(``Scatter.FULL`` vs ``Scatter.PARTIAL``) without a GPU or a model load.

The TP collective is the one thing mocked here: ``tensor_model_parallel_all_reduce``
is replaced by an identity-times-world-size function. This is deliberate: a real
single-rank all-reduce is a no-op, so a bug that reduces twice would be
invisible. With the mock, one call gives the correct 2x and a second call
would give 4x, so double-reduce bugs fail the equality assertions loudly.
The real multi-rank path is covered by ``test_qwen3_deferred_allreduce_e2e.py``.
"""

import contextlib
import os
import tempfile

import pytest
import torch
import torch.nn as nn

import vllm.model_executor.layers.fusion.ar_rms_quant as arq
import vllm.model_executor.layers.fusion.residual_stream as rs_mod
from vllm.config import LoRAConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fusion.residual_stream import (
    ResidualStream,
    Scatter,
    finalize_norm,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.models.interfaces import EagleModelMixin

from ..utils import ensure_current_vllm_config

HIDDEN = 8


class _ReduceCounter:
    """Counting stand-in for the TP all-reduce collective."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x * self.world_size


@pytest.fixture(autouse=True)
def _single_rank():
    """Real single-rank parallel state, over gloo.

    The stream reads the TP world size at construction time and the row linears
    need a group for weight creation, but these tests are pure state-machine
    tests: gloo (instead of the platform's NCCL backend) keeps them runnable
    with no GPU.
    """
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            yield
    finally:
        cleanup_dist_env_and_memory()
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def _config() -> VllmConfig:
    return VllmConfig()


def _row_linear(reduce_results: bool) -> RowParallelLinear:
    return RowParallelLinear(
        HIDDEN,
        HIDDEN,
        bias=False,
        reduce_results=reduce_results,
        params_dtype=torch.float32,
    )


@pytest.fixture
def tp2(monkeypatch):
    """TP=2 with a counting explicit all-reduce (fused kernel disabled)."""
    counter = _ReduceCounter(2)
    for mod in (rs_mod, arq):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(mod, "tensor_model_parallel_all_reduce", counter)
    monkeypatch.setattr(arq, "flashinfer_trtllm_fused_allreduce_norm", None)
    return counter


def test_layer_entry_is_full_and_installs_residual(tp2):
    stream = ResidualStream(vllm_config=_config())
    norm = RMSNorm(HIDDEN)
    hidden = torch.randn(4, HIDDEN)

    out, residual = stream.prepare(hidden, norm, None)

    assert torch.equal(out, norm(hidden))
    assert residual is hidden
    # The embedding input is already reduced: nothing is owed at layer 0.
    assert tp2.calls == 0


def test_deferred_boundary_reduces_exactly_once(tp2):
    norm = RMSNorm(HIDDEN)
    residual = torch.randn(4, HIDDEN)
    stream = ResidualStream(residual=residual, vllm_config=_config())
    attn_out = torch.randn(4, HIDDEN)
    expected_residual = residual + 2 * attn_out

    stream.record(attn_out, Scatter.PARTIAL)
    out, new_residual = stream.prepare(attn_out, norm, None)

    assert tp2.calls == 1
    assert torch.equal(new_residual, expected_residual)
    assert torch.equal(out, norm(expected_residual))


def test_full_boundary_does_not_reduce(tp2):
    norm = RMSNorm(HIDDEN)
    residual = torch.randn(4, HIDDEN)
    stream = ResidualStream(residual=residual, vllm_config=_config())
    attn_out = torch.randn(4, HIDDEN)
    expected_residual = residual + attn_out

    stream.record(attn_out, Scatter.FULL)
    out, new_residual = stream.prepare(attn_out, norm, None)

    assert tp2.calls == 0
    assert torch.equal(new_residual, expected_residual)
    assert torch.equal(out, norm(expected_residual))


@pytest.mark.parametrize(
    "reduce_results,expected",
    [
        (False, True),  # row linear skips its reduce -> a PARTIAL sum is owed
        (True, False),  # row linear reduced itself -> FULL
    ],
)
def test_defer_reads_reduce_results(reduce_results, expected):
    stream = ResidualStream(vllm_config=_config())
    linear = _row_linear(reduce_results)

    assert stream.defer_attn(linear) is expected
    assert stream.defer_mlp(linear) is expected


def test_defer_defaults_to_full_for_missing_flag():
    stream = ResidualStream(vllm_config=_config())
    assert stream.defer_attn(nn.Identity()) is False
    assert stream.defer_mlp(nn.Identity()) is False


def test_record_twice_raises():
    stream = ResidualStream(vllm_config=_config())
    stream.record(torch.randn(2, HIDDEN), Scatter.PARTIAL)

    with pytest.raises(RuntimeError, match="unconsumed"):
        stream.record(torch.randn(2, HIDDEN), Scatter.PARTIAL)


def test_recorded_tensor_must_be_the_forwarded_one(tp2):
    stream = ResidualStream(vllm_config=_config())
    recorded = torch.randn(2, HIDDEN)
    stream.record(recorded, Scatter.PARTIAL)

    with pytest.raises(RuntimeError, match="does not belong"):
        stream.input(torch.randn(2, HIDDEN))


def test_snapshot_does_not_consume_or_double_reduce(tp2):
    """Aux-hidden-state capture must materialize the sum without consuming it.

    The pending PARTIAL value still owes its single all-reduce to the real
    consumer; snapshotting must not reduce it in place, which would make the
    later boundary reduce an already-reduced value and scale it by tp_size.
    """
    norm = RMSNorm(HIDDEN)
    residual = torch.randn(4, HIDDEN)
    attn_out = torch.randn(4, HIDDEN)
    stream = ResidualStream(residual=residual, vllm_config=_config())
    expected_sum = residual + 2 * attn_out

    stream.record(attn_out, Scatter.PARTIAL)
    snapshot = stream.snapshot(attn_out)

    assert torch.equal(snapshot, expected_sum)
    assert tp2.calls == 1

    out, new_residual = stream.prepare(attn_out, norm, None)

    assert tp2.calls == 2
    assert torch.equal(new_residual, expected_sum)
    assert torch.equal(out, norm(expected_sum))


def test_snapshot_at_model_entry_returns_hidden(tp2):
    stream = ResidualStream(vllm_config=_config())
    hidden = torch.randn(2, HIDDEN)

    assert stream.snapshot(hidden) is hidden
    assert tp2.calls == 0


def test_from_handoff_consumes_pending_without_reducing(tp2):
    norm = RMSNorm(HIDDEN)
    hidden = torch.randn(4, HIDDEN)
    residual = torch.randn(4, HIDDEN)
    expected_residual = residual + hidden
    stream = ResidualStream.from_handoff(hidden, residual, Scatter.FULL, _config())

    _, new_residual = stream.prepare(hidden, norm, None)

    assert tp2.calls == 0
    assert torch.equal(new_residual, expected_residual)
    assert stream.pending is None


def test_non_rmsnorm_boundary_reduces_then_normalizes(tp2):
    norm = GemmaRMSNorm(HIDDEN)
    residual = torch.randn(4, HIDDEN)
    attn_out = torch.randn(4, HIDDEN)
    stream = ResidualStream(residual=residual, vllm_config=_config())
    expected_residual = residual + 2 * attn_out

    stream.record(attn_out, Scatter.PARTIAL)
    out, new_residual = stream.prepare(attn_out, norm, None)

    assert tp2.calls == 1
    assert torch.equal(new_residual, expected_residual)
    assert torch.equal(out, norm(expected_residual))


@pytest.fixture(scope="module")
def lora_vllm_config():
    return VllmConfig(
        model_config=ModelConfig(max_model_len=16),
        lora_config=LoRAConfig(),
    )


@pytest.mark.parametrize("lora", [False, True])
def test_lora_disables_activation_quant(monkeypatch, lora, lora_vllm_config):
    """LoRA wrapping replaces the consumer kernel, so the quant handoff is off."""
    seen = []

    def fake_fused(hidden, residual, norm, consumer_linear, *, do_allreduce):
        seen.append(consumer_linear)
        return norm(hidden), hidden

    monkeypatch.setattr(rs_mod, "fused_ar_rms_norm_quant", fake_fused)
    stream = ResidualStream(vllm_config=lora_vllm_config if lora else _config())
    consumer = _row_linear(reduce_results=True)

    stream.prepare(torch.randn(4, HIDDEN), RMSNorm(HIDDEN), consumer)

    assert seen == [None if lora else consumer]


@pytest.mark.parametrize(
    "world_size,incoming,expected_calls",
    [
        (2, Scatter.PARTIAL, 1),
        (2, Scatter.FULL, 0),
        (1, Scatter.PARTIAL, 0),  # a single rank never owes an all-reduce
    ],
)
def test_finalize_norm_reduces_only_when_owed(
    monkeypatch, world_size, incoming, expected_calls
):
    counter = _ReduceCounter(world_size)
    for mod in (rs_mod, arq):
        monkeypatch.setattr(
            mod, "get_tensor_model_parallel_world_size", lambda: world_size
        )
        monkeypatch.setattr(mod, "tensor_model_parallel_all_reduce", counter)
    monkeypatch.setattr(arq, "flashinfer_trtllm_fused_allreduce_norm", None)

    norm = RMSNorm(HIDDEN)
    hidden = torch.randn(4, HIDDEN)
    residual = torch.randn(4, HIDDEN)
    # Only a fired all-reduce sums the tp_size replicas.
    expected = residual + (world_size if expected_calls else 1) * hidden

    out = finalize_norm(norm, hidden, residual, incoming=incoming)

    assert counter.calls == expected_calls
    assert torch.equal(out, norm(expected))


def test_finalize_norm_handles_non_rmsnorm(tp2):
    norm = GemmaRMSNorm(HIDDEN)
    hidden = torch.randn(4, HIDDEN)
    residual = torch.randn(4, HIDDEN)
    expected = residual + 2 * hidden

    out = finalize_norm(norm, hidden, residual, incoming=Scatter.PARTIAL)

    assert tp2.calls == 1
    assert torch.equal(out, norm(expected))


class _EagleModel(EagleModelMixin):
    start_layer = 0
    aux_hidden_state_layers = (0,)


def test_eagle_aux_capture_uses_snapshot(tp2):
    """Eagle3 aux capture must record the full sum, not the partial one."""
    norm = RMSNorm(HIDDEN)
    residual = torch.randn(4, HIDDEN)
    attn_out = torch.randn(4, HIDDEN)
    stream = ResidualStream(residual=residual, vllm_config=_config())
    expected_sum = residual + 2 * attn_out
    stream.record(attn_out, Scatter.PARTIAL)

    aux: list[torch.Tensor] = []
    _EagleModel()._maybe_add_hidden_state(aux, 0, attn_out, residual, stream)

    assert len(aux) == 1
    assert torch.equal(aux[0], expected_sum)
    # The capture is non-consuming: the boundary still reduces exactly once.
    _, new_residual = stream.prepare(attn_out, norm, None)
    assert tp2.calls == 2
    assert torch.equal(new_residual, expected_sum)


def test_eagle_aux_capture_without_stream_keeps_legacy_sum():
    hidden = torch.randn(2, HIDDEN)
    residual = torch.randn(2, HIDDEN)
    expected = hidden + residual

    aux: list[torch.Tensor] = []
    _EagleModel()._maybe_add_hidden_state(aux, 0, hidden, residual, None)

    assert len(aux) == 1
    assert torch.equal(aux[0], expected)
