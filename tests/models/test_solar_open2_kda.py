# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import solar_open2

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def clear_warmup_state():
    caches = (
        solar_open2._KDA_PREFILL_WARMED_SIGNATURES,
        solar_open2._KDA_PREFILL_FAILED_ATTEMPTS,
        solar_open2._KDA_PREFILL_STATE_DTYPE_FAILURES,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


def _make_layer(warmup_tokens: int = 2048):
    return SimpleNamespace(
        local_num_heads=2,
        head_dim=4,
        _kda_prefill_warmup_tokens=warmup_tokens,
        A_log=torch.ones(2, dtype=torch.float32),
        dt_bias=torch.ones(2, dtype=torch.float32),
        prefix="model.layers.0.self_attn",
        get_state_dtype=lambda: (None, torch.float32),
    )


def _warmup(layer, q_proj_states):
    solar_open2.SolarOpen2KimiDeltaAttention._warmup_prefill_kernels(
        layer, q_proj_states
    )


def test_kda_attention_op_is_a_graph_splitting_op():
    """The KDA op must split the piecewise graph.

    It reads `attn_metadata` from the forward context and writes the KDA
    caches, so capturing it inside a cudagraph segment freezes the metadata
    from capture time. The model then still produces fluent text while
    attending over the wrong state, which no cheap test would catch.
    """
    from vllm.config.compilation import CompilationConfig

    assert "vllm::solar_open2_kda_attention" in CompilationConfig._attention_ops


def test_kda_attention_op_registration_passes_opcheck(monkeypatch):
    """opcheck the custom op: schema and mutation annotations, fake impl.

    The op resolves its layer through the forward context, so a stub layer
    that mutates only the declared-mutable ``core_attn_out`` stands in for
    the engine.
    """
    from vllm import forward_context

    layer_name = "opcheck.kda"

    class _StubLayer:
        def _forward(
            self,
            *,
            q_proj_states,
            k_proj_states,
            v_proj_states,
            g1,
            beta,
            core_attn_out,
        ):
            core_attn_out.add_(1.0)

    ctx = forward_context.ForwardContext(
        no_compile_layers={layer_name: _StubLayer()},
        attn_metadata=None,
        slot_mapping=None,
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)

    n, h, d = 4, 2, 8
    torch.library.opcheck(
        torch.ops.vllm.solar_open2_kda_attention.default,
        (
            torch.randn(n, h * d),
            torch.randn(n, h * d),
            torch.randn(n, h * d),
            torch.randn(1, n, h, d),
            torch.rand(1, n, h),
            torch.zeros(1, n, h, d),
            layer_name,
        ),
    )


def test_warmup_token_count_tracks_the_scheduler_batch_size():
    count = solar_open2._kda_warmup_token_count
    chunk = solar_open2.FLA_CHUNK_SIZE
    cap = solar_open2._KDA_WARMUP_MAX_TOKENS

    # Autotuning on a single chunk picks configs that lose ~20% on a real
    # prefill, so a realistic batch must not collapse back to one chunk.
    assert count(8192) == cap
    assert count(cap) == cap
    assert count(1024) == 1024
    assert count(chunk * 3 + 1) == chunk * 3
    assert cap % chunk == 0

    # An unknown batch size must not fall back to the one value measured to
    # select bad configs.
    assert count(None) == cap
    assert count(0) == cap

    # A genuinely tiny batch size is honoured, down to one runnable chunk.
    for tiny in (1, chunk - 1, chunk):
        assert count(tiny) == chunk


def test_warmup_runs_once_per_signature_at_the_configured_token_count(monkeypatch):
    calls = []

    def fake_kernel(**kwargs):
        calls.append((kwargs["q"].shape[1], kwargs["cu_seqlens"].tolist()))

    monkeypatch.setattr(solar_open2, "chunk_kda_with_fused_gate", fake_kernel)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    q_proj_states = torch.empty(1, dtype=torch.float32)
    _warmup(_make_layer(warmup_tokens=1024), q_proj_states)
    # A second layer with the same signature reuses the warmed kernels.
    _warmup(_make_layer(warmup_tokens=1024), q_proj_states)

    assert calls == [(1024, [0, 1024])]
    assert len(solar_open2._KDA_PREFILL_WARMED_SIGNATURES) == 1


def test_failed_warmup_retries_are_bounded_per_signature(monkeypatch):
    kernel_calls = 0

    def fail(**kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        raise RuntimeError("synthetic warmup failure")

    monkeypatch.setattr(solar_open2, "chunk_kda_with_fused_gate", fail)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    q_proj_states = torch.empty(1, dtype=torch.float32)
    for _ in range(3):
        _warmup(_make_layer(), q_proj_states)

    # Bounded attempts, not one retry per layer: the real model has 36 KDA
    # layers and a deterministic failure must not be retried by each of them.
    assert kernel_calls == solar_open2._KDA_PREFILL_MAX_FAILED_ATTEMPTS
    assert not solar_open2._KDA_PREFILL_WARMED_SIGNATURES
