# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2 vision window attention has to apply the per-head sink logits.

That includes the rank bookkeeping: with `--mm-encoder-tp-mode data` every rank
runs the whole encoder, so the tensor-parallel rank collapses with the size and
each rank keeps all sinks.
"""

import pytest
import torch

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models import mimo_v2_omni
from vllm.model_executor.models.interfaces import (
    supports_multimodal_encoder_tp_data,
)
from vllm.platforms import current_platform

EMBED_DIM = 256
NUM_HEADS = 4
HEAD_DIM = 64
WINDOW = 8
# One sequence shorter than the window, one longer.
SEQ_LENS = [5, 37]


@pytest.fixture
def vision_attn_env(dist_init):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(default_dtype)


def _reference(q, k, v, cu_seqlens, sinks, scale):
    """Dense windowed softmax with the sink added to each sequence's key 0."""
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs = q[start:end].float()
        ks = k[start:end].float().repeat_interleave(groups, dim=1)
        vs = v[start:end].float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        scores[..., 0] += sinks.float().view(-1, 1)
        pos = torch.arange(end - start, device=q.device)
        outside = (pos.view(-1, 1) - pos.view(1, -1)).abs() > WINDOW
        scores.masked_fill_(outside, -torch.inf)
        out[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs)
    return out


def _build_attention(num_kv_heads: int):
    from vllm.model_executor.models.mimo_v2_omni import MiMoVisionAttention

    torch.manual_seed(0)
    attn = MiMoVisionAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        qk_channels=HEAD_DIM,
        kv_channels=HEAD_DIM,
        use_sink=True,
        visual_token_window_size=WINDOW,
    ).cuda()
    attn.sinks.data.normal_()
    return attn


def _stub_encoder_parallelism(
    monkeypatch, mode: str, world_size: int, rank: int
) -> None:
    """Set the encoder mode and the rank it sees.

    The mode goes through a real `MultiModalConfig`, so `is_vit_use_data_parallel`
    runs its own logic; only the rank is stubbed, because a single-process test
    cannot be rank 1 of a multi-rank group.
    """
    monkeypatch.setattr(
        "vllm.model_executor.models.vision.get_multimodal_config",
        lambda: MultiModalConfig(mm_encoder_tp_mode=mode),
    )
    monkeypatch.setattr(
        mimo_v2_omni.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: world_size,
    )
    monkeypatch.setattr(
        mimo_v2_omni.parallel_state, "get_tensor_model_parallel_rank", lambda: rank
    )


def _attention_inputs(num_heads: int, num_kv_heads: int):
    total = sum(SEQ_LENS)
    opts = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(total, num_heads, HEAD_DIM, **opts)
    k = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    v = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, cu_seqlens


def _relative_error(attn, q, k, v, cu_seqlens, sinks):
    out = attn._forward_window_attn(q, k, v, cu_seqlens, max(SEQ_LENS))
    ref = _reference(q, k, v, cu_seqlens, sinks, attn.scale)
    # bf16 attention lands at ~2e-3 here; dropping the sinks lands at ~1e-1.
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires flash-attn")
@pytest.mark.parametrize("num_kv_heads", [NUM_HEADS, NUM_HEADS // 2])
def test_window_attention_applies_sinks(vision_attn_env, num_kv_heads):
    attn = _build_attention(num_kv_heads)
    q, k, v, cu_seqlens = _attention_inputs(NUM_HEADS, num_kv_heads)

    error = _relative_error(attn, q, k, v, cu_seqlens, attn.sinks)
    assert error < 1e-2, f"sink-corrected output is off by {error:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("tp_rank", [1, 3])
def test_window_attention_applies_sinks_under_encoder_data_parallel(
    vision_attn_env, monkeypatch, tp_rank
):
    """Every rank runs the whole encoder, so every rank keeps all sinks."""
    _stub_encoder_parallelism(monkeypatch, "data", world_size=4, rank=tp_rank)
    attn = _build_attention(NUM_HEADS)
    q, k, v, cu_seqlens = _attention_inputs(NUM_HEADS, NUM_HEADS)

    error = _relative_error(attn, q, k, v, cu_seqlens, attn.sinks)
    assert error < 1e-2, (
        f"rank {tp_rank} of the data-parallel encoder is off by {error:.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_weights_mode_slices_sinks_per_rank(vision_attn_env, monkeypatch):
    """The collapse is data-parallel only: a sharded encoder keeps its slices."""
    _stub_encoder_parallelism(monkeypatch, "weights", world_size=2, rank=1)
    attn = _build_attention(NUM_HEADS)
    heads = NUM_HEADS // 2
    q, k, v, cu_seqlens = _attention_inputs(heads, heads)

    error = _relative_error(attn, q, k, v, cu_seqlens, attn.sinks[heads:])
    assert error < 1e-2, f"rank 1 of the sharded encoder is off by {error:.2e}"


def test_model_advertises_encoder_tp_data():
    """Without the flag the config rejects the mode and falls back to weights."""
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniForCausalLM

    assert supports_multimodal_encoder_tp_data(MiMoV2OmniForCausalLM)
