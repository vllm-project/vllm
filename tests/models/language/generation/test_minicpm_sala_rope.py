# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the MiniCPM-SALA lightning-layer RoPE.

Guards two real, historical failure modes:

1. An earlier revision ZEROED q/k under ``use_rope`` -- justified at the
   time by a bisect run whose HF reference had zeroed ``cos_cached``. That
   was a loading-harness artifact: the reference registers cos/sin as
   NON-persistent buffers rebuilt in ``__init__`` (they are never read from
   safetensors), so a correct HF load always applies real rotations. Zeroed
   q/k silence all 24 lightning layers (their attention output becomes
   exactly 0). ``test_rope_is_not_degenerate`` fails loudly on any such
   policy returning.

2. Numerics drift vs HF: ``apply_rotary_pos_emb`` in the reference builds
   fp32 cos/sin, upcasts q/k to fp32 for the rotation, and casts back.
   ``test_rope_matches_hf_reference`` compares the port's helper against an
   independent, literal transcription of the HF 4D formula.

Pure-torch: no GPU, no weights, no network.
"""

import pytest
import torch

from vllm.model_executor.models.minicpm_sala import (
    _apply_hf_rotary_bhtd,
    _build_rope_inv_freq,
)

pytestmark = pytest.mark.hybrid_model

HEAD_DIM = 128
ROPE_THETA = 10000.0


def _hf_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_apply_rotary_4d(
    q: torch.Tensor,  # (bs, heads, seq, head_dim)
    k: torch.Tensor,
    position_ids: torch.Tensor,  # (bs, seq)
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal transcription of the reference ``MiniCPMRotaryEmbedding`` +
    ``apply_rotary_pos_emb`` pair (modeling_minicpm_sala.py), kept
    independent of the port's helper on purpose."""
    orig_dtype = k.dtype
    seq_len = int(position_ids.max().item()) + 1
    t = torch.arange(seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[position_ids].unsqueeze(1)  # fp32, (bs, 1, seq, dim)
    sin = emb.sin()[position_ids].unsqueeze(1)
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    q_embed = (q_fp32 * cos) + (_hf_rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (_hf_rotate_half(k_fp32) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "positions",
    [
        list(range(8)),  # prefill from 0
        [5000],  # single decode step at a deep offset
        [7, 123, 4096],  # non-contiguous (mixed batch slots)
    ],
)
def test_rope_matches_hf_reference(dtype: torch.dtype, positions: list[int]) -> None:
    torch.manual_seed(0)
    num_heads = 4
    n = len(positions)
    pos = torch.tensor(positions, dtype=torch.long)
    inv_freq = _build_rope_inv_freq(HEAD_DIM, ROPE_THETA)

    q = torch.randn(n, num_heads, HEAD_DIM, dtype=dtype)
    k = torch.randn(n, num_heads, HEAD_DIM, dtype=dtype)

    q_out, k_out = _apply_hf_rotary_bhtd(q, k, pos, inv_freq)

    # HF layout: (bs=1, heads, seq, head_dim), position_ids (1, seq).
    q4 = q.transpose(0, 1).unsqueeze(0)
    k4 = k.transpose(0, 1).unsqueeze(0)
    q4_ref, k4_ref = _hf_apply_rotary_4d(q4, k4, pos.unsqueeze(0), inv_freq)
    q_ref = q4_ref.squeeze(0).transpose(0, 1)
    k_ref = k4_ref.squeeze(0).transpose(0, 1)

    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=0)
    assert q_out.dtype == dtype
    assert k_out.dtype == dtype


def test_rope_position_zero_is_identity() -> None:
    """cos(0)=1, sin(0)=0: position 0 must pass q/k through unchanged."""
    torch.manual_seed(1)
    inv_freq = _build_rope_inv_freq(HEAD_DIM, ROPE_THETA)
    q = torch.randn(1, 2, HEAD_DIM)
    k = torch.randn(1, 2, HEAD_DIM)
    q_out, k_out = _apply_hf_rotary_bhtd(q, k, torch.tensor([0]), inv_freq)
    torch.testing.assert_close(q_out, q, rtol=0, atol=1e-6)
    torch.testing.assert_close(k_out, k, rtol=0, atol=1e-6)


def test_rope_is_not_degenerate() -> None:
    """CRITICAL: RoPE is a rotation -- it must preserve norms and must NOT
    zero its inputs. A zeroing 'RoPE policy' (as an earlier revision had)
    kills the entire lightning branch of the model."""
    torch.manual_seed(2)
    inv_freq = _build_rope_inv_freq(HEAD_DIM, ROPE_THETA)
    n, h = 16, 4
    pos = torch.arange(n)
    q = torch.randn(n, h, HEAD_DIM)
    k = torch.randn(n, h, HEAD_DIM)
    q_out, k_out = _apply_hf_rotary_bhtd(q, k, pos, inv_freq)

    assert q_out.abs().sum() > 0, "RoPE must not zero q"
    assert k_out.abs().sum() > 0, "RoPE must not zero k"
    # Rotations are norm-preserving on each (even, odd) coordinate pair.
    torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_out.norm(dim=-1), k.norm(dim=-1), rtol=1e-5, atol=1e-5)
    # And positions > 0 must actually rotate (differ from the input).
    assert not torch.allclose(q_out[1:], q[1:]), "RoPE must rotate q at pos>0"
