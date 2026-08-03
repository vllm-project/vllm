# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MLA decode with fewer than 16 query heads on non-CDNA4 ROCm.

AITER's small-head decode kernel (``mla_gluon``) asserts gfx950, so elsewhere
the backend pads q up to 16 heads and drops the filler from the output. Head
counts that divide 16 are widened by whole repeats; the rest (e.g. Kimi-K3's
96 heads at TP=8 -> 12) are zero-padded. Both must round-trip exactly, which
holds only because the decode kernel treats query heads independently.
"""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops
from vllm.platforms import current_platform

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM


def _on_rocm_aiter() -> bool:
    return current_platform.is_rocm() and is_aiter_found()


pytestmark = pytest.mark.skipif(
    not _on_rocm_aiter(),
    reason="AITER MLA small-head padding is ROCm-only",
)


@pytest.mark.parametrize("num_heads", [1, 4, 5, 8, 9, 12])
def test_pad_unpad_round_trip(num_heads: int):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    q = torch.randn(4, num_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    padded = AiterMLAHelper.get_mla_padded_q(num_heads, q)

    assert padded.shape[1] == AiterMLAHelper._AITER_MIN_MLA_HEADS
    assert torch.equal(AiterMLAHelper.get_mla_unpadded_o(num_heads, padded), q)


def test_gluon_decode_only_on_gfx950():
    """Small-head decode must not select Gluon where the kernel cannot run."""
    from vllm.platforms.rocm import on_gfx950
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    assert AiterMLAHelper.use_gluon_decode(12, 1) is on_gfx950()


@pytest.mark.parametrize("num_heads", [1, 5, 9, 12])
@torch.inference_mode()
def test_decode_ignores_padded_heads(num_heads: int):
    """Zeroing heads >= num_heads must not perturb the retained head outputs."""
    torch.manual_seed(0)
    batch, seq_len = 4, 128
    device = "cuda"
    scale = HEAD_DIM**-0.5
    min_heads = 16

    q = torch.randn(batch, min_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(batch * seq_len, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_indptr = torch.arange(
        0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(batch * seq_len, dtype=torch.int32, device=device)
    last_page_len = torch.ones(batch, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)

    def decode(query: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            batch, min_heads, KV_LORA_RANK, dtype=query.dtype, device=device
        )
        rocm_aiter_ops.mla_decode_fwd(
            query,
            kv.unsqueeze(2),
            out,
            scale,
            qo_indptr,
            1,
            kv_indptr,
            kv_indices,
            last_page_len,
            q_scale=None,
            kv_scale=None,
        )
        return out

    reference = decode(q)
    zero_filled = q.clone()
    zero_filled[:, num_heads:, :] = 0

    assert torch.equal(reference[:, :num_heads], decode(zero_filled)[:, :num_heads])
