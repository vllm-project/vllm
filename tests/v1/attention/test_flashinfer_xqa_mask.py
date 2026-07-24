# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact checks for the XQA draft-block mask packing.

The XQA decode kernel requires a packed ``uint16`` draft-block mask for every
``q_len > 1`` decode (causal spec decode and non-causal drafts alike). These
tests pin our packing to FlashInfer's reference layout (its
``generate_spec_dec_mask`` / ``generate_ragged_spec_dec_mask`` test helpers) so
a divergence is caught without a GPU.
"""

import pytest
import torch

fi = pytest.importorskip("vllm.v1.attention.backends.flashinfer")


def _reference_uniform(q_seq_len: int, mode: str) -> torch.Tensor:
    """Mirror of FlashInfer's ``generate_spec_dec_mask`` (batch dim dropped)."""
    num_packed = (q_seq_len + 31) // 32
    q = torch.arange(q_seq_len, dtype=torch.int32).unsqueeze(1)
    k = torch.arange(q_seq_len, dtype=torch.int32).unsqueeze(0)
    m = (
        (k <= q)
        if mode == "causal"
        else torch.ones(q_seq_len, q_seq_len, dtype=torch.bool)
    )
    padded = num_packed * 32
    if padded > q_seq_len:
        m = torch.cat(
            [m, torch.zeros(q_seq_len, padded - q_seq_len, dtype=torch.bool)], dim=1
        )
    bits = torch.tensor([1 << i for i in range(32)], dtype=torch.int64)
    mask_u32 = (
        (m.view(-1, num_packed, 32).to(torch.int64) * bits).sum(-1).to(torch.uint32)
    )
    return mask_u32.view(torch.uint16).reshape(q_seq_len, -1)


def _reference_ragged(q_lens: list[int], max_q: int, mode: str) -> torch.Tensor:
    """Mirror of FlashInfer's ``generate_ragged_spec_dec_mask``."""
    num_packed = (max_q + 31) // 32
    padded = num_packed * 32
    rows = []
    for q_len in q_lens:
        q = torch.arange(q_len, dtype=torch.int32).view(-1, 1)
        k = torch.arange(padded, dtype=torch.int32).view(1, -1)
        m = (
            ((k <= q) & (k < q_len))
            if mode == "causal"
            else (k < q_len).expand(q_len, -1)
        )
        rows.append(m)
    bits = torch.tensor([1 << i for i in range(32)], dtype=torch.int64)
    bool_mask = torch.cat(rows, dim=0).view(-1, num_packed, 32)
    mask_u32 = (bool_mask.to(torch.int64) * bits).sum(-1).to(torch.uint32)
    return mask_u32.view(torch.uint16)


@pytest.mark.parametrize("q_len", [2, 4, 8, 31, 32, 33, 64])
@pytest.mark.parametrize("causal", [True, False])
def test_uniform_mask_matches_flashinfer_reference(q_len: int, causal: bool) -> None:
    device = torch.device("cpu")
    ours = fi.make_xqa_draft_block_mask(q_len, causal, device)
    ref = _reference_uniform(q_len, "causal" if causal else "full")
    assert ours.dtype == torch.uint16
    assert ours.shape == (q_len, ((q_len + 31) // 32) * 2)
    # uint16 has no torch.equal support on some builds; compare as int16 bits.
    assert torch.equal(ours.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("causal", [True, False])
def test_ragged_mask_matches_flashinfer_reference(causal: bool) -> None:
    device = torch.device("cpu")
    q_lens = [3, 5, 2, 4]
    max_q = max(q_lens)
    ours = fi.make_xqa_ragged_draft_block_mask(q_lens, max_q, causal, device)
    ref = _reference_ragged(q_lens, max_q, "causal" if causal else "full")
    assert ours.shape == (sum(q_lens), ((max_q + 31) // 32) * 2)
    assert torch.equal(ours.view(torch.int16), ref.view(torch.int16))
