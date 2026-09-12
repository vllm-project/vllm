# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin how the AITER MLA decode kernel treats causality.

Two facts are asserted here, because the ROCm MLA backend's non-causal
multi-token support is built on the second one only being reachable via the
first:

1. The persistent ASM decode kernel masks causally regardless of the
   ``is_causal`` flag handed to ``get_mla_metadata_v1``: a block built with
   ``is_causal=False`` still matches a causal reference. If a future aiter
   release starts honouring the flag, this test fails and the backend can be
   simplified to just pass it through.

2. Splitting the block into ``qseqlen=1`` rows, each over the request's whole
   KV range, does produce non-causal attention. This is the mechanism
   ``AiterMLAImpl._forward_decode_non_causal`` uses.
"""

import pytest
import torch

from vllm.platforms import current_platform

NUM_HEADS = 16
QLEN = 2
SEQ_LEN = 10
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM


def _reference(q, kv, scale, causal):
    """Attention over a single request's whole KV range.

    Under causal masking, query row ``t`` of a ``QLEN`` block sees
    ``SEQ_LEN - QLEN + t + 1`` positions; non-causally every row sees all
    ``SEQ_LEN``.
    """
    out = torch.empty(QLEN, NUM_HEADS, KV_LORA_RANK, dtype=torch.float32)
    for t in range(QLEN):
        end = SEQ_LEN - QLEN + t + 1 if causal else SEQ_LEN
        scores = (q[t].float() @ kv[:end].float().T) * scale
        probs = scores.softmax(dim=-1)
        out[t] = probs @ kv[:end, :KV_LORA_RANK].float()
    return out


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-6)).item()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm/aiter only")
def test_aiter_mla_decode_causality():
    aiter = pytest.importorskip("aiter")
    from aiter import get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd

    torch.manual_seed(0)
    device = "cuda"
    scale = HEAD_DIM**-0.5
    # FP8 KV is what puts a small-head verify block on the ASM path at all.
    kv_dtype = torch.float8_e4m3fn

    q_ref = torch.randn(QLEN, NUM_HEADS, HEAD_DIM, device=device) / 4
    kv_ref = torch.randn(SEQ_LEN, HEAD_DIM, device=device) / 4
    q = q_ref.to(kv_dtype)
    kv_buffer = kv_ref.to(kv_dtype).view(SEQ_LEN, 1, 1, HEAD_DIM)

    kv_indices = torch.arange(SEQ_LEN, dtype=torch.int32, device=device)
    ones = torch.ones(1, dtype=torch.float32, device=device)

    def run(qo_indptr, kv_indptr, indices, last_page, num_rows, max_qo, is_causal):
        batch = last_page.numel()
        # The buffer shapes do not depend on causality, which is why serving a
        # non-causal block needs no re-sizing.
        bufs = [
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype in aiter.get_mla_metadata_info_v1(
                batch, max_qo, NUM_HEADS, kv_dtype, kv_dtype, False
            )
        ]
        (
            work_meta_data,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        ) = bufs
        get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            last_page,
            NUM_HEADS,
            1,
            is_causal,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=max_qo,
            uni_seqlen_qo=max_qo,
            fast_mode=True,
            dtype_q=kv_dtype,
            dtype_kv=kv_dtype,
        )
        meta = dict(
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )
        out = torch.zeros(
            num_rows, NUM_HEADS, KV_LORA_RANK, dtype=torch.bfloat16, device=device
        )
        mla_decode_fwd(
            q.view(num_rows, NUM_HEADS, HEAD_DIM),
            kv_buffer,
            out,
            qo_indptr,
            kv_indptr,
            indices,
            last_page,
            max_qo,
            sm_scale=scale,
            q_scale=ones,
            kv_scale=ones,
            **meta,
        )
        torch.cuda.synchronize()
        return out.float().cpu()

    ref_causal = _reference(q_ref.cpu(), kv_ref.cpu(), scale, causal=True)
    ref_non_causal = _reference(q_ref.cpu(), kv_ref.cpu(), scale, causal=False)
    # fp8 quantization noise; the causal/non-causal gap is an order of
    # magnitude larger, so this separates them comfortably.
    tol = 5e-2
    # Sanity: the two references really are far apart on row 0, so a match
    # below tol identifies which one the kernel computed.
    assert _rel(ref_causal[0], ref_non_causal[0]) > 4 * tol

    # --- 1. one qseqlen=2 block, built both ways -------------------------
    block_qo = torch.tensor([0, QLEN], dtype=torch.int32, device=device)
    block_kv = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=device)
    block_last = torch.ones(1, dtype=torch.int32, device=device)
    got_causal = run(block_qo, block_kv, kv_indices, block_last, QLEN, QLEN, True)
    got_flagged = run(block_qo, block_kv, kv_indices, block_last, QLEN, QLEN, False)

    # Row 0 is where the two causalities disagree (row QLEN-1 sees everything
    # either way).
    assert _rel(got_causal[0], ref_causal[0]) < tol
    # is_causal=False computed the causal answer anyway: the flag reaches the
    # work metadata but never the mask.
    assert _rel(got_flagged[0], ref_causal[0]) < tol
    assert _rel(got_flagged[0], ref_non_causal[0]) > tol, (
        "aiter now honours is_causal in the MLA decode metadata; "
        "AiterMLAImpl._forward_decode_non_causal can be replaced by "
        "threading the flag through get_mla_metadata_v1"
    )

    # --- 2. the same block as qseqlen=1 rows over the full range ---------
    row_qo = torch.arange(QLEN + 1, dtype=torch.int32, device=device)
    row_kv = torch.arange(
        0, SEQ_LEN * (QLEN + 1), SEQ_LEN, dtype=torch.int32, device=device
    )
    row_last = torch.ones(QLEN, dtype=torch.int32, device=device)
    got_rows = run(row_qo, row_kv, kv_indices.repeat(QLEN), row_last, QLEN, 1, True)
    for t in range(QLEN):
        assert _rel(got_rows[t], ref_non_causal[t]) < tol
    assert _rel(got_rows[0], ref_causal[0]) > tol
