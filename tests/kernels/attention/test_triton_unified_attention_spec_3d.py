# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""3D (split-KV) launch for short multi-token causal queries (spec-decode verify).

Compares the 3D path, enabled through VLLM_TRITON_3D_MAX_Q, against a PyTorch
reference. The negative-logit sweep covers rows whose keys are all masked in
one segment: without the epilogue guard those rows report M = 0 and underflow
the real segments in reduce_segments.
"""

import pytest
import torch
from vllm.platforms import current_platform

import vllm.v1.attention.ops.triton_unified_attention as ua

pytestmark = pytest.mark.skip_global_cleanup

DEVICE_TYPE = current_platform.device_type
FP8_DTYPE = current_platform.fp8_dtype()
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE = 24, 4, 256
NUM_SEGMENTS = 16
SEQ_THRESHOLD_3D = 32


def _reference(q, k_seq, v_seq, context_len):
    rep = NUM_Q_HEADS // NUM_KV_HEADS
    k = k_seq.repeat_interleave(rep, dim=1).float()
    v = v_seq.repeat_interleave(rep, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * HEAD_SIZE**-0.5
    pos = torch.arange(k.shape[0], device=q.device)[None, :]
    limit = context_len + torch.arange(q.shape[0], device=q.device)[:, None]
    scores = scores.masked_fill(pos > limit, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def _run(seq_lens, q_len, block_size, kv_dtype, max_q_3d, negative):
    device = DEVICE_TYPE
    num_seqs = len(seq_lens)
    blocks_per_seq = [-(-s // block_size) for s in seq_lens]
    num_blocks = sum(blocks_per_seq) + 1
    shape = (num_blocks, block_size, NUM_KV_HEADS, HEAD_SIZE)
    k_cache = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(shape, device=device, dtype=torch.bfloat16)
    q = torch.randn(
        num_seqs * q_len, NUM_Q_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16
    )
    if negative:
        k_cache = k_cache.abs() + 1
        v_cache = v_cache.abs() + 1
        q = -8 * q.abs()
    if kv_dtype == "fp8":
        k_cache, v_cache = k_cache.to(FP8_DTYPE), v_cache.to(FP8_DTYPE)

    block_table = torch.zeros(
        num_seqs, max(blocks_per_seq), dtype=torch.int32, device=device
    )
    perm = torch.randperm(num_blocks - 1, device=device) + 1
    start = 0
    for i, n in enumerate(blocks_per_seq):
        block_table[i, :n] = perm[start : start + n]
        start += n
    cu_seqlens_q = torch.arange(
        0, (num_seqs + 1) * q_len, q_len, dtype=torch.int32, device=device
    )
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    descale = torch.ones(num_seqs, NUM_KV_HEADS, device=device, dtype=torch.float32)
    out = torch.empty_like(q)
    segm_output = torch.empty(
        SEQ_THRESHOLD_3D,
        NUM_Q_HEADS,
        NUM_SEGMENTS,
        HEAD_SIZE,
        device=device,
        dtype=torch.float32,
    )
    segm_max = torch.empty(
        SEQ_THRESHOLD_3D, NUM_Q_HEADS, NUM_SEGMENTS, device=device, dtype=torch.float32
    )
    segm_expsum = torch.empty_like(segm_max)

    ua._TRITON_3D_MAX_Q = max_q_3d
    ua.unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=max(seq_lens),
        softmax_scale=HEAD_SIZE**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
        seq_threshold_3D=SEQ_THRESHOLD_3D,
        num_par_softmax_segments=NUM_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
    )

    refs = []
    for i, s in enumerate(seq_lens):
        idx = block_table[i, : blocks_per_seq[i]].long()
        k_seq = k_cache[idx].float().reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:s]
        v_seq = v_cache[idx].float().reshape(-1, NUM_KV_HEADS, HEAD_SIZE)[:s]
        refs.append(_reference(q[i * q_len : (i + 1) * q_len], k_seq, v_seq, s - q_len))
    return out.float(), torch.cat(refs)


@pytest.fixture(autouse=True)
def _restore_gate():
    saved = ua._TRITON_3D_MAX_Q
    yield
    ua._TRITON_3D_MAX_Q = saved


@pytest.mark.parametrize(
    "seq_lens",
    [
        [5],
        [37],
        [100],
        [600],
        [1601],
        [4100],
        [16384],
        [32768],
        [600, 16384],
        [5, 37, 4100, 100],
    ],
)
@pytest.mark.parametrize("q_len", [2, 4])
@pytest.mark.parametrize("block_size", [16, 800, 1600])
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("negative", [False, True])
@torch.inference_mode()
def test_spec_verify_3d_matches_reference(
    seq_lens, q_len, block_size, kv_dtype, negative
):
    torch.manual_seed(0)
    out, ref = _run(seq_lens, q_len, block_size, kv_dtype, q_len, negative)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("negative", [False, True])
@torch.inference_mode()
def test_spec_verify_3d_masked_segment_sweep(negative):
    # Every short length: some put a segment boundary just before the last
    # query tokens, leaving earlier query rows with a fully masked segment.
    torch.manual_seed(0)
    for seq_len in range(8, 521):
        out, ref = _run([seq_len], 4, 16, "bf16", 4, negative)
        assert not torch.isnan(out).any(), seq_len
        torch.testing.assert_close(
            out, ref, atol=2e-2, rtol=2e-2, msg=f"seq_len={seq_len}"
        )
