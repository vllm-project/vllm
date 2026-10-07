# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's native NVFP4 decode against FLASHINFER_MLA_SPARSE's FP8-staged decode.

The backend decodes nvfp4_ds_mla one of two ways. Staged: each token's top-k rows are
copied as FP8 (dequant / k_scale) and the FP8 TRTLLM-gen kernel runs with bmm scales
that carry k_scale. Native: flashinfer.mla.nvfp4_sparse_mla_decode reads the rows as
stored, with both scales divided by k_scale. Both must attend over the same rows of a
per-layer cache whose blocks sit between another layer's blocks.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_nvfp4_sparse_mla_decode
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseImpl
from vllm.v1.attention.backends.mla.nvfp4_ds_mla_fp8_gather import (
    FP8_STAGING_PAGE_SIZE,
    FP8_STAGING_ROW_DIM,
    NVFP4_DS_MLA_ROW_BYTES,
    gather_nvfp4_ds_mla_topk_to_fp8,
)
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(100)
            or current_platform.is_device_capability(103)
        )
        and has_flashinfer_nvfp4_sparse_mla_decode()
    ),
    reason="needs SM100/SM103 and FlashInfer's native NVFP4 sparse MLA decode",
)

NUM_HEADS, HEAD_DIM, V_HEAD_DIM = 16, 576, 512
QK_NOPE_HEAD_DIM, KV_LORA_RANK, QK_ROPE_HEAD_DIM = 192, 512, 64
BLOCK_SIZE, NUM_BLOCKS, NUM_LAYERS, TOPK = 64, 1024, 2, 2048
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _layer_cache(g: torch.Generator) -> torch.Tensor:
    """Layer 1 of a two-layer cache: its blocks alternate with layer 0's."""
    shape = (NUM_BLOCKS, NUM_LAYERS, BLOCK_SIZE, NVFP4_DS_MLA_ROW_BYTES)
    kv = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=g)
    rope = torch.randn(*shape[:3], 64, device="cuda", generator=g) * 0.5
    kv[..., 256:320] = rope.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(*shape[:3], 32, device="cuda", generator=g) * 0.99 + 0.01
    kv[..., 320:] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    return kv[:, 1]


def _dequantize(rows: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1, device=rows.device)
    raw = rows[idx.clamp_min(0).long()]
    nope = torch.stack(
        [lut[(raw[..., :256] & 15).long()], lut[(raw[..., :256] >> 4).long()]], -1
    ).flatten(-2)
    b = torch.arange(32, device=rows.device)
    scale = raw[..., 320 + 8 * (b % 4) + b // 4].contiguous().view(torch.float8_e4m3fn)
    nope = nope * scale.float().repeat_interleave(16, -1)
    rope = raw[..., 256:320].contiguous().view(torch.float8_e4m3fn).float()
    return torch.cat([nope, rope], -1)


def _reference(q, rows, idx):
    k = _dequantize(rows, idx)
    s = torch.einsum("thd,tkd->thk", q.float(), k) * SOFTMAX_SCALE
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    p = torch.softmax(s, -1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", p, k[..., :V_HEAD_DIM])


def _relative_error(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("num_tokens", [1, 5, 20, 35])
@pytest.mark.parametrize("k_scale", [1.0, 0.5])
def test_native_decode_matches_staged_decode(num_tokens, k_scale):
    from flashinfer.mla import (
        nvfp4_sparse_mla_decode,
        trtllm_batch_decode_with_kv_cache_mla,
    )

    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    kv_cache = _layer_cache(g)
    rows, block_stride_rows = flat_kv_row_view(kv_cache, BLOCK_SIZE)
    assert block_stride_rows == NUM_LAYERS * BLOCK_SIZE
    blocks = torch.randint(
        0, NUM_BLOCKS, (num_tokens, TOPK), device="cuda", generator=g
    )
    offsets = torch.randint(
        0, BLOCK_SIZE, (num_tokens, TOPK), device="cuda", generator=g
    )
    physical_topk = (blocks * block_stride_rows + offsets).to(torch.int32)
    physical_topk[::3, TOPK - 300 :] = -1  # short contexts end in empty slots
    q = (
        torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device="cuda", generator=g) * 0.5
    ).to(torch.float8_e4m3fn)
    # The backend's fp8 scales: k_scale is folded into both.
    bmm1_scale, bmm2_scale, inv_k_scale = (
        SOFTMAX_SCALE * k_scale,
        k_scale,
        1.0 / k_scale,
    )

    staging_rows = torch.empty(
        num_tokens * TOPK, FP8_STAGING_ROW_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    staging_indices = torch.empty(num_tokens, TOPK, dtype=torch.int32, device="cuda")
    gather_nvfp4_ds_mla_topk_to_fp8(
        kv_cache, physical_topk, staging_rows, staging_indices, inv_k_scale
    )
    staged = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=staging_rows.view(
            -1, FP8_STAGING_PAGE_SIZE, FP8_STAGING_ROW_DIM
        ).unsqueeze(1),
        workspace_buffer=torch.zeros(256 << 20, dtype=torch.int8, device="cuda"),
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=staging_indices.unsqueeze(1),
        seq_lens=(physical_topk >= 0).sum(1).to(torch.int32),
        max_seq_len=TOPK,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sparse_mla_top_k=TOPK,
    ).view(num_tokens, NUM_HEADS, V_HEAD_DIM)

    native = torch.empty(
        num_tokens, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    nvfp4_sparse_mla_decode(
        q,
        rows,
        physical_topk,
        bmm1_scale=bmm1_scale * inv_k_scale,
        bmm2_scale=bmm2_scale * inv_k_scale,
        out=native,
        backend="cuda",
    )
    torch.accelerator.synchronize()

    ref = _reference(q, rows, physical_topk)
    # Native reads exact values; staging re-rounds every dequantized value to e4m3.
    assert _relative_error(native, ref) < 1e-2
    assert _relative_error(staged, ref) < 6e-2
    assert _relative_error(native, staged.float()) < 6e-2


@pytest.mark.parametrize("k_scale", [1.0, 0.5])
def test_backend_native_decode_folds_k_scale(monkeypatch, k_scale):
    """FlashInferMLASparseImpl._nvfp4_decode divides both fp8 bmm scales by k_scale."""
    num_tokens = 12
    g = torch.Generator(device="cuda").manual_seed(100)
    kv_cache = _layer_cache(g)
    rows, block_stride_rows = flat_kv_row_view(kv_cache, BLOCK_SIZE)
    blocks = torch.randint(
        0, NUM_BLOCKS, (num_tokens, TOPK), device="cuda", generator=g
    )
    offsets = torch.randint(
        0, BLOCK_SIZE, (num_tokens, TOPK), device="cuda", generator=g
    )
    physical_topk = (blocks * block_stride_rows + offsets).to(torch.int32)
    physical_topk[::3, TOPK - 300 :] = -1
    q = (
        torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device="cuda", generator=g) * 0.5
    ).to(torch.float8_e4m3fn)

    impl = object.__new__(FlashInferMLASparseImpl)
    impl._nvfp4_native_decode = True
    impl._nvfp4_inv_k_scale = 1.0 / k_scale
    # The fp8 scales _prepare_mqa_kernel builds, with a unit q_scale.
    impl.bmm1_scale = SOFTMAX_SCALE * k_scale
    impl.bmm2_scale = k_scale
    monkeypatch.setattr(
        impl,
        "_convert_logical_to_physical_topk",
        lambda topk, metadata, **_: (topk, (topk >= 0).sum(1)),
    )
    out = torch.empty(
        num_tokens, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    impl._nvfp4_decode(
        q, kv_cache, physical_topk, SimpleNamespace(block_size=BLOCK_SIZE), out
    )
    torch.accelerator.synchronize()

    assert _relative_error(out, _reference(q, rows, physical_topk)) < 1e-2
