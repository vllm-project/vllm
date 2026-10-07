# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's SM120 FP8 paged MQA logits (the DSA indexer decode stage)
against the torch reference and, where DeepGEMM has a matching SM120 kernel,
against DeepGEMM."""

import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    calc_diff,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.flashinfer import (
    flashinfer_sm120_fp8_paged_mqa_logits,
    flashinfer_sm120_get_paged_mqa_logits_metadata,
    flashinfer_sm120_paged_mqa_logits_route_available,
    has_flashinfer_sm120_paged_mqa_logits,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv

INDEX_DIM = 128


def _sm120_with_flashinfer() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and has_flashinfer_sm120_paged_mqa_logits()
    )


def _quantize_kv_cache(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused FP8 page layout of the indexer cache (what
    indexer_k_quant_and_cache_kernel writes): per block, page_kv x 128 e4m3
    values followed by one fp32 scale per token.

    Returns the cache bytes viewed as ``[blocks, page_kv, 1, 132]`` and the
    dequantized fp32 values ``[blocks, page_kv, 128]`` for the reference.
    """
    num_blocks, page_kv, num_heads, head_dim = kv_cache.shape
    assert num_heads == 1
    amax = kv_cache.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = amax / 448.0
    values = (kv_cache / scale).to(torch.float8_e4m3fn)
    fused = torch.empty(
        (num_blocks, page_kv * (head_dim + 4)),
        dtype=torch.uint8,
        device=kv_cache.device,
    )
    fused[:, : page_kv * head_dim] = values.view(num_blocks, page_kv * head_dim).view(
        torch.uint8
    )
    fused[:, page_kv * head_dim :] = scale.view(num_blocks, page_kv).view(torch.uint8)
    dequantized = (values.float() * scale).squeeze(2)
    return fused.view(num_blocks, page_kv, num_heads, head_dim + 4), dequantized


def _reference_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_context_len: int,
) -> torch.Tensor:
    """fp32 reference on the dequantized inputs: per head ReLU(q . k) scaled by
    the row's head weight and summed over heads, causal within each request's
    next_n rows, -inf elsewhere."""
    batch_size, next_n, _, _ = q.shape
    page_kv = kv.shape[1]
    logits = torch.full(
        (batch_size * next_n, max_context_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    for b in range(batch_size):
        ctx = int(context_lens[b])
        num_pages = cdiv(ctx, page_kv)
        keys = kv[block_tables[b, :num_pages]].reshape(num_pages * page_kv, -1)[:ctx]
        scores = torch.einsum("nhd,ld->nhl", q[b].float(), keys)
        row_weights = weights[b * next_n : (b + 1) * next_n]
        scores = (torch.relu(scores) * row_weights[:, :, None]).sum(dim=1)
        positions = torch.arange(ctx, device=q.device)
        query_positions = ctx - next_n + torch.arange(next_n, device=q.device)
        logits[b * next_n : (b + 1) * next_n, :ctx] = torch.where(
            positions[None, :] <= query_positions[:, None], scores, float("-inf")
        )
    return logits


@pytest.mark.skipif(
    not _sm120_with_flashinfer(),
    reason="needs an SM12x GPU and a FlashInfer build with sm120_paged_mqa_logits",
)
@pytest.mark.parametrize("page_kv", [64, 128])
# next_n = 1 + num_speculative_tokens over every depth the catalog can export
# (the pinned package ships 1, 2 and 4; a newer catalog adds 3, 5 and 6). The
# route query below skips a depth the installed build does not ship.
@pytest.mark.parametrize(
    "batch_size,next_n", [(4, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
)
@pytest.mark.parametrize("heads", [32, 64])
# A block-outermost KV cache layout (DeepSeek-V4/V4.1) hands the indexer a
# strided per-layer view: every layer's page sits in one block.
@pytest.mark.parametrize("layers_per_block", [1, 2])
def test_flashinfer_sm120_fp8_paged_mqa_logits(
    page_kv: int, batch_size: int, next_n: int, heads: int, layers_per_block: int
) -> None:
    if not flashinfer_sm120_paged_mqa_logits_route_available(heads, page_kv, next_n):
        pytest.skip(
            f"FlashInfer ships no SM120 route for heads={heads}, "
            f"page_kv={page_kv}, next_n={next_n}"
        )
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda")
    # Stands in for max_model_len: the layer sizes the logits by it so the
    # top-k consumer sees the same width as with DeepGEMM.
    max_context_len = 4096
    avg_kv = 2048
    num_blocks = batch_size * cdiv(max_context_len, page_kv) + 8

    q = torch.randn(
        (batch_size, next_n, heads, INDEX_DIM), device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        (num_blocks, page_kv, 1, INDEX_DIM), device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device=device, dtype=torch.float32
    )
    context_lens = torch.randint(
        int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,), device=device
    ).to(torch.int32)
    # One request short enough to end inside its first page.
    context_lens[-1] = next_n + 5

    # Sized like the model runner's block table (max_model_len / page_kv
    # columns, unused entries 0): FlashInfer requires
    # max_context_len <= block_tables.shape[1] * page_kv.
    block_tables = torch.zeros(
        (batch_size, cdiv(max_context_len, page_kv)), dtype=torch.int32, device=device
    )
    block_pool = list(range(num_blocks))
    random.shuffle(block_pool)
    counter = 0
    for b in range(batch_size):
        for j in range(cdiv(int(context_lens[b]), page_kv)):
            block_tables[b, j] = block_pool[counter]
            counter += 1

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fused, kv_dequantized = _quantize_kv_cache(kv_cache)
    if layers_per_block > 1:
        # The indexer's pages live in the last layer slot of each block, so the
        # view is not contiguous and its block stride spans every layer.
        layer_bytes = page_kv * (INDEX_DIM + 4)
        blocks = torch.zeros(
            (num_blocks, layers_per_block * layer_bytes),
            dtype=torch.uint8,
            device=device,
        )
        blocks[:, -layer_bytes:] = kv_fused.reshape(num_blocks, layer_bytes)
        kv_fused = torch.as_strided(
            blocks,
            (num_blocks, page_kv, 1, INDEX_DIM + 4),
            (layers_per_block * layer_bytes, INDEX_DIM + 4, INDEX_DIM + 4, 1),
            storage_offset=(layers_per_block - 1) * layer_bytes,
        )
        assert not kv_fused.is_contiguous()

    # Row j of request b attends to context_lens[b] - next_n + j + 1 tokens:
    # the (B, next_n) int32 layout the metadata builder produces.
    offsets = torch.arange(next_n, device=device, dtype=torch.int32)
    context_lens_2d = (context_lens[:, None] - next_n + 1 + offsets).contiguous()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    schedule = flashinfer_sm120_get_paged_mqa_logits_metadata(
        context_lens_2d, page_kv, num_sms
    )
    assert schedule.shape == (num_sms + 1, 2)
    assert schedule.dtype == torch.int32

    logits = flashinfer_sm120_fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens_2d,
        block_tables,
        schedule,
        max_context_len,
        clean_logits=False,
    )
    # Same [B * next_n, max_context_len] fp32 view DeepGEMM returns (a
    # row-padded buffer may sit underneath; consumers go through stride(0)).
    assert logits.shape == (batch_size * next_n, max_context_len)
    assert logits.dtype == torch.float32
    assert logits.stride(1) == 1

    ref_logits = _reference_logits(
        q_fp8, kv_dequantized, weights, context_lens, block_tables, max_context_len
    )

    # Cells at/after a row's own context length are unspecified for both
    # kernels (clean_logits=False); the top-k masks them by length too.
    positions = torch.arange(max_context_len, device=device)[None, :]
    valid = positions < context_lens_2d.reshape(-1, 1)
    logits_valid = logits.masked_fill(~valid, 0)
    ref_valid = ref_logits.masked_fill(~valid, 0)
    assert torch.isfinite(logits_valid).all()
    diff = calc_diff(logits_valid, ref_valid)
    assert diff < 1e-3, f"{diff=}"

    # DeepGEMM's SM120 kernel takes block_kv 64 and, per vLLM's gate, native
    # next_n 1 and 2; both kernels read identical FP8 inputs (DeepGEMM also
    # addresses pages through the view's block stride).
    if has_deep_gemm() and page_kv == 64 and next_n in (1, 2):
        deep_gemm_schedule = get_paged_mqa_logits_metadata(
            context_lens_2d, page_kv, num_sms
        )
        deep_gemm_logits = fp8_fp4_paged_mqa_logits(
            (q_fp8, None),
            kv_fused,
            weights,
            context_lens_2d,
            block_tables,
            deep_gemm_schedule,
            max_context_len,
            clean_logits=False,
        )
        torch.testing.assert_close(
            logits_valid,
            deep_gemm_logits.masked_fill(~valid, 0),
            rtol=1e-3,
            atol=1e-2,
        )
