# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM120 FP8 paged MQA logits kernels of the DSA indexer decode stage:
FlashInfer's route against the torch reference and, where DeepGEMM has a
matching SM120 kernel, against DeepGEMM; and DeepGEMM's SM120 kernel against
the reference at every depth vLLM hands it natively."""

import random
from dataclasses import dataclass

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
# Stands in for max_model_len: the layer sizes the logits by it so the top-k
# consumer sees the same width as with DeepGEMM.
MAX_CONTEXT_LEN = 4096
AVG_KV = 2048


def _sm120_device() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        120
    )


def _sm120_with_flashinfer() -> bool:
    return _sm120_device() and has_flashinfer_sm120_paged_mqa_logits()


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


@dataclass
class _PagedMQAInputs:
    """One paged MQA-logits problem in the kernels' shared contract, plus the
    dequantized cache for the reference."""

    q_fp8: torch.Tensor  # [B, next_n, H, 128] e4m3
    kv_fused: torch.Tensor  # [blocks, page_kv, 1, 132] uint8 view
    kv_dequantized: torch.Tensor  # [blocks, page_kv, 128] fp32
    weights: torch.Tensor  # [B * next_n, H] fp32
    context_lens: torch.Tensor  # [B] int32: each request's last token
    context_lens_2d: torch.Tensor  # [B, next_n] int32: each Q row's length
    block_tables: torch.Tensor  # [B, max_context_len / page_kv] int32
    page_kv: int
    max_context_len: int
    num_sms: int

    @property
    def valid(self) -> torch.Tensor:
        """Cells before each row's own context length. Cells at/after it are
        unspecified for both kernels (clean_logits=False); the top-k masks them
        by length too."""
        positions = torch.arange(self.max_context_len, device=self.q_fp8.device)
        return positions[None, :] < self.context_lens_2d.reshape(-1, 1)


def _paged_mqa_inputs(
    page_kv: int, batch_size: int, next_n: int, heads: int, layers_per_block: int
) -> _PagedMQAInputs:
    """Random FP8 indexer inputs over a shuffled block pool, one request short
    enough to end inside its first page. ``layers_per_block`` > 1 hands the
    kernel the strided per-layer view of a block-outermost KV cache layout
    (DeepSeek-V4/V4.1): every layer's page sits in one block."""
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda")
    num_blocks = batch_size * cdiv(MAX_CONTEXT_LEN, page_kv) + 8

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
        int(0.8 * AVG_KV), int(1.2 * AVG_KV), (batch_size,), device=device
    ).to(torch.int32)
    context_lens[-1] = next_n + 5

    # Sized like the model runner's block table (max_model_len / page_kv
    # columns, unused entries 0): FlashInfer requires
    # max_context_len <= block_tables.shape[1] * page_kv.
    block_tables = torch.zeros(
        (batch_size, cdiv(MAX_CONTEXT_LEN, page_kv)), dtype=torch.int32, device=device
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
    return _PagedMQAInputs(
        q_fp8=q_fp8,
        kv_fused=kv_fused,
        kv_dequantized=kv_dequantized,
        weights=weights,
        context_lens=context_lens,
        context_lens_2d=context_lens_2d,
        block_tables=block_tables,
        page_kv=page_kv,
        max_context_len=MAX_CONTEXT_LEN,
        num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
    )


def _assert_matches_reference(
    inputs: _PagedMQAInputs, logits: torch.Tensor
) -> torch.Tensor:
    """Check a kernel's logits view and values against the fp32 reference;
    return the logits with the unspecified cells zeroed."""
    batch_size, next_n = inputs.q_fp8.shape[:2]
    # The [B * next_n, max_context_len] fp32 view DeepGEMM returns (a
    # row-padded buffer may sit underneath; consumers go through stride(0)).
    assert logits.shape == (batch_size * next_n, inputs.max_context_len)
    assert logits.dtype == torch.float32
    assert logits.stride(1) == 1

    ref_logits = _reference_logits(
        inputs.q_fp8,
        inputs.kv_dequantized,
        inputs.weights,
        inputs.context_lens,
        inputs.block_tables,
        inputs.max_context_len,
    )
    valid = inputs.valid
    logits_valid = logits.masked_fill(~valid, 0)
    ref_valid = ref_logits.masked_fill(~valid, 0)
    assert torch.isfinite(logits_valid).all()
    diff = calc_diff(logits_valid, ref_valid)
    assert diff < 1e-3, f"{diff=}"
    return logits_valid


def _deep_gemm_logits(inputs: _PagedMQAInputs) -> torch.Tensor:
    """DeepGEMM's SM120 paged MQA logits on ``inputs``: the kernel accepts
    block_kv 32, 64 and 128 (pages above 64 run as 64-row tiles), 16, 32 and
    64 heads and, templated on next_n, every depth; it addresses pages through
    the view's block stride like FlashInfer."""
    schedule = get_paged_mqa_logits_metadata(
        inputs.context_lens_2d, inputs.page_kv, inputs.num_sms
    )
    return fp8_fp4_paged_mqa_logits(
        (inputs.q_fp8, None),
        inputs.kv_fused,
        inputs.weights,
        inputs.context_lens_2d,
        inputs.block_tables,
        schedule,
        inputs.max_context_len,
        clean_logits=False,
    )


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
    inputs = _paged_mqa_inputs(page_kv, batch_size, next_n, heads, layers_per_block)

    schedule = flashinfer_sm120_get_paged_mqa_logits_metadata(
        inputs.context_lens_2d, page_kv, inputs.num_sms
    )
    assert schedule.shape == (inputs.num_sms + 1, 2)
    assert schedule.dtype == torch.int32

    logits = flashinfer_sm120_fp8_paged_mqa_logits(
        inputs.q_fp8,
        inputs.kv_fused,
        inputs.weights,
        inputs.context_lens_2d,
        inputs.block_tables,
        schedule,
        inputs.max_context_len,
        clean_logits=False,
    )
    logits_valid = _assert_matches_reference(inputs, logits)

    # Both kernels read identical FP8 inputs and DeepGEMM accepts every shape
    # FlashInfer routes here, so where both exist their logits must agree.
    if has_deep_gemm():
        deep_gemm_valid = _deep_gemm_logits(inputs).masked_fill(~inputs.valid, 0)
        torch.testing.assert_close(logits_valid, deep_gemm_valid, rtol=1e-3, atol=1e-2)


@pytest.mark.skipif(
    not (_sm120_device() and has_deep_gemm()),
    reason="needs an SM12x GPU and DeepGEMM",
)
@pytest.mark.parametrize("page_kv", [64, 128])
# Every depth vLLM hands DeepGEMM natively on SM12x (_supports_native_decode):
# the kernel accepts next_n as a template parameter (kNextN), so the depths
# the pinned FlashInfer catalog does not export (3, 5 and 6), where "auto"
# stays on DeepGEMM, and the odd depths' padded Q atoms run here against the
# reference instead of only where FlashInfer has a route to compare with.
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("heads", [32, 64])
def test_deep_gemm_sm120_fp8_paged_mqa_logits_native_next_n(
    page_kv: int, next_n: int, heads: int
) -> None:
    inputs = _paged_mqa_inputs(
        page_kv, batch_size=2, next_n=next_n, heads=heads, layers_per_block=1
    )
    _assert_matches_reference(inputs, _deep_gemm_logits(inputs))
