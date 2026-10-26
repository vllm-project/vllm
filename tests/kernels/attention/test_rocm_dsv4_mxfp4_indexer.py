# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4's ROCm MXFP4 indexer K writer: the pooling kernel plus aiter's
norm + RoPE + MXFP4 cache op, against the same op fed by torch's pooling.

The dense K6 path selects top1024/2048 against dequantized Torch scores at64heads.
"""

import itertools

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import get_cdna_version

if get_cdna_version() != 4:
    pytest.skip("CDNA4-only", allow_module_level=True)

from tests.kernels.attention.test_rocm_paged_mxfp4_indexer import (
    _assert_topk,
    _dequant,
    _scores,
)
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v4.amd.rocm import (
    compress_pool,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla.indexer import DeepSeekV32IndexerDecodeMetadata
from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
    RocmMxfp4IndexerMetadata,
    native_decode,
)
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    build_rocm_mxfp4_decode_schedule,
    rocm_mxfp4_decode_schedule_words,
    rocm_mxfp4_indexer_k_store,
    rocm_mxfp4_indexer_q_quant,
    rocm_mxfp4_sparse_attn_indexer,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

# DeepSeek-V4-Pro's indexer: 64 heads of 128, ratio 4 with overlap, 256-token
# blocks, so 64-entry K pages.
HEADS, HEAD_DIM, RATIO, PAGE = 64, 128, 4, 64
WIDTH = HEAD_DIM // 2 + HEAD_DIM // 32
STATE_BLOCK, MAX_POS = 16, 4096
DEVICE = "cuda"


class _Case:
    """Requests with random compressor states and their boundary tokens'
    slots in a paged K cache."""

    def __init__(self, seq_lens, max_pos=MAX_POS):
        self.state_width = 2 * HEAD_DIM  # coff * head_dim, kv then score
        state_blocks = sum(cdiv(n, STATE_BLOCK) for n in seq_lens)
        self.state_cache = torch.randn(
            state_blocks, STATE_BLOCK, 2 * self.state_width, device=DEVICE
        )
        width = max(cdiv(n, STATE_BLOCK) for n in seq_lens)
        self.block_table = torch.zeros(
            len(seq_lens), width, dtype=torch.int32, device=DEVICE
        )
        perm = torch.randperm(state_blocks)
        used = 0
        for req, n in enumerate(seq_lens):
            self.block_table[req, : cdiv(n, STATE_BLOCK)] = perm[
                used : used + cdiv(n, STATE_BLOCK)
            ]
            used += cdiv(n, STATE_BLOCK)

        self.positions = torch.cat([torch.arange(n) for n in seq_lens]).to(DEVICE)
        self.req = torch.cat(
            [torch.full((n,), r, dtype=torch.int32) for r, n in enumerate(seq_lens)]
        ).to(DEVICE)
        # each request's pages, scattered over a pool with spare pages
        pages = [cdiv(cdiv(n, RATIO), PAGE) for n in seq_lens]
        first_page = torch.tensor([0, *itertools.accumulate(pages)][:-1])
        self.num_pages = sum(pages) + 2
        page_of = torch.randperm(self.num_pages).to(DEVICE)
        comp = self.positions // RATIO
        page = page_of[first_page.to(DEVICE)[self.req.long()] + comp // PAGE]
        boundary = (self.positions + 1) % RATIO == 0
        self.kv_slot = torch.where(boundary, page * PAGE + comp % PAGE, -1)
        self.cos_sin = torch.randn(max_pos, 64, device=DEVICE)
        self.norm = torch.rand(HEAD_DIM, device=DEVICE) + 0.5

    def torch_pooled(self):
        """Softmax-gated pooling of each token's 2 * RATIO-token window: the
        older RATIO tokens read the first state half, the newer the second."""
        out = torch.zeros(len(self.positions), HEAD_DIM, device=DEVICE)
        for t in range(len(self.positions)):
            p = int(self.positions[t])
            if (p + 1) % RATIO:
                continue
            kv, score = [], []
            for j, q in enumerate(range(p - 2 * RATIO + 1, p + 1)):
                if q < 0:
                    continue
                row = self.state_cache[
                    self.block_table[self.req[t], q // STATE_BLOCK], q % STATE_BLOCK
                ]
                half = HEAD_DIM if j >= RATIO else 0
                kv.append(row[half : half + HEAD_DIM])
                half += self.state_width
                score.append(row[half : half + HEAD_DIM])
            out[t] = (torch.stack(kv) * torch.stack(score).softmax(0)).sum(0)
        return out

    def run_compressor(self):
        """Run the real pooling and cache-write operators on the paged input."""
        kv_cache = torch.zeros(
            self.num_pages, PAGE, WIDTH, dtype=torch.uint8, device=DEVICE
        )
        pooled = torch.empty(
            len(self.positions), HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        compress_pool(
            self.state_cache,
            self.req,
            self.positions,
            self.kv_slot,
            self.block_table,
            STATE_BLOCK,
            pooled,
            self.state_width,
            RATIO,
            True,
        )
        rocm_mxfp4_indexer_k_store(
            pooled,
            self.positions,
            self.cos_sin,
            self.norm,
            1e-6,
            kv_cache,
            self.kv_slot,
            RATIO,
            True,
            num_heads=HEADS,
        )
        return pooled, kv_cache


SEQ_LENS = pytest.mark.parametrize("seq_lens", [[37, 300, 5], [1, 260]])


@SEQ_LENS
def test_pooled_rows_match_torch(seq_lens):
    set_random_seed(0)
    case = _Case(seq_lens)
    pooled, _ = case.run_compressor()
    boundary = case.kv_slot >= 0
    torch.testing.assert_close(
        pooled[boundary].float(),
        case.torch_pooled()[boundary],
        atol=1e-2,
        rtol=1e-2,
    )


@SEQ_LENS
def test_k_pages_match_aiter_fed_by_torch(seq_lens):
    """The pages hold what aiter writes from torch's pooling: the same layout,
    and the same MXFP4 codes up to bf16 rounding of the pooled rows."""
    set_random_seed(0)
    case = _Case(seq_lens)
    _, kv_cache = case.run_compressor()
    ref = torch.zeros_like(kv_cache)
    rocm_mxfp4_indexer_k_store(
        case.torch_pooled().bfloat16(),
        case.positions,
        case.cos_sin,
        case.norm,
        1e-6,
        ref,
        case.kv_slot,
        RATIO,
        True,
        num_heads=HEADS,
    )
    assert bool(ref.any())
    assert (kv_cache != ref).float().mean().item() < 5e-3


@pytest.mark.parametrize("num_requests", [6, 8, 12, 16, 24, 32])
@pytest.mark.parametrize("context", [10800, 170000])
@pytest.mark.parametrize("topk,logits_width", [(2048, None), (1024, 262144)])
def test_k6_dense_selection_matches_torch_at_64_heads(
    num_requests, context, topk, logits_width
):
    """DSV4's K6 path selects the quantized reference at its native logits stride.

    Real pooling, writers, Q quantization, metadata and forward context feed
    the public indexer op. The reference reads token-major keys, independent
    of the shuffled, strided pages consumed by the indexer.
    """
    from aiter.ops.triton.fusions.k_norm_rope_mxfp4_cache import (
        k_norm_rope_mxfp4_cache,
    )

    set_random_seed(42)
    next_n = 7
    seq_lens = [context + 23 * r for r in range(num_requests)]
    case = _Case(seq_lens, max_pos=max(seq_lens))
    pooled, contiguous_cache = case.run_compressor()
    pool = torch.zeros(
        case.num_pages, PAGE * WIDTH + 576, dtype=torch.uint8, device=DEVICE
    )
    cache = torch.as_strided(pool, contiguous_cache.shape, (pool.stride(0), WIDTH, 1))
    cache.copy_(contiguous_cache)
    offsets = [0, *itertools.accumulate(n // RATIO for n in seq_lens)]
    prefix = torch.tensor(offsets[:-1], device=DEVICE)
    natural = torch.zeros(offsets[-1], 1, WIDTH, dtype=torch.uint8, device=DEVICE)
    boundary = case.kv_slot >= 0
    natural_slots = torch.where(
        boundary, prefix[case.req.long()] + case.positions // RATIO, -1
    )
    k_norm_rope_mxfp4_cache(
        pooled,
        case.positions,
        case.cos_sin,
        case.norm,
        1e-6,
        natural,
        natural_slots,
        RATIO,
    )
    positions = torch.tensor(
        [p for n in seq_lens for p in range(n - next_n, n)],
        dtype=torch.int64,
        device=DEVICE,
    )
    rows = len(positions)
    (q, q_scale), weights = rocm_mxfp4_indexer_q_quant(
        positions,
        torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE),
        case.cos_sin,
        torch.rand(rows, HEADS, device=DEVICE),
        HEAD_DIM**-0.5,
        HEADS**-0.5,
    )
    q_ref = _dequant(q, q_scale.view(torch.uint8).reshape(rows, HEADS, -1))
    row_lens = ((positions + 1) // RATIO).to(torch.int32)
    context_lens = torch.tensor(
        [n // RATIO for n in seq_lens], dtype=torch.int32, device=DEVICE
    )
    width = max(cdiv(n // RATIO, PAGE) for n in seq_lens)
    block_table = torch.zeros(num_requests, width, dtype=torch.int32, device=DEVICE)
    for req in range(num_requests):
        slots = case.kv_slot[(case.req == req) & boundary]
        pages = (slots[::PAGE] // PAGE).to(torch.int32)
        block_table[req, : len(pages)] = pages
    row_table = block_table.repeat_interleave(next_n, dim=0)
    query_lens = [next_n] * num_requests
    native = native_decode(row_lens, row_table, query_lens, next_n, context_lens)
    assert native is not None
    max_context = max(seq_lens) // RATIO
    logits_width = logits_width or max_context
    schedule = torch.empty(
        rocm_mxfp4_decode_schedule_words(HEADS, HEAD_DIM, PAGE, next_n),
        dtype=torch.int32,
        device=DEVICE,
    )
    metadata = RocmMxfp4IndexerMetadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        max_seq_len=max(seq_lens),
        slot_mapping=torch.empty(0, dtype=torch.int64, device=DEVICE),
        num_decodes=num_requests,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=row_table,
            seq_lens=row_lens.reshape(num_requests, next_n),
            decode_lens=torch.full(
                (num_requests,), next_n, dtype=torch.int32, device=DEVICE
            ),
            requires_padding=False,
            schedule_metadata=torch.empty(0, dtype=torch.int32, device=DEVICE),
        ),
        positions=positions,
        block_table=block_table,
        decode_row_lens=row_lens,
        decode_native=native,
        decode_schedule=build_rocm_mxfp4_decode_schedule(
            row_lens, HEADS, HEAD_DIM, PAGE, schedule, logits_width, native
        ),
    )
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=DEVICE)
    init_workspace_manager(torch.device(DEVICE))
    vllm_config = VllmConfig()
    try:

        def run():
            with set_forward_context(
                {"indexer": metadata}, vllm_config, num_tokens=rows
            ):
                rocm_mxfp4_sparse_attn_indexer(
                    pooled[:rows],
                    "indexer",
                    cache,
                    q,
                    q_scale,
                    weights,
                    topk,
                    HEAD_DIM,
                    logits_width,
                    out,
                    compress_ratio=RATIO,
                )

        run()
        if topk == 1024 and num_requests in (8, 16) and context == 170000:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            weights.neg_()
            out.fill_(-1)
            graph.replay()
        lengths = row_lens.cpu().tolist()
        for row, end in enumerate(lengths):
            req = row // next_n
            keys = natural[offsets[req] : offsets[req] + end, 0]
            keys = _dequant(keys[:, : HEAD_DIM // 2], keys[:, HEAD_DIM // 2 :])
            _assert_topk(out[row], _scores(q_ref[row], weights[row], keys), topk)
    finally:
        reset_workspace_manager()
