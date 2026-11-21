# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4's ROCm MXFP4 indexer K writer: the fused pool + norm + RoPE +
MXFP4 kernel, against aiter's norm + RoPE + MXFP4 cache op fed by torch's
pooling.

The dense scoring path it feeds is covered by test_rocm_paged_mxfp4_indexer.py.
"""

import itertools
import types

import pytest
import torch
from torch import nn

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import get_cdna_version

if get_cdna_version() != 4:
    pytest.skip("CDNA4-only", allow_module_level=True)

from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4RocmMxfp4Compressor,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    rocm_mxfp4_indexer_k_store,
    rocm_paged_mxfp4_cache_layout,
)

# DeepSeek-V4-Pro's indexer: 64 heads of 128, ratio 4 with overlap, 256-token
# blocks, so 64-entry K pages.
HEADS, HEAD_DIM, RATIO, PAGE = 64, 128, 4, 64
WIDTH = HEAD_DIM // 2 + HEAD_DIM // 32
STATE_BLOCK, MAX_POS = 16, 4096
DEVICE = "cuda"


class _Case:
    """Requests with random compressor states and their boundary tokens'
    slots in a paged K cache."""

    def __init__(self, seq_lens):
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
        self.cos_sin = torch.randn(MAX_POS, 64, device=DEVICE)
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
        """The ROCm compressor's store step on this case; returns its K cache."""
        comp = DeepseekV4RocmMxfp4Compressor.__new__(DeepseekV4RocmMxfp4Compressor)
        nn.Module.__init__(comp)
        kv_cache = torch.zeros(
            self.num_pages, PAGE, WIDTH, dtype=torch.uint8, device=DEVICE
        )
        comp.head_dim, comp.compress_ratio, comp.overlap = HEAD_DIM, RATIO, True
        comp.rope_head_dim = 64
        comp.use_fp4_cache, comp.num_index_heads = True, HEADS
        comp._quant_block, comp._token_stride = 32, HEAD_DIM // 2
        comp._scale_dim = HEAD_DIM // 32
        comp.rms_norm_eps = 1e-6
        comp.norm = types.SimpleNamespace(weight=self.norm)
        comp.k_cache_prefix = "indexer.k_cache"
        comp._static_forward_context = {
            comp.k_cache_prefix: types.SimpleNamespace(kv_cache=kv_cache)
        }
        state_metadata = types.SimpleNamespace(
            slot_mapping=torch.zeros_like(self.positions),
            token_to_req_indices=self.req,
            block_table=self.block_table,
            block_size=STATE_BLOCK,
        )
        comp._compress_norm_rope_store(
            state_metadata,
            self.state_cache,
            self.state_width,
            self.positions,
            types.SimpleNamespace(cos_sin_cache=self.cos_sin),
            {comp.k_cache_prefix: types.SimpleNamespace(slot_mapping=self.kv_slot)},
            {},
        )
        return kv_cache


SEQ_LENS = pytest.mark.parametrize("seq_lens", [[37, 300, 5], [1, 260]])


@SEQ_LENS
def test_k_pages_match_aiter_fed_by_torch(seq_lens):
    """The pages hold what aiter writes from torch's pooling: the same layout,
    and the same MXFP4 codes up to bf16 rounding of the pooled rows."""
    set_random_seed(0)
    case = _Case(seq_lens)
    kv_cache = case.run_compressor()
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


def test_warmup_compiles_the_aiter_layout_variant(monkeypatch):
    """Startup warmup compiles the one variant the compressor launches."""
    from vllm.v1.attention.backends.mla import indexer

    monkeypatch.setattr(indexer, "dsa_indexer_uses_fp4", lambda _: True)
    hf_config = types.SimpleNamespace(
        index_head_dim=HEAD_DIM, qk_rope_head_dim=64, index_n_heads=HEADS
    )
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        cache_config=types.SimpleNamespace(
            block_size=PAGE * RATIO, cache_dtype="fp8_ds_mla"
        ),
    )
    kernel = _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL
    (key,) = kernel.get_warmup_keys(vllm_config)
    layout = rocm_paged_mxfp4_cache_layout(HEADS, HEAD_DIM, PAGE)
    assert (key.n_per_tile, key.d_per_tile, key.scale_lanes) == tuple(layout)
    kernel.compile(key)
