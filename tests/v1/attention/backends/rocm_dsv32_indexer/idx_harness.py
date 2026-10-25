# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive the DeepSeek V3.2 sparse indexer on ROCm the way the runner does.

``IndexerCase`` owns a VllmConfig shaped like one DSA model, the ROCm indexer
K cache layer (``DeepseekV32ROCmIndexerCache``) bound to a cache allocated
through the runner's view helpers, ``DeepseekV32IndexerMetadataBuilder`` and
the ``SparseAttnIndexer`` op. ``run`` writes the batch's K rows through the
op and returns the top-k buffer; ``logits`` recomputes every query's scores
in float32 from the quantized operands.
"""

import math
from dataclasses import dataclass, field

import torch

from tests.v1.attention.backends.harness import PagedBatch, make_paged_batch
from tests.v1.attention.backends.indexer_harness import (
    quantize_indexer_k,
    ref_indexer_logits,
)
from tests.v1.attention.backends.mla_harness import token_positions
from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    make_config,
)
from tests.v1.attention.model_shapes import MLAShape
from tests.v1.attention.utils import dense_kv_cache_views
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import SpeculativeConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v32.amd.rocm import DeepseekV32ROCmIndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_ops
from vllm.v1.kv_cache_interface import KVCacheLayout

DEVICE = torch.device("cuda")
QUANT_BLOCK = 128
PREFIX = "model.layers.0.self_attn.indexer.k_cache"

# Kernel labels.
K_WRITE = "indexer_k_quant_and_cache_triton"
K_GATHER = "cp_gather_indexer_k_quant_cache_triton"
PREFILL_LOGITS = "rocm_fp8_mqa_logits"
DECODE_LOGITS = "rocm_fp8_paged_mqa_logits"
AITER_TOPK_PREFILL = "_launch_aiter_top_k_per_row_prefill"
AITER_TOPK_DECODE = "indexer_top_k_decode"
TOPK_PREFILL = "top_k_per_row_prefill"
TOPK_DECODE = "top_k_per_row_decode"
PACK = "pack_seq_triton"


def watch_all_kernels(spy) -> None:
    spy.watch(sparse_ops, "indexer_k_quant_and_cache_triton", K_WRITE)
    spy.watch(sparse_ops, "cp_gather_indexer_k_quant_cache_triton", K_GATHER)
    spy.watch(sparse_ops, "rocm_fp8_mqa_logits", PREFILL_LOGITS)
    spy.watch(sparse_ops, "rocm_fp8_paged_mqa_logits", DECODE_LOGITS)
    spy.watch(sparse_ops, "_launch_aiter_top_k_per_row_prefill", AITER_TOPK_PREFILL)
    spy.watch(rocm_aiter_ops, "indexer_top_k_decode", AITER_TOPK_DECODE)
    spy.watch(torch.ops._C, "top_k_per_row_prefill", TOPK_PREFILL)
    spy.watch(torch.ops._C, "top_k_per_row_decode", TOPK_DECODE)
    spy.watch(sparse_ops, "pack_seq_triton", PACK)


@dataclass
class IndexerCase:
    shape: MLAShape
    block_size: int = 64
    topk: int = 256
    spec_tokens: int = 0
    max_model_len: int = 16384
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    num_blocks: int = 0
    seed: int = 0

    config: VllmConfig = field(init=False)
    k_cache: DeepseekV32ROCmIndexerCache = field(init=False)
    builder: DeepseekV32IndexerMetadataBuilder = field(init=False)
    op: SparseAttnIndexer = field(init=False)
    topk_buffer: torch.Tensor = field(init=False)
    k_fp8: torch.Tensor = field(init=False)
    k_scale: torch.Tensor = field(init=False)
    k_rows: torch.Tensor = field(init=False)

    @property
    def head_dim(self) -> int:
        return self.shape.index_head_dim or 128

    @property
    def block_table_width(self) -> int:
        return math.ceil(self.max_model_len / self.block_size)

    @property
    def num_heads(self) -> int:
        return self.shape.index_n_heads or 64

    def setup(self, num_blocks: int) -> "IndexerCase":
        torch.manual_seed(self.seed)
        self.num_blocks = num_blocks
        self.config = make_config(
            self.shape,
            topk=self.topk,
            block_size=self.block_size,
            num_blocks=num_blocks,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        if self.spec_tokens:
            self.config.speculative_config = SpeculativeConfig(
                method="ngram", num_speculative_tokens=self.spec_tokens
            )
        self.topk_buffer = torch.full(
            (self.max_num_batched_tokens, self.topk),
            -1,
            dtype=torch.int32,
            device=DEVICE,
        )
        with set_current_vllm_config(self.config), torch.device(DEVICE):
            self.k_cache = DeepseekV32ROCmIndexerCache(
                head_dim=self.head_dim + self.head_dim // QUANT_BLOCK * 4,
                dtype=torch.uint8,
                prefix=PREFIX,
                cache_config=self.config.cache_config,
            )
            spec = DeepseekV32IndexerBackend.customize_spec(
                self.k_cache.get_kv_cache_spec(self.config)
            )
            self.op = SparseAttnIndexer(
                self.k_cache,
                QUANT_BLOCK,
                "ue8m0",
                self.topk,
                self.head_dim,
                self.max_model_len,
                get_max_prefill_buffer_size(self.config),
                self.topk_buffer,
            )
            self.builder = DeepseekV32IndexerMetadataBuilder(
                spec,
                [PREFIX],
                self.config,
                DEVICE,
                block_table_width=self.block_table_width,
            )
        raw = torch.zeros(
            num_blocks * spec.page_size_bytes, dtype=torch.int8, device=DEVICE
        )
        (view,) = dense_kv_cache_views(
            raw, spec, num_blocks, num_layers=1, layout=KVCacheLayout.LBNHC
        )
        self.k_cache.bind_kv_cache(view)
        self._fill_cache()
        return self

    def _fill_cache(self) -> None:
        """Write a random K row to every slot through the cache writer."""
        num_slots = self.num_blocks * self.block_size
        self.k_rows = torch.randn(
            num_slots, self.head_dim, dtype=torch.bfloat16, device=DEVICE
        )
        self.k_fp8, self.k_scale = quantize_indexer_k(self.k_rows)
        sparse_ops.indexer_k_quant_and_cache_triton(
            self.k_rows,
            self.k_cache.kv_cache,
            torch.arange(num_slots, dtype=torch.int64, device=DEVICE),
            QUANT_BLOCK,
            "ue8m0",
        )

    def batch(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        *,
        num_padded_tokens: int | None = None,
    ) -> PagedBatch:
        batch = make_paged_batch(
            query_lens,
            seq_lens,
            self.block_size,
            DEVICE,
            num_blocks=self.num_blocks,
            num_padded_tokens=num_padded_tokens,
            seed=self.seed,
        )
        # The runner's block table is always block_table_width wide.
        table = batch.common.block_table_tensor
        wide = table.new_zeros(table.shape[0], self.block_table_width)
        wide[:, : table.shape[1]] = table
        batch.common = batch.common.replace(block_table_tensor=wide)
        return batch

    def build(self, batch: PagedBatch) -> DeepseekV32IndexerMetadata:
        return self.builder.build(0, batch.common)

    def inputs(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(q_fp8 [T, H, D], weights [T, H] float32)."""
        fp8 = current_platform.fp8_dtype()
        q = torch.randn(num_tokens, self.num_heads, self.head_dim, device=DEVICE)
        weights = torch.randn(num_tokens, self.num_heads, device=DEVICE) * 0.1
        return q.to(fp8), weights

    def run(
        self,
        batch: PagedBatch,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: DeepseekV32IndexerMetadata | None = None,
    ) -> torch.Tensor:
        """Run the indexer; returns the top-k rows of the batch's tokens."""
        metadata = metadata if metadata is not None else self.build(batch)
        n = q_fp8.shape[0]
        k = self.k_rows[batch.query_slots]
        if k.shape[0] < n:
            k = torch.cat([k, k.new_zeros(n - k.shape[0], k.shape[1])])
        hidden = torch.empty(n, 1, device=DEVICE)
        self.topk_buffer.fill_(-7)  # the op must overwrite every row it owns
        with set_forward_context({PREFIX: metadata}, self.config):
            self.op.forward_hip(hidden, q_fp8, k, weights)
        return self.topk_buffer[: batch.num_tokens].clone()

    def logits(
        self, batch: PagedBatch, q_fp8: torch.Tensor, weights: torch.Tensor
    ) -> list[torch.Tensor]:
        """Per query token, float32 scores of its visible positions."""
        out = []
        token = 0
        for req, (q_len, seq_len) in enumerate(zip(batch.query_lens, batch.seq_lens)):
            slots = batch.slots[req]
            scores = ref_indexer_logits(
                q_fp8[token : token + q_len],
                self.k_fp8[slots],
                self.k_scale[slots],
                weights[token : token + q_len],
            )
            for i in range(q_len):
                pos = seq_len - q_len + i
                out.append(scores[i, : pos + 1])
            token += q_len
        assert len(out) == len(token_positions(batch))
        return out


def blocks_for(seq_lens: list[int], block_size: int, slack: int = 8) -> int:
    return sum(-(-s // block_size) for s in seq_lens) + slack


def make_idx_case(shape: MLAShape, seq_lens: list[int], **kw) -> IndexerCase:
    case = IndexerCase(shape, **kw)
    return case.setup(blocks_for(seq_lens, case.block_size))
