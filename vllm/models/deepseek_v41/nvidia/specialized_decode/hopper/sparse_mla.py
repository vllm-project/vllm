# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlashMLA sparse decode of a Hopper executor step.

It is the decode branch of ``DeepseekV4FlashMLAAttention.forward_mqa`` on
per-step tables that the layers share (``FlashMlaTables``):

- One FlashMLA tile-scheduler plan per layer type (compress ratio). The stock
  decode keeps one per type in each SWA KV-cache group's metadata, so every
  group plans each type again. A plan depends only on the batch shape and on
  the SWA and compressed top-k lengths, which every group derives alike from
  the batch (one window, one set of sequence lengths, one index top-k), so the
  first layer of a type plans for all of them.
- One global top-k table per ``sparse_index_key``. Compressed layers that share
  an index source map its top-k indices through the same compressed block
  table, so the first of them builds the table and the others reuse it. The
  map also reads a SWA group's token-to-request map and token validity, which
  every group derives alike from the batch.

Both are built on the forward stream and issue no collective. Steps of up to
``mla_small.MAX_ROWS`` uniform rows run the small-batch kernels instead
(``mla_small``), which need neither.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    get_mla_metadata,
)

from ...flashmla import DeepseekV4FlashMLAAttention
from ..attention.sparse_mla import sparse_index_key
from .mla_small import SmallMlaDecode

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class FlashMlaTables:
    """The tile-scheduler plans and global top-k tables of one decode step."""

    def __init__(
        self, attns: list[DeepseekV4FlashMLAAttention], small: bool = True
    ) -> None:
        self.keys = {attn: sparse_index_key(attn) for attn in attns}
        self._plans: dict[int, FlashMLASchedMeta] = {}
        self._tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.small = SmallMlaDecode(attns, attns[0].attn_sink.device) if small else None

    def reset(self) -> None:
        """Forget the previous step's plans and tables, and choose the step's
        path."""
        self._plans.clear()
        self._tables.clear()
        if self.small is not None:
            self.small.start()

    def join(self) -> None:
        """Wait for the small path's SWA unions; the FlashMLA tables are built
        on the forward stream."""
        if self.small is not None:
            self.small.join()

    def plan(self, attn: DeepseekV4FlashMLAAttention) -> FlashMLASchedMeta:
        """The step's plan of ``attn``'s layer type, planned by the first
        decode launch that takes it."""
        plan = self._plans.get(attn.compress_ratio)
        if plan is None:
            plan = self._plans[attn.compress_ratio] = get_mla_metadata()[0]
        return plan

    def global_topk(
        self,
        attn: DeepseekV4FlashMLAAttention,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``compute_global_topk_indices_and_lens`` of a compressed layer, on
        the step's shared tables."""
        key = self.keys[attn]
        table = self._tables.get(key)
        if table is None:
            num_tokens = swa_metadata.num_decode_tokens
            assert attn.topk_indices_buffer is not None
            assert swa_metadata.is_valid_token is not None
            table = compute_global_topk_indices_and_lens(
                attn.topk_indices_buffer[:num_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[: swa_metadata.num_decodes],
                attn_metadata.block_size // attn.compress_ratio,
                swa_metadata.is_valid_token[:num_tokens],
                launch_pdl=True,
            )
            # A key's layers are consecutive, and the next index source
            # overwrites the top-k indices: drop the previous key's table.
            self._tables = {key: table}
        return table


class FlashMlaDecode:
    """``forward_mqa`` of a pure-decode step on the shared tables."""

    def __init__(self, tables: FlashMlaTables) -> None:
        self.tables = tables

    def chain_step(self, rows: int) -> bool:
        """Whether this step's MLA runs the small path, whose combine can also
        apply the inverse RoPE (``run_rotated``)."""
        from . import attn_chain

        small = self.tables.small
        return small is not None and small.query_len > 0 and rows <= attn_chain.MAX_ROWS

    def run_rotated(
        self,
        attn: DeepseekV4FlashMLAAttention,
        q: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """``attn``'s sparse MLA of ``q`` on the small path, inverse-rotated:
        the ``wo_a`` input [rows, groups, heads per group * 512]."""
        by_layer = get_forward_context().attn_metadata
        assert isinstance(by_layer, dict)
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", by_layer[attn.swa_cache_layer.prefix]
        )
        attn_metadata = None
        if attn.compress_ratio != 0:
            assert attn.compressed_cache_prefix is not None
            attn_metadata = cast(
                DeepseekV4FlashMLAMetadata, by_layer[attn.compressed_cache_prefix]
            )
        small = self.tables.small
        assert small is not None
        rotated = small.run(attn, q, None, swa_metadata, attn_metadata, positions)
        assert rotated is not None
        return rotated

    def run(
        self, attn: DeepseekV4FlashMLAAttention, q: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Write ``attn``'s sparse MLA of ``q`` (padded heads) into ``out``."""
        by_layer = get_forward_context().attn_metadata
        assert isinstance(by_layer, dict)
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", by_layer[attn.swa_cache_layer.prefix]
        )
        assert swa_metadata.num_prefills == 0
        num_tokens = swa_metadata.num_decode_tokens
        if num_tokens == 0:
            return
        attn_metadata = None
        if attn.compress_ratio != 0:
            assert attn.compressed_cache_prefix is not None
            attn_metadata = cast(
                DeepseekV4FlashMLAMetadata, by_layer[attn.compressed_cache_prefix]
            )
        small = self.tables.small
        if small is not None and small.query_len:
            small.run(attn, q, out, swa_metadata, attn_metadata)
            return
        kv_cache = indices = lens = None
        if attn_metadata is not None:
            global_indices, lens = self.tables.global_topk(
                attn, swa_metadata, attn_metadata
            )
            indices = global_indices.view(num_tokens, 1, -1)
            kv_cache = attn._compressed_kv_cache().unsqueeze(-2)
        flash_mla_with_kvcache(
            q=q[:num_tokens].unsqueeze(1),
            k_cache=attn.swa_cache_layer.kv_cache.unsqueeze(-2),
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=self.tables.plan(attn),
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_metadata.decode_swa_indices,
            topk_length=swa_metadata.decode_swa_lens,
            softmax_scale=attn.scale,
            attn_sink=attn.attn_sink,
            extra_k_cache=kv_cache,
            extra_indices_in_kvcache=indices,
            extra_topk_length=lens,
            out=out[:num_tokens].unsqueeze(1),
        )
