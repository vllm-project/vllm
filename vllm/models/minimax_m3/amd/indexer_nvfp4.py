# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 lightning indexer over NVFP4 index-K pages on ROCm (gfx950).

Keys score as ``e4m3_rn_satfinite(e2m1 * scale)`` with AITER's init / local
sentinels; the top-k writes logical block ids for the NVFP4 attend, no page
table. With ``indexer_decode_cp`` at TP4, decode steps of a multiple of 4 rows
per request (DSpark 3 or 7) and at most 256 4-row requests score
context-parallel (``ops/index_cp``).
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_msa import (
    MSA_INDEX_HEAD_DIM,
    MSA_SCORE_TYPE,
    MSA_SPARSE_BLOCK_SIZE,
    MSA_TOPK_BLOCKS,
    MiniMaxM3IndexerMSABackend,
    MiniMaxM3IndexerMSAMetadata,
    _score_width,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
)
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_enabled

logger = init_logger(__name__)

# Index heads per rank (TP4 / TP2 / TP1).
NUM_INDEX_HEADS = (1, 2, 4)
# Decode CP: rows per request (DSpark 3) and the request cap.
CP_QUERY_LEN = 4
CP_MAX_REQS = 256


def _cp_requests(
    d: MiniMaxM3IndexerDecodeMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The CP scorer's 4-row requests: a request of k * 4 rows scores as k
    requests on its block table row, each 4 tokens shorter than the next, so
    every row keeps its causal length. Built once a step on the shared
    metadata."""
    k = d.decode_query_len // CP_QUERY_LEN
    if k == 1:
        return d.block_table, d.seq_lens
    split = getattr(d, "_cp_split", None)
    if split is None:
        back = torch.arange(k - 1, -1, -1, device=d.seq_lens.device) * CP_QUERY_LEN
        seq_lens = (d.seq_lens[:, None] - back.to(d.seq_lens.dtype)).clamp_(min=0)
        split = (d.block_table.repeat_interleave(k, dim=0), seq_lens.flatten())
        d._cp_split = split  # type: ignore[attr-defined]
    return split


def rocm_nvfp4_indexer_unsupported_reason(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    num_index_heads: int,
    index_head_dim: int,
    score_type: str,
) -> str | None:
    """Why the ROCm NVFP4 indexer cannot serve this config, None when it can."""
    if not current_platform.is_rocm():
        return "needs ROCm"
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "needs gfx950"
    if (topk_blocks, sparse_block_size, index_head_dim, score_type) != (
        MSA_TOPK_BLOCKS,
        MSA_SPARSE_BLOCK_SIZE,
        MSA_INDEX_HEAD_DIM,
        MSA_SCORE_TYPE,
    ):
        return (
            f"needs topk_blocks={MSA_TOPK_BLOCKS}, block_size="
            f"{MSA_SPARSE_BLOCK_SIZE}, index_head_dim={MSA_INDEX_HEAD_DIM} and "
            f"score_type={MSA_SCORE_TYPE!r}"
        )
    if num_index_heads not in NUM_INDEX_HEADS:
        return f"needs num_index_heads in {NUM_INDEX_HEADS}, got {num_index_heads}"
    return None


def init_rocm_nvfp4_indexer_decode_cp(max_model_len: int) -> bool:
    """Set up decode CP if configured (a TP collective); returns whether it is on."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    from .ops.index_cp import host as index_cp

    if not get_current_vllm_config().attention_config.indexer_decode_cp:
        return False
    if get_tensor_model_parallel_world_size() != index_cp.WORLD:
        logger.warning_once(
            "indexer_decode_cp needs TP %d on ROCm; scoring decode locally.",
            index_cp.WORLD,
        )
        return False
    index_cp.init(
        torch.device("cuda", torch.accelerator.current_device_index()),
        _score_width(max_model_len, MSA_SPARSE_BLOCK_SIZE),
    )
    if index_cp.get() is None:
        return False
    logger.info_once("MiniMax M3 NVFP4 indexer: decode context parallelism on (TP4)")
    return True


class MiniMaxM3IndexerRocmNvfp4Impl(MiniMaxM3IndexerImpl):
    """FlyDSL NVFP4 index scores + top-k block ids for prefill and decode."""

    # Its builder materializes the prefill rows' block counts.
    indexer_backend_cls = MiniMaxM3IndexerMSABackend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.indexer_kv_dtype != "nvfp4":
            raise ValueError(
                f"{type(self).__name__} requires indexer_kv_dtype='nvfp4', got "
                f"{self.indexer_kv_dtype!r}"
            )
        reason = rocm_nvfp4_indexer_unsupported_reason(
            topk_blocks=self.topk_blocks,
            sparse_block_size=self.block_size,
            num_index_heads=self.num_index_heads,
            index_head_dim=self.index_head_dim,
            score_type=self.score_type,
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 NVFP4 indexer on ROCm {reason}")

    def _new_score(self, rows: int, max_seq_len: int) -> torch.Tensor:
        """Uninitialized: the scorer writes every block the top-k reads."""
        return torch.empty(
            (
                self.num_index_heads,
                rows,
                _score_width(max_seq_len, self.block_size),
            ),
            dtype=torch.float32,
            device=self.index_cache.kv_cache.device,
        )

    def _decode_cp(self, num_decodes: int, query_len: int):
        from .ops.index_cp import host as index_cp

        cp = index_cp.get()
        if (
            cp is None
            # The push has no back-pressure; under DBO no all-reduce orders it.
            or dbo_enabled()
            or self.num_index_heads != 1
            or query_len % CP_QUERY_LEN
            or num_decodes * query_len // CP_QUERY_LEN > CP_MAX_REQS
            or num_decodes * query_len > cp.max_rows
        ):
            return None
        return cp

    def forward(
        self,
        index_query: torch.Tensor,
        **_aiter_table_args,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from .ops.index_topk_decode.host import (
            minimax_m3_index_topk_decode_indices,
            minimax_m3_index_topk_prefill_indices,
        )
        from .ops.nvfp4_index_decode.host import minimax_m3_nvfp4_index_score_decode
        from .ops.nvfp4_index_prefill.host import (
            minimax_m3_nvfp4_index_score_prefill,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMSAMetadata)
        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache
        buf = self.topk_indices_buffer
        if buf is None:
            buf = torch.empty(
                (self.num_index_heads, num_tokens, self.topk_blocks),
                dtype=torch.int32,
                device=iq.device,
            )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            cp = self._decode_cp(md.num_decodes, d.decode_query_len)
            if cp is not None:
                block_table, seq_lens = _cp_requests(d)
                # A view of the receive buffer, valid until the next call.
                score = cp.score_push(
                    iq[:nd, 0],
                    kv,
                    block_table,
                    seq_lens,
                    self.init_blocks,
                    self.local_blocks,
                    _score_width(d.max_seq_len, self.block_size),
                )
            else:
                score = self._new_score(nd, d.max_seq_len)
                minimax_m3_nvfp4_index_score_decode(
                    iq[:nd],
                    kv,
                    score,
                    d.block_table,
                    d.seq_lens,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    query_len=d.decode_query_len,
                    max_seq_len=d.max_seq_len,
                )
            decode_topk = buf[:, :nd, :]
            minimax_m3_index_topk_decode_indices(
                score,
                decode_topk,
                d.seq_lens,
                max_seq_len=d.max_seq_len,
                query_len=d.decode_query_len,
            )

        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None and md.prefill_num_valid_pages is not None
            score = self._new_score(num_tokens - nd, p.max_seq_len)
            minimax_m3_nvfp4_index_score_prefill(
                iq[nd:],
                kv,
                score,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                max_query_len=p.max_query_len,
                max_seq_len=p.max_seq_len,
            )
            prefill_topk = buf[:, nd:num_tokens, :]
            minimax_m3_index_topk_prefill_indices(
                score,
                prefill_topk,
                max_seq_len=p.max_seq_len,
                num_valid_pages=md.prefill_num_valid_pages,
            )
        return decode_topk, prefill_topk
