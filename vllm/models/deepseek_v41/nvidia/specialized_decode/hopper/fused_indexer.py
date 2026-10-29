# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused FP8 decode indexer of the Hopper executor, for uniform decode
steps of up to ``MAX_ROWS`` rows.

It replaces the stock decode branch of ``sparse_attn_indexer`` (DeepGEMM
paged MQA logits, the candidate select or mask, the top-k and the ``-1``
fill) on the stock indexer K cache:

- ``IdxLogits`` (``indexer_cute_logits``): FP8 WGMMA logits of all verify rows
  of a request per index-K tile, masked to the step's candidate bitmap on a
  candidate consumer, ``-inf`` past each row's length;
- on the candidate source, ``IdxSelect2`` (``indexer_cute_select``) writes the
  candidate blocks (the stock list, for steps on the stock path) and the
  bitmap the consumers read;
- the stock ``cooperative_topk`` selects the top-k, writing the whole ``-1``
  padded row.

The candidate bitmap is per step: the source writes it before its consumers
read it, in the same graph.
"""

from functools import cache
from typing import Any

import torch

from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

MAX_ROWS = 48
_TOP_K, _HEADS, _HEAD_DIM, _CANDIDATE_BLOCK = 512, 32, 128, 8
_TILE = 64


@cache
def _compiled(kind: str, *flags: Any):
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    i, p = Int32(1), Int64(0)
    if kind == "logits":
        from .indexer_cute_logits import IdxLogits

        rows_per_request, tiles_per_cta, masked = flags
        kernel: Any = IdxLogits(rows_per_request, tiles_per_cta, 3, masked)
        args = [p, p, p, p, i, p, i, p, p, i, p, i, i, i]
    else:
        from .indexer_cute_select import IdxSelect2

        cand_k, width = flags
        kernel = IdxSelect2(cand_k, width)
        args = [p, i, p, p, i, p, i, i]
    return cute.compile(kernel, *args, make_fake_stream())


def _launch(kind: str, flags: tuple, *args: Any) -> None:
    from cutlass import Int32, Int64

    from ...ops.cute_dsl.primitives import current_cu_stream

    values: list = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            values.append(Int64(arg.data_ptr()))
        elif isinstance(arg, tuple):
            values.append(Int64(arg[1]))
        else:
            values.append(Int32(arg))
    _compiled(kind, *flags)(*values, current_cu_stream())


def applies(op: SparseAttnIndexer) -> bool:
    """Whether the kernels take the indexer: an FP8 K cache, no context
    parallelism, the 512-wide top-k and 8-position candidate blocks."""
    if not isinstance(op, SparseAttnIndexer):
        return False
    candidates = getattr(op, "candidate_blocks", None)
    return (
        not op.use_fp4_cache
        and op.dcp_world_size == 1
        and not op.use_pcp
        and op.topk_tokens == _TOP_K
        and op.head_dim == _HEAD_DIM
        and (candidates is None or op.candidate_block_size == _CANDIDATE_BLOCK)
    )


class FusedDecodeIndexer:
    """The buffers of the fused indexer, shared by every layer of a step."""

    def __init__(self, max_model_len: int, device: torch.device) -> None:
        from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE

        self.width = -(-max_model_len // _TILE) * _TILE
        self.logits = torch.empty(
            MAX_ROWS, self.width, dtype=torch.float32, device=device
        )
        blocks = -(-max_model_len // _CANDIDATE_BLOCK)
        self.cand_bits = torch.zeros(
            MAX_ROWS, -(-blocks // 32), dtype=torch.int32, device=device
        )
        self.workspace = torch.empty(
            RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        self.empty = torch.zeros(4, dtype=torch.int32, device=device)

    @classmethod
    def for_ops(
        cls, ops: list[SparseAttnIndexer], device: torch.device
    ) -> "FusedDecodeIndexer":
        """The buffers shared by ``ops``: ratio-1 and ratio-2 indexers have
        different row lengths, so they are sized for the longest."""
        return cls(max(op.max_model_len for op in ops), device)

    def warm_up(self, query_len: int, cand_k: int) -> None:
        """Compile the kernels of every request count before graph capture."""
        for masked in (False, True):
            _compiled("logits", query_len, 1, masked)
        _compiled("select", cand_k, self.width)

    def serves(self, metadata: Any, query_len: int) -> bool:
        """Whether a step's indexer ``metadata`` is a uniform decode the
        kernels take, ``query_len`` rows per request, whose rows fit the
        logits buffer (host fields only)."""
        decode = getattr(metadata, "decode", None)
        rows = getattr(metadata, "num_decode_tokens", 0)
        return (
            decode is not None
            and metadata.num_prefills == 0
            and metadata.max_seq_len <= self.width
            and decode.indices is None
            and decode.global_seq_lens is None
            and 0 < query_len <= 8
            and 0 < rows <= MAX_ROWS
            and rows % query_len == 0
        )

    def __call__(
        self,
        op: SparseAttnIndexer,
        q: torch.Tensor,
        weights: torch.Tensor,
        metadata: Any,
        query_len: int,
    ) -> None:
        """The step's top-k indices of ``op`` into its ``topk_indices_buffer``
        (and on its candidate source, the candidate blocks and bitmap), for
        rows of ``query_len`` per request."""
        decode = metadata.decode
        rows = metadata.num_decode_tokens
        requests = rows // query_len
        assert q.dtype == torch.float8_e4m3fn and q.shape[1:] == (_HEADS, _HEAD_DIM)
        k_cache = op.k_cache.kv_cache
        page = k_cache.shape[1]
        seq_lens = decode.seq_lens.reshape(-1)[:rows]
        block_table = decode.block_table
        # About one CTA per SM over the requests; each strides over its
        # request's tiles up to that request's length.
        tiles_per_cta = 1
        chunks = max(1, 132 // requests)
        candidates = op.candidate_blocks
        masked = candidates is not None and not op.candidate_write
        _launch(
            "logits",
            (query_len, tiles_per_cta, masked),
            q,
            weights,
            k_cache,
            ("i64", k_cache.stride(0)),
            page.bit_length() - 1,
            block_table,
            block_table.stride(0),
            seq_lens,
            self.logits,
            self.width,
            self.cand_bits if masked else self.empty,
            self.cand_bits.stride(0) if masked else 0,
            chunks,
            requests,
        )
        if candidates is not None and op.candidate_write:
            blocks = candidates[:rows]
            _launch(
                "select",
                (blocks.shape[1], self.width),
                self.logits,
                self.width,
                seq_lens,
                blocks,
                blocks.stride(0),
                self.cand_bits,
                self.cand_bits.stride(0),
                rows,
            )
        torch.ops._C.cooperative_topk(
            self.logits[:rows],
            seq_lens.view(-1, 1),
            op.topk_indices_buffer[:rows, :_TOP_K],
            self.workspace,
            _TOP_K,
            metadata.max_seq_len,
        )
