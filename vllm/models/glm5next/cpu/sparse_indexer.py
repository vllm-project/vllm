# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first CPU implementation of the GLM5Next KeyPool indexer.

This module intentionally uses eager PyTorch operations.  It is the CPU
execution boundary for GLM's pool compression and selection semantics; native
operators can replace the individual helpers once this path is validated.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for GLM's FWHT-128 plus UE8M0 FP8 quantization."""
    if q.ndim != 2 or q.shape[-1] != 128:
        raise ValueError(f"expected [rows, 128], got {tuple(q.shape)}")
    x = q.float()
    width = 1
    while width < 128:
        y = x.reshape(-1, 128 // (2 * width), 2, width)
        a, b = y.unbind(dim=2)
        x = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        width *= 2
    x = x * (128.0**-0.5)
    x = x.to(torch.bfloat16).float()
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    quant = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quant, scale


def _pool_compress(
    keys: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Return one BF16 pool vector for each ``[pool, head_dim]`` group."""
    scores = (gate + ape).float()
    probs = torch.softmax(scores, dim=0)
    pooled = (keys.float() * probs).sum(dim=0)
    return pooled.to(keys.dtype)


def _hadamard128(x: torch.Tensor) -> torch.Tensor:
    y = x.float()
    width = 1
    while width < 128:
        z = y.reshape(-1, 128 // (2 * width), 2, width)
        a, b = z.unbind(dim=2)
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
        width *= 2
    return y * (128.0**-0.5)


def _quantize_cache_vector(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = _hadamard128(x.reshape(1, 128)).reshape(128)
    x = x.to(torch.bfloat16).float()
    absmax = x.abs().amax().clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    values = (x / scale).clamp(-448.0, 448.0)
    return values.to(torch.float8_e4m3fn).view(torch.uint8), scale


def _dequantize_cache_vector(row: torch.Tensor) -> torch.Tensor:
    values = row[:128].view(torch.float8_e4m3fn).float().reshape(4, 32)
    scale = row[128:132].view(torch.float32)
    return (values.reshape(128) * scale).reshape(128)


def _weighted_indexer_score(
    key: torch.Tensor, query: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Compute one weighted multi-head indexer logit."""
    per_head = (query.float() * key.float()).sum(dim=-1)
    return (per_head.relu() * weights.float().reshape(-1)).sum()


def _expand_pool_ids(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Expand pool indices to causal token indices and append the tail."""
    rows: list[torch.Tensor] = []
    width = max_tokens or (pool_ids.shape[-1] * pool_size + max(pool_size - 1, 0))
    for ids, seq_len in zip(pool_ids.tolist(), seq_lens.tolist()):
        values: list[int] = []
        for pool in ids:
            if pool < 0:
                continue
            values.extend(range(pool * pool_size, (pool + 1) * pool_size))
        tail_start = (int(seq_len) // pool_size) * pool_size
        values.extend(range(tail_start, int(seq_len)))
        values = [value for value in values if value < int(seq_len)]
        values = values[:width]
        rows.append(
            torch.tensor(
                values + [-1] * (width - len(values)),
                dtype=pool_ids.dtype,
                device=pool_ids.device,
            )
        )
    if not rows:
        return pool_ids.new_empty((0, width))
    return torch.stack(rows)


class SparseAttnIndexerKpool(nn.Module):
    """CPU GLM KeyPool indexer.

    The class deliberately keeps the same callable contract as the CUDA/ROCm
    ``SparseAttnIndexerKpool``.  Cache writes and score computation are eager;
    unsupported metadata variants fail explicitly instead of falling through
    to a GPU implementation.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ) -> None:
        super().__init__()
        if use_fp4_cache:
            raise NotImplementedError("GLM5Next CPU indexer does not support FP4")
        if head_dim != 128:
            raise NotImplementedError(
                "GLM5Next CPU indexer currently requires index_head_dim=128"
            )
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.index_kpool = 1
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_quant.device.type != "cpu":
            raise RuntimeError("GLM5Next CPU indexer received a non-CPU tensor")
        if gate_score is None or compress_ape is None:
            raise NotImplementedError("GLM5Next CPU indexer requires KeyPool inputs")
        if index_kpool <= 0:
            raise ValueError("index_kpool must be positive")
        self.index_kpool = index_kpool
        if positions is None:
            positions = torch.arange(k.shape[0], device=k.device)

        context = get_forward_context().attn_metadata
        if not isinstance(context, dict):
            return self.topk_indices_buffer
        metadata = context.get(self.k_cache.prefix)
        if not isinstance(metadata, DeepseekV32IndexerMetadata):
            raise RuntimeError("GLM5Next CPU indexer requires indexer metadata")

        slot_mapping = metadata.slot_mapping[: k.shape[0]]
        if not self.skip_k_cache_insert:
            self._write_pools(
                k,
                gate_score,
                compress_ape,
                slot_mapping,
                index_kpool,
                positions,
                context,
            )

        self.topk_indices_buffer[: hidden_states.shape[0]].fill_(-1)
        if metadata.num_prefills:
            self._prefill_topk(
                q_quant,
                weights,
                metadata,
                index_kpool,
                positions,
            )
        if metadata.num_decodes:
            self._decode_topk(q_quant, weights, metadata, index_kpool, positions)
        return self.topk_indices_buffer

    def _write_pools(
        self,
        keys: torch.Tensor,
        gate: torch.Tensor,
        ape: torch.Tensor,
        slot_mapping: torch.Tensor,
        pool_size: int,
        positions: torch.Tensor,
        context: dict,
    ) -> None:
        cache = self.k_cache.kv_cache
        flat_cache = cache.reshape(cache.shape[0], cache.shape[1], -1)
        for end in range(keys.shape[0]):
            position = int(positions[end])
            if position < pool_size - 1 or position % pool_size != pool_size - 1:
                continue
            start = end - pool_size + 1
            expected = torch.arange(
                position - pool_size + 1, position + 1, device=positions.device
            )
            if not torch.equal(positions[start : end + 1], expected):
                continue
            slot = int(slot_mapping[end])
            if slot < 0:
                continue
            pooled = _pool_compress(
                keys[start : end + 1],
                gate[start : end + 1],
                ape,
            )
            self._store_index_cache(flat_cache, slot, pooled)
        self._write_tail(keys, gate, ape, positions, context)

    def _store_index_cache(
        self, flat_cache: torch.Tensor, slot: int, pooled: torch.Tensor
    ) -> None:
        block, offset = divmod(slot, flat_cache.shape[1])
        if block >= flat_cache.shape[0]:
            raise IndexError(f"index cache slot {slot} is out of bounds")
        values, scales = _quantize_cache_vector(pooled)
        flat_cache[block, offset, :128].copy_(values)
        flat_cache[block, offset, 128:132].copy_(scales.reshape(1).view(torch.uint8))

    def _tail_rows(self) -> tuple[torch.Tensor, int] | None:
        if self.tail_cache is None:
            return None
        cache = self.tail_cache.kv_cache
        if cache.ndim != 4:
            raise ValueError(
                "GLM5Next CPU tail cache must be a 4-D paged cache, "
                f"got {tuple(cache.shape)}"
            )
        if cache.shape[1] == 2:
            return cache, 1
        if cache.shape[2] == 2:
            return cache, 2
        raise ValueError(
            "GLM5Next CPU tail cache must expose two K/gate heads, "
            f"got {tuple(cache.shape)}"
        )

    def _tail_read(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        result = self._tail_rows()
        if result is None:
            raise RuntimeError("GLM5Next CPU tail cache is not configured")
        cache, head_axis = result
        block_size = cache.shape[2] if head_axis == 1 else cache.shape[1]
        block, offset = divmod(slot, block_size)
        if head_axis == 1:
            return cache[block, 0, offset], cache[block, 1, offset]
        return cache[block, offset, 0], cache[block, offset, 1]

    def _tail_write(self, slot: int, key: torch.Tensor, gate: torch.Tensor) -> None:
        result = self._tail_rows()
        if result is None:
            return
        cache, head_axis = result
        block_size = cache.shape[2] if head_axis == 1 else cache.shape[1]
        block, offset = divmod(slot, block_size)
        if block >= cache.shape[0] or offset >= block_size:
            raise IndexError(f"tail cache slot {slot} is out of bounds")
        if head_axis == 1:
            cache[block, 0, offset].copy_(key)
            cache[block, 1, offset].copy_(gate)
        else:
            cache[block, offset, 0].copy_(key)
            cache[block, offset, 1].copy_(gate)

    def _write_tail(
        self,
        keys: torch.Tensor,
        gate: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        context: dict,
    ) -> None:
        if self.tail_cache is None:
            return
        tail_meta = context.get(self.tail_cache.prefix)
        if tail_meta is None:
            return
        tail_slots = tail_meta.slot_mapping[: keys.shape[0]]
        main_meta = context.get(self.k_cache.prefix)
        flat_cache = self.k_cache.kv_cache.reshape(
            self.k_cache.kv_cache.shape[0], self.k_cache.kv_cache.shape[1], -1
        )
        for row, position in enumerate(positions.tolist()):
            slot = int(tail_slots[row])
            if slot < 0 or position < 0:
                continue
            self._tail_write(slot, keys[row], gate[row])
            if position % self.index_kpool != self.index_kpool - 1:
                continue
            if main_meta is None:
                continue
            main_slot = int(main_meta.slot_mapping[row])
            if main_slot < 0:
                continue
            rows = [
                self._tail_read(slot - (self.index_kpool - 1) + i)
                for i in range(self.index_kpool)
            ]
            pooled = _pool_compress(
                torch.stack([item[0] for item in rows]),
                torch.stack([item[1] for item in rows]),
                ape,
            )
            self._store_index_cache(flat_cache, main_slot, pooled)

    def _prefill_topk(
        self,
        q_quant: torch.Tensor,
        weights: torch.Tensor,
        metadata: DeepseekV32IndexerMetadata,
        pool_size: int,
        positions: torch.Tensor,
    ) -> None:
        prefill = metadata.prefill
        if prefill is None:
            return
        for chunk in prefill.chunks:
            start, end = chunk.token_start, chunk.token_end
            q = q_quant[start:end].float()
            w = weights[start:end].float()
            page_table = chunk.block_table
            seq_lens = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
            scores = q.new_full(
                (end - start, int(seq_lens.max().item())), -float("inf")
            )
            cache = self.k_cache.kv_cache.reshape(
                self.k_cache.kv_cache.shape[0], self.k_cache.kv_cache.shape[1], -1
            )
            block_size = self.k_cache.kv_cache.shape[1]
            for row in range(end - start):
                req = int(
                    torch.searchsorted(
                        chunk.local_cu_seq_lens,
                        chunk.cu_seqlen_ks[row],
                        right=True,
                    )
                    - 1
                )
                length = int(seq_lens[row])
                for token in range(length):
                    block, offset = divmod(token, block_size)
                    physical = int(page_table[req, block])
                    if physical < 0:
                        continue
                    key = _dequantize_cache_vector(cache[physical, offset])
                    scores[row, token] = _weighted_indexer_score(key, q[row], w[row])
            select = min(self.topk_tokens // pool_size, scores.shape[1])
            pool_ids = torch.full(
                (end - start, select),
                -1,
                dtype=torch.int32,
                device=scores.device,
            )
            for row, length_value in enumerate(seq_lens.tolist()):
                length = min(int(length_value), scores.shape[1])
                if length:
                    count = min(select, length)
                    pool_ids[row, :count] = torch.topk(
                        scores[row, :length], count
                    ).indices.to(torch.int32)
            expanded = _expand_pool_ids(
                pool_ids,
                positions[start:end] + 1,
                pool_size,
                self.topk_indices_buffer.shape[1],
            )
            self.topk_indices_buffer[start:end, : expanded.shape[1]].copy_(expanded)

    def _decode_topk(
        self,
        q_quant: torch.Tensor,
        weights: torch.Tensor,
        metadata: DeepseekV32IndexerMetadata,
        pool_size: int,
        positions: torch.Tensor,
    ) -> None:
        decode = metadata.decode
        if decode is None:
            return
        if decode.requires_padding:
            raise NotImplementedError(
                "CPU GLM sparse indexer does not support padded speculative decode yet"
            )
        seq_lens = decode.seq_lens.reshape(-1)
        page_table = decode.block_table
        cache = self.k_cache.kv_cache.reshape(
            self.k_cache.kv_cache.shape[0], self.k_cache.kv_cache.shape[1], -1
        )
        block_size = self.k_cache.kv_cache.shape[1]
        out = self.topk_indices_buffer[: seq_lens.shape[0]]
        for row, length_value in enumerate(seq_lens.tolist()):
            length = int(length_value)
            scores = q_quant[row].float().new_full((length,), -float("inf"))
            for token in range(length):
                block, offset = divmod(token, block_size)
                physical = int(page_table[row, block])
                if physical < 0:
                    continue
                key = _dequantize_cache_vector(cache[physical, offset])
                scores[token] = _weighted_indexer_score(key, q_quant[row], weights[row])
            select = min(self.topk_tokens // pool_size, length)
            pool_ids = torch.full(
                (1, self.topk_tokens // pool_size),
                -1,
                dtype=torch.int32,
                device=scores.device,
            )
            if select:
                pool_ids[0, :select] = torch.topk(scores, select).indices.to(
                    torch.int32
                )
            expanded = _expand_pool_ids(
                pool_ids,
                positions[row : row + 1] + 1,
                pool_size,
                self.topk_indices_buffer.shape[1],
            )
            out[row, : expanded.shape[1]].copy_(expanded[0])
