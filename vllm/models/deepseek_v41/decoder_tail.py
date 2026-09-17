# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental dependency-preserving compaction of decoder prefill rows."""

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.models.common.ops.sequence_parallel import (
    sp_padding_mask,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def decoder_tail_rows(
    query_starts: np.ndarray, remaining: int, window: int, alignment: int = 1
):
    """Retain enough input to reconstruct all future SWA tails and final logits."""
    limit = window + (remaining - 1) * (window - 1)
    limit = (limit + alignment - 1) // alignment * alignment
    keep = np.minimum(np.diff(query_starts), limit)
    rows = np.concatenate(
        [
            np.arange(end - count, end, dtype=np.int64)
            for end, count in zip(query_starts[1:], keep)
        ]
    )
    starts = np.concatenate(([0], np.cumsum(keep))).astype(np.int32)
    return rows, starts, keep


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    # The managed pinned allocator retains the source until the copy completes.
    return torch.from_numpy(array).pin_memory().to(device, non_blocking=True)


def redistribute_sp_states(states, rows_cpu: np.ndarray, total: int):
    """Move selected rows to their new SP owners, preserving their byte values."""
    group = get_tp_group()
    size, rank = group.world_size, group.rank_in_group
    old_chunk = (total + size - 1) // size
    new_chunk = (len(rows_cpu) + size - 1) // size
    source = rows_cpu // old_chunk
    destination = np.arange(len(rows_cpu)) // new_chunk
    owned = source == rank
    received = destination == rank
    local_rows = to_device(
        (rows_cpu[owned] - rank * old_chunk).astype(np.int64), states[0].device
    )
    send_counts = np.bincount(destination[owned], minlength=size).tolist()
    recv_counts = np.bincount(source[received], minlength=size).tolist()
    fields = [(i, states[i]) for i in (0, 2, 3, 4, 5, 6) if states[i] is not None]
    widths = [int(np.prod(t.shape[1:])) * t.element_size() for _, t in fields]
    parts = [
        t.index_select(0, local_rows)
        .contiguous()
        .view(torch.uint8)
        .reshape(len(local_rows), width)
        for (_, t), width in zip(fields, widths)
    ]
    send = torch.cat(parts, dim=1)
    recv = send.new_empty((new_chunk, sum(widths)))
    valid = int(received.sum())
    # Monotonic row selection means source-rank order is already token order.
    torch.distributed.all_to_all_single(
        recv[:valid], send, recv_counts, send_counts, group=group.device_group
    )
    recv[valid:].zero_()
    result = {}
    offset = 0
    for (i, t), width in zip(fields, widths):
        result[i] = (
            recv[:, offset : offset + width]
            .contiguous()
            .view(t.dtype)
            .reshape(new_chunk, *t.shape[1:])
        )
        offset += width
    return result


class DecoderTail:
    def __init__(
        self,
        config,
        start: int,
        end: int,
        window: int,
        sequence_parallel: bool,
        topk: torch.Tensor,
        candidates: torch.Tensor | None,
    ):
        self.config = config
        self.start, self.end, self.window = start, end, window
        self.sequence_parallel = sequence_parallel
        self.allowed = False
        self.min_num_tokens = 16384
        self.builders: dict[Any, Any] = {}
        self.calls = 0
        self.full_rows = 0
        self.kept_rows = 0
        self.row_buffers = [b for b in (topk, candidates) if b is not None]
        self.reset()

    def reset(self):
        self.original_rows: torch.Tensor | None = None
        self.query_starts: np.ndarray | None = None
        self.original_count = 0
        self.skip = False
        self.metadata_cache: dict[int, Any] = {}
        self.query_cpu: torch.Tensor | None = None
        self.query: torch.Tensor | None = None
        self.padding: torch.Tensor | None = None

    @property
    def active(self):
        return self.original_rows is not None

    def restore_output(self, output):
        assert self.original_rows is not None
        restored = output.new_zeros((self.original_count, *output.shape[1:]))
        return restored.index_copy_(0, self.original_rows, output)

    def _builder(self, source, common):
        if source not in self.builders:
            kwargs = {}
            if isinstance(source, DeepseekV32IndexerMetadataBuilder):
                kwargs["block_table_width"] = common.block_table_tensor.shape[1]
            with set_current_vllm_config(self.config):
                self.builders[source] = type(source)(
                    kv_cache_spec=source.kv_cache_spec,
                    layer_names=source.layer_names,
                    vllm_config=source.vllm_config,
                    device=source.device,
                    **kwargs,
                )
            if source.kernel_block_size is not None:
                self.builders[source].set_kernel_block_size(source.kernel_block_size)
        return self.builders[source]

    def run(self, index: int, layer, states: tuple, kwargs: dict):
        ctx = get_forward_context()
        if (
            not self.allowed
            or self.skip
            or (not self.active and states[1].shape[0] < self.min_num_tokens)
            or index < self.start
            or index >= self.end
            or ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or not isinstance(ctx.attn_metadata, dict)
        ):
            return layer(*states, **kwargs), states[1], states[2]
        attn = layer.attn
        source: Any = ctx.attn_metadata[attn.swa_cache_layer.prefix]
        if source.num_prefills == 0 or source.prefill_left_visible is not None:
            return layer(*states, **kwargs), states[1], states[2]
        _, common = source._decoder_tail_source
        original_starts = common.query_start_loc_cpu[: common.num_reqs + 1].numpy()
        starts = self.query_starts if self.query_starts is not None else original_starts
        rows_cpu, new_starts, kept = decoder_tail_rows(
            starts, self.end - self.start, self.window, alignment=256
        )
        if rows_cpu.size == int(starts[-1]) and not self.active:
            self.skip = True
            return layer(*states, **kwargs), states[1], states[2]
        device = states[0].device
        num_tokens = len(rows_cpu)
        total = states[1].shape[0]
        changed = num_tokens != total
        compact: list[Any]
        if changed:
            rows = to_device(rows_cpu, device)
            if not self.active:
                self.original_count = total
                self.original_rows = torch.arange(total, device=device)
            assert self.original_rows is not None
            self.original_rows = self.original_rows.index_select(0, rows)
            self.query_starts = new_starts
            self.query_cpu = torch.from_numpy(new_starts)
            self.query = self.query_cpu.pin_memory().to(device, non_blocking=True)
            self.metadata_cache = {}
            # Model state order: x, positions, input_ids, pre_mix, post_mix,
            # res_mix, residual, engram_hashes, engram_mask.
            sp_indices = {0, 2, 3, 4, 5, 6}
            exchanged = (
                redistribute_sp_states(states, rows_cpu, total)
                if self.sequence_parallel
                else {}
            )
            compact = []
            for i, value in enumerate(states):
                if value is None or i in (7, 8):
                    compact.append(None)
                elif self.sequence_parallel and i in sp_indices:
                    compact.append(exchanged[i])
                else:
                    compact.append(value.index_select(0, rows))
            for buf in self.row_buffers:
                buf[:num_tokens].copy_(buf.index_select(0, rows))
            self.padding = (
                sp_padding_mask(None, compact[1]) if self.sequence_parallel else None
            )
        else:
            compact = list(states)
        query_cpu, query = self.query_cpu, self.query
        assert query_cpu is not None and query is not None
        assert self.original_rows is not None
        metadata = dict(ctx.attn_metadata)
        names = [attn.swa_cache_layer.prefix, attn.compressed_cache_prefix]
        if attn.indexer is not None:
            names.append(attn.indexer.k_cache.prefix)
        built = self.metadata_cache
        for name in names:
            original: Any = ctx.attn_metadata[name]
            if id(original) not in built:
                builder, full = original._decoder_tail_source
                reduced = replace(
                    full,
                    query_start_loc=query,
                    query_start_loc_cpu=query_cpu,
                    num_actual_tokens=num_tokens,
                    max_query_len=int(kept.max()),
                    slot_mapping=full.slot_mapping.index_select(0, self.original_rows),
                    positions=compact[1],
                    _num_computed_tokens_cache=None,
                    _token_to_req_indices_cache=None,
                )
                md = self._builder(builder, full).build(0, reduced)
                if isinstance(md, DeepseekSparseSWAMetadata):
                    # The discarded leading halo must not read unwritten SWA slots.
                    # Untrimmed requests preserve their normal old-cache history.
                    trimmed = to_device(np.diff(original_starts) > kept, device)
                    lengths = query[1:] - query[:-1]
                    md.prefill_gather_lens = torch.where(
                        trimmed[md.num_decodes :],
                        lengths[md.num_decodes :],
                        md.prefill_gather_lens,
                    )
                built[id(original)] = md
            metadata[name] = built[id(original)]
        reduced_ctx = replace(
            ctx, attn_metadata=metadata, is_padding=self.padding, batch_descriptor=None
        )
        with override_forward_context(reduced_ctx):
            result = layer(*compact, **kwargs)
        self.calls += 1
        self.full_rows += int(original_starts[-1])
        self.kept_rows += num_tokens
        return result, compact[1], compact[2]
