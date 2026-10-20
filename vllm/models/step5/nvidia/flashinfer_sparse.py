# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 sparse selection and attention using FlashInfer and PyTorch."""

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config_or_none

from .ops.prefill_indexer_metadata import prepare_prefill_indexer
from .ops.sparse_decode_attention import (
    sparse_decode_attention,
    sparse_decode_splits,
)
from .ops.sparse_indexer import (
    pack_decode_summary_pages,
    pack_summary_pages,
    select_regions,
)
from .ops.sparse_indexer_scores import decode_summary_scores
from .ops.sparse_prefill import PrefillChunk, gather_prefill_kv
from .ops.sparse_prefill_pages import CompactPagedPrefill


def _direct_sparse_decode(query, kv_cache, out) -> bool:
    """Whether the direct paged sparse decode kernel supports these tensors."""
    return (
        query.shape[2] == 192
        and query.shape[1] <= 16
        and kv_cache.shape[-1] == 384
        and kv_cache.stride(-1) == 1
        and query.stride(2) == 1
        and out.stride(2) == 1
    )


class Step5FlashInferSparse:
    def __init__(self, device: torch.device):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        from flashinfer.attn_scores import fp8_paged_mqa_logits

        config = get_current_vllm_config_or_none()
        self._topk_backend = (
            config.kernel_config.sparse_indexer_topk_backend if config else "auto"
        )
        if torch.cuda.get_device_capability(device) == (9, 0):
            from .ops.sparse_indexer_scores import paged_mqa_logits_sm90

            self._logits = paged_mqa_logits_sm90
        else:
            self._logits = fp8_paged_mqa_logits
        properties = torch.cuda.get_device_properties(device)
        self._num_sms = properties.multi_processor_count
        self._workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
        self._attention = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace, "NHD", backend="fa2"
        )
        self._prefill_plan_event = torch.cuda.Event()
        self._paged_prefill_plans: list[CompactPagedPrefill | None] = [None, None]
        self._paged_prefill_index = 0

    def prepare_prefill_attention(
        self,
        query_shape,
        dtype,
        device,
        block_size,
        positions,
        region_count,
        scale,
        *,
        contiguous_query=True,
    ) -> CompactPagedPrefill | None:
        """Plan paged prefill over dense model-dtype single-group pages.

        FP8 and multi-group caches qualify once staged by the caller.
        """
        if not (
            query_shape[1] in (8, 16)
            and query_shape[2] == 192
            and dtype == torch.bfloat16
            and block_size in (16, 32, 64, 128)
        ):
            return None
        # Alternate plans: planning ahead of the forward must not overwrite
        # pinned buffers the previous step's copies may still be reading.
        self._paged_prefill_index ^= 1
        plan = self._paged_prefill_plans[self._paged_prefill_index]
        if plan is None:
            plan = CompactPagedPrefill(self._workspace)
            self._paged_prefill_plans[self._paged_prefill_index] = plan
        plan.plan(
            query_shape,
            dtype,
            device,
            positions,
            region_count,
            scale,
            block_size,
            contiguous_query=contiguous_query,
        )
        return plan

    def select_and_run_decode_attention(
        self,
        query,
        kv_cache,
        scores,
        current,
        topk,
        block_table,
        positions,
        active,
        out,
        scale,
        *,
        topk_lengths=None,
        clear_inactive=True,
        k_scale=None,
        v_scale=None,
        query_len=1,
        gate=None,
        capture=None,
    ):
        """Decode attention for rows of ``query_len`` tokens per request.

        ``gate`` ([tokens, all heads]) applies the headwise sigmoid gate.
        """
        if not _direct_sparse_decode(query, kv_cache, out):
            raise NotImplementedError(
                "Step5 sparse decode requires head_size 192, at most 16 query "
                "heads per KV group and a contiguous packed KV cache"
            )
        regions = self.select_regions(
            scores, current, topk, topk_lengths=topk_lengths, sort=False
        )
        if capture is not None:
            capture.capture_regions(regions)
        # Reads selected regions straight from paged KV: no gather scratch,
        # no host plan, split merge fused for small split counts.
        sparse_decode_attention(
            query,
            kv_cache,
            regions,
            block_table,
            positions,
            active,
            out,
            scale=scale,
            query_len=query_len,
            splits=sparse_decode_splits(query.shape[0]),
            clear_inactive=clear_inactive or gate is not None,
            k_scale=k_scale,
            v_scale=v_scale,
            gate=gate,
        )

    def prefill_metadata(self, owners, positions, regions, topk, *, paged_prefill=None):
        return prepare_prefill_indexer(
            owners,
            positions,
            regions,
            topk,
            self._num_sms,
            paged_prefill=paged_prefill,
        )

    def warmup_prefill_scoring(self, max_regions, topk, paired_heads, device):
        """JIT the scored-prefill variants a startup warmup step cannot reach.

        Per-chunk metadata specializes on the chunk size, which shrinks as the
        context grows, and paired logits compile for ``next_n = 2``; either
        would otherwise compile mid-serving on the first such prefill.
        """
        from flashinfer.attn_scores import compute_paged_mqa_logits_schedule

        from .ops.prefill_indexer_metadata import prefill_indexer_chunk_size

        rows = torch.arange(2, device=device)
        owners = torch.zeros(2, dtype=torch.long, device=device)
        chunk_regions: dict[int, int] = {}
        for regions in range(topk + 1, max(max_regions, topk + 1) + 1, 256):
            chunk_regions.setdefault(prefill_indexer_chunk_size(regions), regions)
        for regions in chunk_regions.values():
            positions = rows + (regions - 1) * 8
            self.prefill_metadata(owners, positions, regions, topk)
        if paired_heads:
            regions = topk + 1
            pages = (regions + 127) // 128 * 4
            packed = torch.zeros((pages, 32, 1, 260), dtype=torch.uint8, device=device)
            lengths = torch.full((1,), regions, dtype=torch.int32, device=device)
            self._logits(
                q=torch.zeros(
                    (1, 2, paired_heads, 256), dtype=torch.float8_e4m3fn, device=device
                ),
                kv_fused=packed,
                weights=torch.zeros((2, paired_heads), device=device),
                seq_lens=lengths,
                block_tables=torch.arange(pages, dtype=torch.int32, device=device)[
                    None
                ],
                max_seq_len=regions,
                schedule_meta=compute_paged_mqa_logits_schedule(lengths),
            )

    def prefill_pair_inputs(self, indexer):
        """Per-chunk (context, table, schedule) of consecutive row pairs."""
        from flashinfer.attn_scores import compute_paged_mqa_logits_schedule

        pairs = []
        for meta in indexer:
            lengths = meta.lengths[1::2].contiguous()
            pairs.append(
                (
                    lengths,
                    meta.block_table[::2].contiguous(),
                    compute_paged_mqa_logits_schedule(lengths),
                )
            )
        return pairs

    def pack_summaries(self, mean, block_table, regions):
        return pack_summary_pages(mean, block_table, regions)

    def pack_decode_summaries(self, mean, block_table, regions, slots, seq_lens):
        return pack_decode_summary_pages(
            mean,
            block_table,
            regions,
            slots,
            seq_lens,
            num_sms=self._num_sms,
            live_only=block_table.shape[0] >= 8,
        )

    def decode_logits(
        self,
        query,
        weights,
        mean,
        table,
        regions,
        slots,
        seq_lens,
        row_groups=1,
        query_len=1,
    ):
        """Score regions for (token, KV group) rows.

        ``query`` is ``[tokens, row_groups, heads, 256]`` and ``weights`` is
        ``[tokens, row_groups, heads]``; row ``r`` is group ``r % row_groups``
        of token ``r // row_groups``. ``query_len`` tokens share a request of
        ``table``/``seq_lens``.
        """
        assert query.shape[1] == row_groups
        if (
            query.shape[0] > 0
            and query.shape[2:] == (8, 256)
            and query.dtype == torch.float8_e4m3fn
            and query.stride(3) == 1
            and query.stride(2) == 256
            and weights.shape == query.shape[:3]
            and weights.dtype == torch.float32
            and weights.stride(-1) == 1
            and (mean.shape[1:] == (8, 256) or mean.shape[1:] in ((2, 256), (4, 256)))
            and mean.dtype == torch.uint8
            and mean.stride(1) == 256
            and mean.stride(2) == 1
            and table.stride(1) == 1
            and regions > 0
        ):
            return decode_summary_scores(
                query,
                weights,
                mean,
                table,
                slots,
                seq_lens,
                regions,
                num_sms=self._num_sms,
                row_groups=row_groups,
                query_len=query_len,
            )
        if query_len > 1:
            offset = torch.arange(slots.shape[0], device=slots.device) % query_len
            seq_lens = seq_lens.repeat_interleave(query_len) - (query_len - 1 - offset)
            table = table.repeat_interleave(query_len, dim=0)
        if row_groups > 1:
            slots, table, seq_lens = (
                t.repeat_interleave(row_groups, dim=0) for t in (slots, table, seq_lens)
            )
        rows = query.shape[0] * row_groups
        query = query.reshape(rows, 1, *query.shape[2:])
        weights = weights.reshape(rows, 1, weights.shape[2])
        packed, decode = self.pack_decode_summaries(
            mean, table, regions, slots, seq_lens
        )
        scores = self.logits(
            query,
            weights,
            packed,
            decode.block_table.shape[1],
            None,
            decode.lengths,
            regions,
            block_table=decode.block_table,
            schedule_meta=decode.schedule,
        )
        return scores, decode

    def select_regions(
        self,
        scores,
        current,
        topk,
        *,
        topk_lengths=None,
        paged_prefill=None,
        sort=True,
    ):
        return select_regions(
            scores,
            current,
            topk,
            self._topk_backend,
            topk_lengths=topk_lengths,
            paged_prefill=paged_prefill,
            sort=sort,
        )

    def logits(
        self,
        query,
        weights,
        packed,
        pages,
        requests,
        lengths,
        regions,
        *,
        block_table=None,
        schedule_meta=None,
        pairs=None,
        paired_heads=0,
    ):
        """With ``pairs`` (from ``prefill_pair_inputs``), consecutive row pairs
        of one request are scored as ``next_n=2`` on their ``paired_heads``
        real heads rather than zero-padding each row to the MMA minimum."""
        q = query[:, 0]
        w = weights[:, 0]
        if pairs is not None:
            assert q.dtype == torch.float8_e4m3fn and q.shape[0] % 2 == 0
            pair_lengths, pair_table, pair_schedule = pairs
            return self._logits(
                q=q[:, :paired_heads].unflatten(0, (-1, 2)),
                kv_fused=packed,
                weights=w[:, :paired_heads],
                seq_lens=pair_lengths,
                block_tables=pair_table,
                max_seq_len=regions,
                schedule_meta=pair_schedule,
            )
        if q.dtype != torch.float8_e4m3fn:
            q = q.float()
            w = w.float()
            heads = q.shape[1]
            if heads < 8:
                q = F.pad(q, (0, 0, 0, 8 - heads))
                w = F.pad(w, (0, 8 - heads))
            q = q.to(torch.float8_e4m3fn)
        q = q.unsqueeze(1)
        if block_table is None:
            block_table = (
                requests[:, None] * pages + torch.arange(pages, device=q.device)
            ).int()
        return self._logits(
            q=q,
            kv_fused=packed,
            weights=w.contiguous(),
            seq_lens=lengths.int().contiguous(),
            block_tables=block_table,
            max_seq_len=regions,
            schedule_meta=schedule_meta,
        )

    def run_attention(
        self,
        query,
        kv_cache,
        regions,
        block_table,
        requests,
        positions,
        out,
        scale,
        *,
        prefill_chunks: list[PrefillChunk],
        k_scale=None,
        v_scale=None,
    ):
        for index, chunk in enumerate(prefill_chunks):
            start = index * 32
            end = start + chunk.cu_q.numel() - 1
            k, v = gather_prefill_kv(
                kv_cache,
                regions[start:end],
                block_table,
                requests[start:end],
                positions[start:end],
                chunk.cu_k_device,
                chunk.total_tokens,
                chunk.max_length,
                output_dtype=query.dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            # FlashInfer reuses pinned host plan data across asynchronous copies.
            self._prefill_plan_event.synchronize()
            self._attention.plan(
                chunk.cu_q,
                chunk.cu_k,
                query.shape[1],
                1,
                query.shape[2],
                head_dim_vo=query.shape[2],
                causal=False,
                sm_scale=scale,
                q_data_type=query.dtype,
                kv_data_type=query.dtype,
                disable_split_kv=True,
            )
            self._prefill_plan_event.record(torch.cuda.current_stream(query.device))
            self._attention.run(query[start:end].contiguous(), k, v, out=out[start:end])
