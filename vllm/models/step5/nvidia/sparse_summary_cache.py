# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch CSA summaries attached to the attention KV pages."""

from typing import Any

import numpy as np
import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend


class Step5SparseSummaryCache:
    """FP8 history and ordered FP32 accumulators for each page's open region."""

    def __init__(
        self,
        pages: int,
        block_size: int,
        device: torch.device,
        max_num_seqs: int = 1,
        request_journal_width: int = 0,
    ):
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        self.block_size = block_size
        self.pages = pages
        # Padded decode requests write distinct scratch pages, never live KV pages.
        self._mean = torch.zeros(
            (pages + max_num_seqs, block_size // 8, 256),
            dtype=torch.uint8,
            device=device,
        )
        self.mean = self._mean[:pages]
        self.numerator = torch.zeros(
            (pages + max_num_seqs, 256), device=device, dtype=torch.float32
        )
        self.denominator = torch.zeros_like(self.numerator)
        self.maximum = torch.full_like(self.numerator[:, :1], float("-inf"))
        self.last_slot = torch.full(
            (pages + max_num_seqs,), -1, dtype=torch.long, device=device
        )
        # CPU reference journal; CUDA uses ``update_request_journal``.
        self._speculative_tails: dict[int, dict[int, tuple[torch.Tensor, ...]]] = {}
        # MRv2 journal addressed by persistent request state index; see
        # ``update_request_journal``.
        self.request_journal_width = request_journal_width
        if request_journal_width:
            rows = max_num_seqs * request_journal_width
            self.request_journal_values = torch.empty(
                (rows, 513), device=device, dtype=torch.float32
            )
            self.request_journal_last = torch.empty(
                rows, device=device, dtype=torch.int64
            )
            self.request_journal_tag = torch.full(
                (rows,), -1, device=device, dtype=torch.int64
            )

    def update_request_journal(
        self,
        key,
        z,
        slots,
        query_start_loc,
        seq_lens,
        request_state,
        prefix_metadata=None,
    ):
        """Speculative update undone on device; no host metadata when uniform.

        ``prefix_metadata`` covers rows before each request's last window.
        """
        from .ops.summary_update import update_prefill, update_request_journal

        args = (
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            request_state,
            self.request_journal_values,
            self.request_journal_last,
            self.request_journal_tag,
            self.mean,
            self.numerator,
            self.denominator,
            self.maximum,
            self.last_slot,
            self.pages,
            self.block_size,
            self.request_journal_width,
        )
        if prefix_metadata is None:
            update_request_journal(*args)
            return
        update_request_journal(*args, update=False)
        update_prefill(
            key,
            z,
            prefix_metadata,
            self.mean,
            self.numerator,
            self.denominator,
            self.maximum,
            self.last_slot,
            self.block_size,
        )
        update_request_journal(*args, restore=False)

    def _invalidate_request_journal(self, blocks) -> None:
        if self.request_journal_width and blocks:
            tag = self.request_journal_tag
            pages = torch.tensor(sorted(blocks), device=tag.device)
            tag.masked_fill_(torch.isin(tag // self.block_size, pages), -1)

    def update_speculative(
        self,
        key,
        z,
        slots,
        starts,
        tables,
        num_speculative_tokens,
        *,
        owned_pages=None,
        slot_mapping=None,
    ):
        """Restore the accepted tail before overwriting rejected draft positions.

        Only the last verification window needs undo state. Completed regions
        before that window are immutable; overwritten regions are recomputed.
        """
        assert not key.is_cuda, "CUDA uses update_request_journal"
        tensors = (self.numerator, self.denominator, self.maximum, self.last_slot)
        for req, (start, end) in enumerate(zip(starts, starts[1:])):
            if start == end:
                continue
            slot = slots[start]
            if slot < 0:
                continue
            page = slot // self.block_size
            snapshot = self._speculative_tails.get(page, {}).get(slot)
            if snapshot is not None:
                for tensor, saved in zip(tensors, snapshot):
                    tensor[page] = saved
            # Metadata contains padding block IDs after the request's last page.
            # Only remove journals for pages actually owned by this request.
            last_page = slots[end - 1] // self.block_size
            table = tables[req]
            last_index = table.index(last_page)
            for old_page in table[: last_index + 1]:
                self._speculative_tails.pop(old_page, None)
            tail = max(start, end - num_speculative_tokens - 1)
            self.update(key[start:tail], z[start:tail], slots[start:tail])
            for row in range(tail, end):
                slot = slots[row]
                if slot < 0:
                    continue
                page = slot // self.block_size
                self._speculative_tails.setdefault(page, {})[slot] = tuple(
                    tensor[page].clone() for tensor in tensors
                )
                self.update(key[row : row + 1], z[row : row + 1], [slot])

    @staticmethod
    def _accumulate(numerator, denominator, maximum, key, z):
        updated_max = torch.maximum(maximum, z.amax(-1, keepdim=True))
        scale = torch.where(denominator > 0, (maximum - updated_max).exp(), 0)
        weight = (z - updated_max).exp()
        return (
            numerator * scale + weight * key,
            denominator * scale + weight,
            updated_max,
        )

    def update_decode(self, key, z, slots):
        if key.is_cuda:
            from .ops.summary_update import update_decode

            update_decode(
                key,
                z,
                slots,
                self._mean,
                self.numerator,
                self.denominator,
                self.maximum,
                self.last_slot,
                self.pages,
                self.block_size,
            )
            return
        active = slots >= 0
        pages = torch.where(
            active,
            slots // self.block_size,
            self.pages + torch.arange(slots.numel(), device=slots.device),
        )
        fresh = (slots % 8 == 0) | ~active
        torch._assert_async(
            (fresh | (self.last_slot[pages] == slots - 1)).all(),
            "Step5 CSA tail missing for a partial region",
        )
        numerator = torch.where(fresh[:, None], 0, self.numerator[pages])
        denominator = torch.where(fresh[:, None], 0, self.denominator[pages])
        maximum = torch.where(fresh[:, None], float("-inf"), self.maximum[pages])
        numerator, denominator, maximum = self._accumulate(
            numerator, denominator, maximum, key[:, 0].float(), z[:, 0].float()
        )
        regions = slots.clamp_min(0) % self.block_size // 8
        self._mean[pages, regions] = (
            (numerator / denominator.clamp_min(1e-20))
            .bfloat16()
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        self.numerator[pages] = numerator
        self.denominator[pages] = denominator
        self.maximum[pages] = maximum
        self.last_slot[pages] = torch.where(active & (slots % 8 != 7), slots, -1)

    def speculative_prefix_metadata(
        self, slots: np.ndarray, starts: np.ndarray, device
    ) -> torch.Tensor | None:
        """Grouped rows before each request's journaled last window, if any."""
        slots, starts = np.asarray(slots), np.asarray(starts)
        begin, end = starts[:-1], starts[1:]
        live = (begin < end) & (slots[np.minimum(begin, len(slots) - 1)] >= 0)
        tail = np.where(
            live, np.maximum(begin, end - self.request_journal_width), begin
        )
        # Mark [begin, tail) of each live request; everything else is skipped.
        marks = np.zeros(len(slots) + 1, dtype=np.int64)
        np.add.at(marks, begin, 1)
        np.add.at(marks, tail, -1)
        prefix = np.where(np.cumsum(marks[:-1]) > 0, slots, -1)
        return self.prefill_metadata(prefix, device)

    def prefill_metadata(self, slots: np.ndarray, device) -> torch.Tensor | None:
        from .ops.summary_update import make_prefill_metadata_from_slots

        return make_prefill_metadata_from_slots(
            np.asarray(slots), self.block_size, device
        )

    @staticmethod
    def _group_slots(slots: list[int]) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for row, slot in enumerate(slots):
            if slot >= 0:
                groups.setdefault(slot // 8, [-1] * 8)[slot % 8] = row
        return groups

    def update(
        self,
        key: torch.Tensor,
        z: torch.Tensor,
        slots: list[int],
        prefill_metadata: torch.Tensor | None = None,
    ) -> None:
        if prefill_metadata is not None:
            from .ops.summary_update import update_prefill

            update_prefill(
                key,
                z,
                prefill_metadata,
                self.mean,
                self.numerator,
                self.denominator,
                self.maximum,
                self.last_slot,
                self.block_size,
            )
            return
        groups = self._group_slots(slots)
        if not groups:
            return
        if key.is_cuda:
            from .ops.summary_update import update_prefill

            update_prefill(
                key,
                z,
                groups,
                self.mean,
                self.numerator,
                self.denominator,
                self.maximum,
                self.last_slot,
                self.block_size,
            )
        else:
            self._update_torch(key, z, groups)

    def _update_torch(self, key, z, groups):
        regions = list(groups)
        rows = torch.tensor(list(groups.values()), device=key.device)
        valid = rows >= 0
        keys = key[:, 0][rows.clamp_min(0)].float()
        logits = z[:, 0][rows.clamp_min(0)].float()
        region_ids = torch.tensor(regions, device=key.device)
        pages = region_ids * 8 // self.block_size
        fresh = valid[:, 0, None]
        first = torch.tensor(
            [next(j for j, row in enumerate(m) if row >= 0) for m in groups.values()],
            device=key.device,
        )
        if not bool(
            (fresh[:, 0] | (self.last_slot[pages] == region_ids * 8 + first - 1)).all()
        ):
            raise RuntimeError("Step5 CSA tail missing for a partial region")
        numerator = torch.where(fresh, 0, self.numerator[pages])
        denominator = torch.where(fresh, 0, self.denominator[pages])
        maximum = torch.where(fresh, float("-inf"), self.maximum[pages])
        for offset in range(8):
            updated = self._accumulate(
                numerator, denominator, maximum, keys[:, offset], logits[:, offset]
            )
            numerator, denominator, maximum = (
                torch.where(valid[:, offset, None], new, old)
                for new, old in zip(updated, (numerator, denominator, maximum))
            )
        mean = numerator / denominator.clamp_min(1e-20)
        self.mean.view(-1, 256)[region_ids] = (
            mean.bfloat16().to(torch.float8_e4m3fn).view(torch.uint8)
        )
        # Only the final region of each physical page can remain open.
        last = {region * 8 // self.block_size: i for i, region in enumerate(regions)}
        dest = list(last)
        src = list(last.values())
        self.numerator[dest] = numerator[src]
        self.denominator[dest] = denominator[src]
        self.maximum[dest] = maximum[src]
        final_slots = [
            regions[i] * 8
            + max(j for j, row in enumerate(groups[regions[i]]) if row >= 0)
            for i in src
        ]
        self.last_slot[dest] = torch.tensor(
            [s if s % 8 != 7 else -1 for s in final_slots], device=key.device
        )

    def reset(self, block_ids: list[int], clear_mean: bool = True) -> None:
        blocks = set(block_ids)
        for block in blocks:
            self._speculative_tails.pop(block, None)
        # Reused pages must not match a finished request's undo tags.
        self._invalidate_request_journal(blocks)
        if clear_mean and blocks:
            self.mean[list(blocks)] = 0
        self.last_slot[list(blocks)] = -1

    def copy(self, copies: list[Any]) -> None:
        copies = [c for c in copies if c.src_block_id != c.dst_block_id]
        if not copies:
            return
        sources = [c.src_block_id for c in copies]
        destinations = [c.dst_block_id for c in copies]
        snapshots = [
            t[sources].clone()
            for t in (
                self.mean,
                self.numerator,
                self.denominator,
                self.maximum,
                self.last_slot,
            )
        ]
        for tensor, snapshot in zip(
            (self.mean, self.numerator, self.denominator, self.maximum), snapshots
        ):
            tensor[destinations] = snapshot
        journals = [self._speculative_tails.get(src, {}) for src in sources]
        for dst, journal in zip(destinations, journals):
            copied = {}
            for slot, saved in journal.items():
                state = tuple(t.clone() for t in saved)
                state[-1].copy_(
                    torch.where(
                        state[-1] >= 0,
                        dst * self.block_size + state[-1] % self.block_size,
                        -1,
                    )
                )
                copied[dst * self.block_size + slot % self.block_size] = state
            self._speculative_tails[dst] = copied
        # A COW destination starts a new request, which has no window to undo.
        self._invalidate_request_journal(destinations)
        old_slots = snapshots[-1]
        offsets = torch.tensor(destinations, device=self.mean.device) * self.block_size
        self.last_slot[destinations] = torch.where(
            old_slots >= 0, offsets + old_slots % self.block_size, -1
        )


class Step5SparseSummaryCacheLayer(torch.nn.Module, AttentionLayerBase):
    """Integrate summary allocation, reset and COW with the vLLM KV lifecycle."""

    def __init__(
        self,
        *,
        prefix,
        target_impl,
        main_layer_name,
        static_forward_context,
    ):
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self.layer_name = prefix
        self._target_impl = target_impl
        self.main_layer_name = main_layer_name
        self.kv_cache = torch.tensor([])
        self._cache: Step5SparseSummaryCache | None = None

    def get_attn_backend(self) -> type[AttentionBackend]:
        from .sparse_attention import Step5DSAAttentionBackend

        return Step5DSAAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        return None

    def _set_summary_cache_budget(self, block_size: int) -> int:
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        return block_size // 8 * 256 + 513 * 4 + 8

    def _request_journal_width(self) -> int:
        tokens = self._target_impl.num_speculative_tokens
        return tokens + 1 if tokens else 0

    def _fixed_runtime_state_budget(self, block_size: int) -> int:
        # Undo rows per request slot: 513 FP32 state, last slot and tag.
        journal = (
            self._target_impl.max_num_seqs
            * self._request_journal_width()
            * (513 * 4 + 16)
        )
        return journal + self._target_impl.max_num_seqs * (
            self._set_summary_cache_budget(block_size)
        )

    def _peers(self, context):
        return [v for v in context.values() if isinstance(v, type(self))]

    def kv_cache_side_storage_memory_profile_key(self, forward_context):
        return self._peers(forward_context)[0]

    def prepare_kv_cache_side_storage_memory_profile(self, forward_context, device):
        from .flashinfer_sparse import Step5FlashInferSparse

        peers = self._peers(forward_context)
        adapter = peers[0]._target_impl.adapter
        if adapter is None:
            adapter = Step5FlashInferSparse(device)
        for peer in peers:
            peer._target_impl.adapter = adapter

    def bind_kv_cache_side_storage(self, forward_context):
        self.prepare_kv_cache_side_storage_memory_profile(
            forward_context, forward_context[self.main_layer_name].kv_cache.device
        )
        kv = forward_context[self.main_layer_name].kv_cache
        self._cache = Step5SparseSummaryCache(
            kv.shape[0],
            kv.shape[2],
            kv.device,
            self._target_impl.max_num_seqs,
            self._request_journal_width(),
        )
        self.kv_cache = self._cache.mean.view(kv.shape[0], -1)
        self._target_impl.summary_cache = self._cache

    def kv_cache_side_storage_transfer_view(self) -> torch.Tensor:
        # Full blocks only hold closed regions; open-region accumulators are
        # reset when a block is allocated, so the FP8 summaries are enough.
        assert self._cache is not None
        return self._cache.mean.view(self._cache.pages, -1)

    def zero_kv_cache_side_storage(self, block_ids):
        if self._cache is not None:
            self._cache.reset(self._ids(block_ids))

    def release_kv_cache_side_storage(self, block_ids):
        if self._cache is not None:
            self._cache.reset(self._ids(block_ids), clear_mean=False)

    def reset_kv_cache_side_storage_runtime_state(self):
        if self._cache is not None:
            self._cache.last_slot.fill_(-1)
            self._cache.mean.zero_()
            self._cache._speculative_tails.clear()
            if self._cache.request_journal_width:
                self._cache.request_journal_tag.fill_(-1)
        return 1, 0

    def copy_kv_cache_side_storage(self, block_copies, num_blocks):
        if self._cache is not None and block_copies is not None:
            self._cache.copy(list(block_copies))

    @staticmethod
    def _ids(block_ids):
        return block_ids.tolist() if isinstance(block_ids, torch.Tensor) else block_ids
