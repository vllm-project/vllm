# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.models.common.ops.sequence_parallel import shard_windows
from vllm.utils.torch_utils import async_tensor_h2d

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class DSAQueryLayout:
    """Local query segments and their locations in the original paged requests."""

    req_indices: torch.Tensor
    trailing_tokens: torch.Tensor
    query_start_loc: torch.Tensor
    token_indices: torch.Tensor
    num_real_tokens: int
    _device_indices: tuple[torch.Tensor, ...] | None = field(default=None, init=False)

    @classmethod
    def from_offsets(
        cls,
        offsets: np.ndarray,
        model_num_tokens: int,
        tp: int,
        rank: int,
        zigzag: bool,
        full_decode: bool,
    ) -> DSAQueryLayout:
        windows = shard_windows(model_num_tokens, tp, rank, zigzag)
        reqs: list[int] = []
        tails: list[int] = []
        starts = [0]
        rows = []
        for window in windows:
            begin, end = window.start, min(window.stop, int(offsets[-1]))
            if end <= begin:
                continue
            rows.append(torch.arange(begin, end))
            first = int(np.searchsorted(offsets, begin, side="right")) - 1
            last = int(np.searchsorted(offsets, end, side="left"))
            for req in range(first, last):
                left = max(begin, int(offsets[req]))
                right = min(end, int(offsets[req + 1]))
                if right <= left:
                    continue
                # The central pair of zigzag chunks can be adjacent.
                if reqs and reqs[-1] == req and offsets[req + 1] - tails[-1] == left:
                    starts[-1] += right - left
                    tails[-1] = int(offsets[req + 1]) - right
                else:
                    reqs.append(req)
                    tails.append(int(offsets[req + 1]) - right)
                    starts.append(starts[-1] + right - left)
        real = starts[-1]
        # FULL decode captures a fixed query/request count. Eager and
        # piecewise need one harmless query on a rank owning only padded rows.
        count = max(1, real)
        if full_decode:
            count = max(
                1, sum(max(0, min(w.stop, model_num_tokens) - w.start) for w in windows)
            )
        for _ in range(count - real):
            reqs.append(0)
            tails.append(0)
            starts.append(starts[-1] + 1)
        if count > real:
            rows.append(torch.zeros(count - real, dtype=torch.long))
        return cls(
            torch.tensor(reqs, dtype=torch.long),
            torch.tensor(tails, dtype=torch.int32),
            torch.tensor(starts, dtype=torch.int32),
            torch.cat(rows),
            real,
        )

    def local_metadata(
        self,
        cm: CommonAttentionMetadata,
        buffers: dict[str, torch.Tensor] | None,
        capacity: int,
    ) -> CommonAttentionMetadata:
        device = cm.seq_lens.device
        # All cache groups share a query layout for this step.
        if self._device_indices is None:
            self._device_indices = tuple(
                async_tensor_h2d(index, device=device)
                for index in (
                    self.req_indices,
                    self.token_indices,
                    self.trailing_tokens,
                    self.query_start_loc,
                )
            )
        reqs, tokens, tails, starts_device = self._device_indices
        starts = self.query_start_loc
        lens = starts.diff()
        real_reqs = reqs.numel() - (tokens.numel() - self.num_real_tokens)

        def stage(name: str, value: torch.Tensor) -> torch.Tensor:
            if buffers is None:
                return value
            if name not in buffers:
                buffers[name] = value.new_empty((capacity + 1, *value.shape[1:]))
            buf = buffers[name][: value.shape[0]]
            buf.copy_(value)
            return buf

        seq_lens = cm.seq_lens[reqs] - tails
        seq_lens[real_reqs:] = 1
        upper = cm.seq_lens_cpu_upper_bound
        assert upper is not None
        upper = upper[self.req_indices] - self.trailing_tokens
        upper[real_reqs:] = 1
        slots = cm.slot_mapping[tokens]
        slots[self.num_real_tokens :] = -1
        positions = cm.positions[tokens] if cm.positions is not None else None
        prefilling = (
            cm.is_prefilling[self.req_indices] if cm.is_prefilling is not None else None
        )
        return cm.replace(
            num_actual_tokens=tokens.numel(),
            num_reqs=reqs.numel(),
            query_start_loc_cpu=starts,
            query_start_loc=stage("starts", starts_device),
            seq_lens=stage("seq_lens", seq_lens),
            seq_lens_cpu_upper_bound=upper,
            max_query_len=int(lens.max()),
            block_table_tensor=stage("blocks", cm.block_table_tensor[reqs]),
            positions=stage("positions", positions) if positions is not None else None,
            slot_mapping=stage("slots", slots),
            is_prefilling=prefilling,
            req_idx=cm.req_idx[self.req_indices.numpy()]
            if cm.req_idx is not None
            else None,
            _num_computed_tokens_cache=None,
            _token_to_req_indices_cache=None,
            dsa_sp_global=cm,
        )
