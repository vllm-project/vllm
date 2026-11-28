# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

_INT32_MAX = np.iinfo(np.int32).max
_COLD_SCAN_BLOCK = 1024


class ThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: "ReasoningConfig | None",
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        natural_end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids or []
        )
        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        if not self.enabled:
            return

        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(-1)
        self.thinking_token_budget.copy_to_uva()
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)

        self.cached_last_start = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_last_end = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_scan_pos = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._reset_reqs: list[int] = []
        self._budget_dirty = False

        self.reasoning_start_token_ids = torch.tensor(
            start_ids, dtype=torch.int32, device=self.device
        )
        self.natural_reasoning_end_token_ids = torch.tensor(
            natural_end_ids, dtype=torch.int32, device=self.device
        )
        self.reasoning_end_token_ids = torch.tensor(
            end_ids, dtype=torch.int32, device=self.device
        )
        implicit_sequences = getattr(
            reasoning_config, "implicit_reasoning_end_token_sequences", []
        )
        self.implicit_reasoning_end_lengths = tuple(map(len, implicit_sequences))
        self.implicit_reasoning_end_token_ids = torch.tensor(
            [token_id for sequence in implicit_sequences for token_id in sequence],
            dtype=torch.int32,
            device=self.device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        if not self.enabled:
            return False
        budget = sampling_params.thinking_token_budget
        use_thinking_budget = budget is not None
        self.use_thinking_budget[req_idx] = use_thinking_budget
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self._reset_reqs.append(req_idx)
        if self.thinking_token_budget.np[req_idx] != budget:
            self.thinking_token_budget.np[req_idx] = budget
            self._budget_dirty = True
        return use_thinking_budget

    def apply_staged_writes(self) -> None:
        if not self.enabled:
            return
        if self._reset_reqs:
            idx = async_tensor_h2d(
                self._reset_reqs, dtype=torch.int64, device=self.device
            )
            self.cached_last_start.index_fill_(0, idx, -1)
            self.cached_last_end.index_fill_(0, idx, -1)
            self.cached_scan_pos.index_fill_(0, idx, 0)
            self._reset_reqs.clear()
        if self._budget_dirty:
            self.thinking_token_budget.copy_to_uva()
            self._budget_dirty = False

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> None:
        idx_mapping_np = ctx.idx_mapping_np
        if not self.enabled or not np.any(self.use_thinking_budget[idx_mapping_np]):
            return

        apply_thinking_budget(
            logits,
            ctx.idx_mapping,
            ctx.expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            ctx.input_ids,
            ctx.expanded_local_pos,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
            self.implicit_reasoning_end_token_ids,
            self.implicit_reasoning_end_lengths,
        )


@triton.jit
def _load_effective_token(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    pos,
):
    if pos < total_len:
        return tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos)
    # In decode/spec-decode, input_ids at local position 0 is the already
    # committed last sampled token. Effective draft-prefix positions start at
    # local position 1.
    input_pos = cur_req_first_pos + pos - total_len + 1
    return tl.load(input_ids_ptr + input_pos)


@triton.jit
def _effective_sequence_matches(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    sequence_ptr,
    sequence_offset,
    sequence_start,
    SEQUENCE_LEN: tl.constexpr,
):
    matches = True
    for j in tl.static_range(SEQUENCE_LEN):
        expected = tl.load(sequence_ptr + sequence_offset + j)
        actual = _load_effective_token(
            all_token_ids_ptr,
            all_token_ids_stride,
            input_ids_ptr,
            cur_req_first_pos,
            req_state_idx,
            total_len,
            sequence_start + j,
        )
        matches = matches & (actual == expected)
    return matches


@triton.jit
def _active_reasoning_start_before(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    limit,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    implicit_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    IMPLICIT_END_LENS: tl.constexpr,
):
    # Only needed for matching forced tails: markers inside that tail must not
    # hide whether forcing began in thinking or in an already open tool call.
    active_start = -1
    found = False
    i = limit - 1
    while i >= 0 and not found:
        start_match = False
        if i + 1 >= START_LEN:
            start_match = _effective_sequence_matches(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                reasoning_start_token_ids_ptr,
                0,
                i + 1 - START_LEN,
                START_LEN,
            )
        end_match = False
        if i + 1 >= NATURAL_END_LEN:
            end_match = _effective_sequence_matches(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                natural_reasoning_end_token_ids_ptr,
                0,
                i + 1 - NATURAL_END_LEN,
                NATURAL_END_LEN,
            )
        if i + 1 >= END_LEN:
            end_match = end_match | _effective_sequence_matches(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                reasoning_end_token_ids_ptr,
                0,
                i + 1 - END_LEN,
                END_LEN,
            )
        offset = 0
        for marker in tl.static_range(len(IMPLICIT_END_LENS)):
            marker_len = IMPLICIT_END_LENS[marker]
            if i + 1 >= marker_len:
                end_match = end_match | _effective_sequence_matches(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    implicit_reasoning_end_token_ids_ptr,
                    offset,
                    i + 1 - marker_len,
                    IMPLICIT_END_LENS[marker],
                )
            offset += marker_len
        if start_match and not end_match:
            active_start = i + 1 - START_LEN
        found = start_match | end_match
        i -= 1
    return active_start


@triton.jit
def _update_committed_marker_cache_kernel(
    req_ids_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    implicit_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    IMPLICIT_END_LENS: tl.constexpr,
    END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if scan_pos == 0 and last_start < 0 and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the first
        # block with a marker; only the relative order of the two positions
        # found matters below.
        block_hi = total_len
        while block_hi > 0 and last_start < 0 and last_end < 0:
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + tl.arange(0, BLOCK)

            start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)

            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)

            implicit_offset = 0
            for marker in tl.static_range(len(IMPLICIT_END_LENS)):
                implicit_len = IMPLICIT_END_LENS[marker]
                implicit_match = (offs < block_hi) & (offs + implicit_len <= total_len)
                for j in tl.static_range(IMPLICIT_END_LENS[marker]):
                    expected = tl.load(
                        implicit_reasoning_end_token_ids_ptr + implicit_offset + j
                    )
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + offs
                        + j,
                        mask=offs + j < total_len,
                        other=-1,
                    )
                    implicit_match = implicit_match & (actual == expected)
                end_match = end_match | implicit_match
                implicit_offset += implicit_len

            last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
            last_end = tl.max(tl.where(end_match, offs, -1), axis=0)
            if len(IMPLICIT_END_LENS):
                force_start = offs - END_LEN + 1
                force_match = (offs < block_hi) & (force_start >= 0)
                for j in tl.static_range(END_LEN):
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + force_start
                        + j,
                        mask=(force_start + j >= 0) & (force_start + j < total_len),
                        other=-1,
                    )
                    force_match = force_match & (actual == expected)
                last_end = tl.maximum(
                    last_end, tl.max(tl.where(force_match, offs, -1), 0)
                )
            block_hi = block_lo
    else:
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = tl.maximum(last_end, i)

            implicit_offset = 0
            for marker in tl.static_range(len(IMPLICIT_END_LENS)):
                implicit_len = IMPLICIT_END_LENS[marker]
                if i + implicit_len <= total_len:
                    implicit_match = True
                    for j in tl.static_range(IMPLICIT_END_LENS[marker]):
                        expected = tl.load(
                            implicit_reasoning_end_token_ids_ptr + implicit_offset + j
                        )
                        actual = tl.load(
                            all_token_ids_ptr
                            + req_state_idx * all_token_ids_stride
                            + i
                            + j
                        )
                        implicit_match = implicit_match & (actual == expected)
                    if implicit_match:
                        last_end = tl.maximum(last_end, i)
                implicit_offset += implicit_len

            if len(IMPLICIT_END_LENS) and i + END_LEN <= total_len:
                force_match = True
                for j in tl.static_range(END_LEN):
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    force_match = force_match & (actual == expected)
                if force_match:
                    last_end = tl.maximum(last_end, i + END_LEN - 1)

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    implicit_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    IMPLICIT_END_LENS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    start_lo = total_len - START_LEN + 1
    if start_lo < 0:
        start_lo = 0
    for i in tl.range(start_lo, effective_len - START_LEN + 1):
        start_match = True
        for j in tl.static_range(0, START_LEN):
            expected = tl.load(reasoning_start_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            start_match = start_match & (actual == expected)
        if start_match:
            last_start = i

    end_lo = total_len - NATURAL_END_LEN + 1
    if end_lo < 0:
        end_lo = 0
    for i in tl.range(end_lo, effective_len - NATURAL_END_LEN + 1):
        end_match = True
        for j in tl.static_range(0, NATURAL_END_LEN):
            expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            end_match = end_match & (actual == expected)
        if end_match:
            last_end = tl.maximum(last_end, i)

    # Draft-prefix boundaries affect only this verification row. Persistent
    # marker caches are updated exclusively from committed tokens above.
    implicit_offset = 0
    for marker in tl.static_range(len(IMPLICIT_END_LENS)):
        implicit_len = IMPLICIT_END_LENS[marker]
        implicit_lo = tl.maximum(0, total_len - implicit_len + 1)
        for i in tl.range(implicit_lo, effective_len - implicit_len + 1):
            implicit_match = True
            for j in tl.static_range(IMPLICIT_END_LENS[marker]):
                expected = tl.load(
                    implicit_reasoning_end_token_ids_ptr + implicit_offset + j
                )
                actual = _load_effective_token(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    i + j,
                )
                implicit_match = implicit_match & (actual == expected)
            if implicit_match:
                last_end = tl.maximum(last_end, i)
        implicit_offset += implicit_len

    if len(IMPLICIT_END_LENS):
        # Include completed forced ends that overlap a possible suffix prefix.
        # Otherwise self-overlapping markers could restart forcing indefinitely.
        force_lo = tl.maximum(0, total_len - 2 * END_LEN + 2)
        latest_forced_end = -1
        for i in tl.range(force_lo, effective_len - END_LEN + 1):
            force_match = _effective_sequence_matches(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                reasoning_end_token_ids_ptr,
                0,
                i,
                END_LEN,
            )
            if force_match:
                last_end = tl.maximum(last_end, i + END_LEN - 1)
                latest_forced_end = i + END_LEN - 1

        forced_prefix_len = 0
        for prefix_len in tl.static_range(1, END_LEN):
            if effective_len >= prefix_len:
                suffix_start = effective_len - prefix_len
                prefix_match = _effective_sequence_matches(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    reasoning_end_token_ids_ptr,
                    0,
                    suffix_start,
                    prefix_len,
                )
                if prefix_match and suffix_start > latest_forced_end:
                    active_start = _active_reasoning_start_before(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start,
                        reasoning_start_token_ids_ptr,
                        natural_reasoning_end_token_ids_ptr,
                        reasoning_end_token_ids_ptr,
                        implicit_reasoning_end_token_ids_ptr,
                        START_LEN,
                        NATURAL_END_LEN,
                        END_LEN,
                        IMPLICIT_END_LENS,
                    )
                    if (
                        active_start >= 0
                        and suffix_start >= active_start + START_LEN + budget
                    ):
                        forced_prefix_len = prefix_len
        if forced_prefix_len:
            force_token_id = tl.load(reasoning_end_token_ids_ptr + forced_prefix_len)
            tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)
            return

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    implicit_offset = 0
    budget_end = reasoning_start + budget
    for marker in tl.static_range(len(IMPLICIT_END_LENS)):
        for prefix_len in tl.static_range(1, IMPLICIT_END_LENS[marker]):
            suffix_start = effective_len - prefix_len
            if (
                suffix_start >= reasoning_start
                and suffix_start < budget_end
                and effective_len < budget_end + IMPLICIT_END_LENS[marker] - 1
            ):
                prefix_match = _effective_sequence_matches(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    implicit_reasoning_end_token_ids_ptr,
                    implicit_offset,
                    suffix_start,
                    prefix_len,
                )
                if prefix_match:
                    return
        implicit_offset += IMPLICIT_END_LENS[marker]

    # If the tail already ends with a prefix of the forced end sequence
    # (even from a resumed prompt), continue from the next marker token.
    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len < max_prefix_len:
        max_prefix_len = effective_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start + j,
                    )
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def apply_thinking_budget(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
    implicit_reasoning_end_token_ids: torch.Tensor | None = None,
    implicit_reasoning_end_lengths: tuple[int, ...] | None = None,
) -> None:
    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]
    if implicit_reasoning_end_lengths is None:
        implicit_reasoning_end_lengths = (
            ()
            if implicit_reasoning_end_token_ids is None
            else (1,) * implicit_reasoning_end_token_ids.shape[0]
        )
    if implicit_reasoning_end_token_ids is None:
        implicit_reasoning_end_token_ids = natural_reasoning_end_token_ids

    _update_committed_marker_cache_kernel[(req_ids.shape[0],)](
        req_ids,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        implicit_reasoning_end_token_ids,
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        IMPLICIT_END_LENS=implicit_reasoning_end_lengths,
        END_LEN=end_len,
        MAX_LEN=max(
            start_len,
            natural_end_len,
            end_len if implicit_reasoning_end_lengths else 0,
            *implicit_reasoning_end_lengths,
        ),
        BLOCK=_COLD_SCAN_BLOCK,
    )

    _thinking_budget_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
        implicit_reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
        IMPLICIT_END_LENS=implicit_reasoning_end_lengths,
    )
