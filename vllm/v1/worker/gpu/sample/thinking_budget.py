# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

_INT32_MAX = np.iinfo(np.int32).max
_COLD_SCAN_BLOCK = 1024

# Sentinel budget for requests tracked only for reasoning loop breaking: the
# largest value the int32 budget tensor holds, so the budget countdown never
# trips and only a loop detection can force the end sequence. It also keeps the
# request inside the marker-cache and forcing kernels, which both skip a
# negative budget.
_LOOP_BREAK_ONLY_BUDGET = _INT32_MAX

# Per-request loop-break state, held on device in a single int32 tensor. The
# detection kernel flips ARMED to 1 (fired) and back; only the host writes OFF.
_LOOP_BREAK_OFF = -1
_LOOP_BREAK_ARMED = 0


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
        self.loop_break_enabled = False
        # ``reasoning_end_str`` may prepend a transition phrase to the parser's
        # own marker, in which case forcing writes a longer sequence than a
        # natural exit and both have to close a section. When they are equal the
        # natural scan already covers both, so the extra one is skipped.
        self.track_forced_end = end_ids != natural_end_ids
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

        self._init_loop_break(reasoning_config)

    def _init_loop_break(self, reasoning_config: "ReasoningConfig | None") -> None:
        """Set up reasoning-scoped loop breaking, if the server configured it.

        Detection semantics mirror ``check_sequence_repetition``, which rejects
        a degenerate parameter set rather than firing on it; the equivalent
        rejection here has to happen on the host, because a ``min_count`` below
        2 would make the on-device tail comparison vacuously true.
        """
        max_pattern = getattr(reasoning_config, "loop_break_max_pattern_size", 0)
        min_pattern = getattr(reasoning_config, "loop_break_min_pattern_size", 0)
        min_count = getattr(reasoning_config, "loop_break_min_count", 0)
        if min_pattern <= 0:
            min_pattern = 1
        if max_pattern <= 0 or min_count < 2 or min_pattern > max_pattern:
            return

        self.loop_break_enabled = True
        self.lb_min_pattern_size = min_pattern
        self.lb_max_pattern_size = max_pattern
        self.lb_min_count = min_count
        self.lb_min_reasoning_tokens = getattr(
            reasoning_config, "loop_break_min_reasoning_tokens", 256
        )
        self.lb_check_interval = max(
            1, getattr(reasoning_config, "loop_break_check_interval", 16)
        )

        self.use_loop_break = np.zeros(self.max_num_reqs, dtype=bool)
        # -1 off, 0 armed, 1 fired. Written per request on the host and flipped
        # on device by the detection kernel.
        self.loop_break_fired = torch.full(
            (self.max_num_reqs,), _LOOP_BREAK_OFF, dtype=torch.int32, device=self.device
        )
        # Reasoning-section length at the last detection run, so a check costs
        # nothing until ``loop_break_check_interval`` more tokens are accepted.
        self.loop_break_last_check = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._lb_reset_reqs: list[int] = []
        self._lb_reset_vals: list[int] = []

    def _loop_break_active_for(self, sampling_params: SamplingParams) -> bool:
        """Server-configured loop breaking, minus a per-request opt-out.

        ``thinking_loop_break=False`` opts a request out; ``None`` (the default)
        follows the server configuration. ``True`` cannot enable the feature on
        a server that has not configured it, because the detection parameters
        live in ``ReasoningConfig``.
        """
        if not self.loop_break_enabled:
            return False
        override = sampling_params.thinking_loop_break
        if override is None:
            return True
        return bool(override)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        if not self.enabled:
            return
        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = budget is not None
        loop_break = self._loop_break_active_for(sampling_params)
        if budget is None:
            budget = _LOOP_BREAK_ONLY_BUDGET if loop_break else -1
        else:
            budget = min(budget, _INT32_MAX)
        if budget >= 0:
            self._reset_reqs.append(req_idx)
        if self.loop_break_enabled:
            # Always staged, including the opt-out: the slot may still hold a
            # fired flag from the request that previously occupied it.
            self.use_loop_break[req_idx] = loop_break
            self._lb_reset_reqs.append(req_idx)
            self._lb_reset_vals.append(
                _LOOP_BREAK_ARMED if loop_break else _LOOP_BREAK_OFF
            )
        if self.thinking_token_budget.np[req_idx] != budget:
            self.thinking_token_budget.np[req_idx] = budget
            self._budget_dirty = True

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
        if self.loop_break_enabled and self._lb_reset_reqs:
            idx = async_tensor_h2d(
                self._lb_reset_reqs, dtype=torch.int64, device=self.device
            )
            vals = async_tensor_h2d(
                self._lb_reset_vals, dtype=torch.int32, device=self.device
            )
            self.loop_break_fired.index_copy_(0, idx, vals)
            self.loop_break_last_check.index_fill_(0, idx, 0)
            self._lb_reset_reqs.clear()
            self._lb_reset_vals.clear()
        if self._budget_dirty:
            self.thinking_token_budget.copy_to_uva()
            self._budget_dirty = False

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        active = self.use_thinking_budget[idx_mapping_np]
        if self.loop_break_enabled:
            active = active | self.use_loop_break[idx_mapping_np]
        if not np.any(active):
            return

        apply_thinking_budget(
            logits,
            idx_mapping,
            expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
            track_forced_end=self.track_forced_end,
            loop_break_fired=(
                self.loop_break_fired if self.loop_break_enabled else None
            ),
            loop_break_last_check=(
                self.loop_break_last_check if self.loop_break_enabled else None
            ),
            loop_break_min_pattern_size=(
                self.lb_min_pattern_size if self.loop_break_enabled else 0
            ),
            loop_break_max_pattern_size=(
                self.lb_max_pattern_size if self.loop_break_enabled else 0
            ),
            loop_break_min_count=(self.lb_min_count if self.loop_break_enabled else 0),
            loop_break_min_reasoning_tokens=(
                self.lb_min_reasoning_tokens if self.loop_break_enabled else 0
            ),
            loop_break_check_interval=(
                self.lb_check_interval if self.loop_break_enabled else 1
            ),
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
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    TRACK_FORCED_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Track the last reasoning start and the last reasoning end.

    A section ends on either sequence: the parser's natural marker, or the
    sequence forcing writes when ``reasoning_end_str`` carries a transition
    phrase and so differs from it. Missing the forced one leaves the section
    open after forcing completes, and forcing then restarts forever.
    ``TRACK_FORCED_END`` is off when the two sequences are equal, where the
    natural scan already covers both.
    """
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
        # block with a marker; only the relative order of the start and end
        # positions found matters below.
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

            found_end = tl.max(tl.where(end_match, offs, -1), axis=0)
            if TRACK_FORCED_END:
                forced_match = (offs < block_hi) & (offs + END_LEN <= total_len)
                for j in tl.static_range(0, END_LEN):
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + offs
                        + j,
                        mask=offs + j < total_len,
                        other=-1,
                    )
                    forced_match = forced_match & (actual == expected)
                found_forced = tl.max(tl.where(forced_match, offs, -1), axis=0)
                if found_forced > found_end:
                    found_end = found_forced

            last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
            last_end = found_end
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
                    last_end = i

            if TRACK_FORCED_END:
                # Order between the two end sequences does not matter: within one
                # position both assign the same i, and across positions the
                # largest matching i is assigned last -- the max the V1 holder
                # takes over the same two searches.
                forced_match = i + END_LEN <= total_len
                for j in tl.static_range(0, END_LEN):
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + i
                        + j,
                        mask=i + j < total_len,
                        other=-1,
                    )
                    forced_match = forced_match & (actual == expected)
                if forced_match:
                    last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _loop_break_detect_kernel(
    req_ids_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    loop_break_fired_ptr,
    loop_break_last_check_ptr,
    START_LEN: tl.constexpr,
    MIN_PATTERN: tl.constexpr,
    MAX_PATTERN: tl.constexpr,
    MIN_COUNT: tl.constexpr,
    MIN_REASONING_TOKENS: tl.constexpr,
    CHECK_INTERVAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flag requests whose open reasoning section ends in a repeating pattern.

    Runs on committed tokens only -- the same tokens the V1 detection sees --
    after ``_update_committed_marker_cache_kernel`` has refreshed the section
    boundary, and before ``_thinking_budget_kernel`` reads the flag.
    """
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    fired = tl.load(loop_break_fired_ptr + req_state_idx)
    if fired < 0:
        # Loop breaking is off for this request.
        return

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)
    if last_start < 0 or last_start <= last_end:
        # No reasoning section is open; re-arm for the next one. This is also
        # what clears the flag once the forced end sequence lands in the
        # committed tokens.
        tl.store(loop_break_fired_ptr + req_state_idx, 0)
        tl.store(loop_break_last_check_ptr + req_state_idx, 0)
        return
    if fired > 0:
        # Already forcing. Under speculative decoding a forced end token can be
        # rejected, so the flag stays set and _thinking_budget_kernel
        # re-asserts the end sequence every step until the section closes.
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    think_len = total_len - last_start - START_LEN
    if think_len < MIN_REASONING_TOKENS:
        return

    last_check = tl.load(loop_break_last_check_ptr + req_state_idx)
    if think_len < last_check:
        # The section shrank behind the checkpoint (a rewind, or a new section
        # that starts further right); re-arm rather than stall the interval.
        last_check = 0
    if think_len - last_check < CHECK_INTERVAL:
        return
    tl.store(loop_break_last_check_ptr + req_state_idx, think_len)

    # The tail comparison never reaches back further than
    # max_pattern_size * min_count, and is clamped to the section start so a
    # pattern cannot span a previous section or the prompt.
    avail = MAX_PATTERN * MIN_COUNT
    if think_len < avail:
        avail = think_len

    offs = tl.arange(0, BLOCK)
    found = 0
    for pattern_len in tl.range(MIN_PATTERN, MAX_PATTERN + 1):
        # ``pattern_len`` only grows, so skipping an oversized candidate is the
        # same as the host implementation's early return.
        if found == 0 and pattern_len * MIN_COUNT <= avail:
            mask = offs < pattern_len
            base = total_len - pattern_len + offs
            tail = tl.load(
                all_token_ids_ptr + req_state_idx * all_token_ids_stride + base,
                mask=mask,
                other=0,
            )
            mismatches = 0
            for m in tl.static_range(1, MIN_COUNT):
                prev = tl.load(
                    all_token_ids_ptr
                    + req_state_idx * all_token_ids_stride
                    + base
                    - pattern_len * m,
                    mask=mask,
                    other=0,
                )
                mismatches += tl.sum(tl.where(mask & (prev != tail), 1, 0), axis=0)
            if mismatches == 0:
                found = 1

    if found == 1:
        tl.store(loop_break_fired_ptr + req_state_idx, 1)


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
    loop_break_fired_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    TRACK_FORCED_END: tl.constexpr,
    HAS_LOOP_BREAK: tl.constexpr,
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
            last_end = i

    if TRACK_FORCED_END:
        # The forced sequence can also complete among the speculative positions,
        # and the ones after it belong to the answer. Take the max rather than
        # assigning, since the natural scan above may already have found a later
        # end within this same window.
        forced_lo = total_len - END_LEN + 1
        if forced_lo < 0:
            forced_lo = 0
        for i in tl.range(forced_lo, effective_len - END_LEN + 1):
            forced_match = True
            for j in tl.static_range(0, END_LEN):
                expected = tl.load(reasoning_end_token_ids_ptr + j)
                actual = _load_effective_token(
                    all_token_ids_ptr,
                    all_token_ids_stride,
                    input_ids_ptr,
                    cur_req_first_pos,
                    req_state_idx,
                    total_len,
                    i + j,
                )
                forced_match = forced_match & (actual == expected)
            if forced_match:
                last_end = tl.maximum(last_end, i)

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        # The budget is not exhausted, so only a detected reasoning loop can
        # force the end sequence here. Both paths share every line below, and
        # therefore share the rejected-end continuation and the multi-token
        # marker handling. The load has to sit inside the constexpr guard: an
        # early return does not stop Triton from compiling what follows it, so
        # a null pointer would still be dereferenced at codegen time.
        fired = 0
        if HAS_LOOP_BREAK:
            fired = tl.load(loop_break_fired_ptr + req_state_idx)
        if fired <= 0:
            return

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
    track_forced_end: bool = True,
    loop_break_fired: torch.Tensor | None = None,
    loop_break_last_check: torch.Tensor | None = None,
    loop_break_min_pattern_size: int = 0,
    loop_break_max_pattern_size: int = 0,
    loop_break_min_count: int = 0,
    loop_break_min_reasoning_tokens: int = 0,
    loop_break_check_interval: int = 1,
) -> None:
    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]

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
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
        # The re-scan overlap has to cover the longest sequence being matched,
        # or a forced end split across two decode steps is never seen.
        MAX_LEN=max(start_len, natural_end_len, end_len if track_forced_end else 0),
        TRACK_FORCED_END=track_forced_end,
        BLOCK=_COLD_SCAN_BLOCK,
    )

    if loop_break_fired is not None:
        _loop_break_detect_kernel[(req_ids.shape[0],)](
            req_ids,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            cached_last_start,
            cached_last_end,
            loop_break_fired,
            loop_break_last_check,
            START_LEN=start_len,
            MIN_PATTERN=loop_break_min_pattern_size,
            MAX_PATTERN=loop_break_max_pattern_size,
            MIN_COUNT=loop_break_min_count,
            MIN_REASONING_TOKENS=loop_break_min_reasoning_tokens,
            CHECK_INTERVAL=loop_break_check_interval,
            BLOCK=triton.next_power_of_2(loop_break_max_pattern_size),
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
        loop_break_fired,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
        TRACK_FORCED_END=track_forced_end,
        HAS_LOOP_BREAK=loop_break_fired is not None,
    )
