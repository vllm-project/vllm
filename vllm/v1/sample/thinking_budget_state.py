# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch thinking token budget state; applied after penalties at sample time."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sampling_params import RepetitionDetectionParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.sched.utils import check_sequence_repetition
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

logger = init_logger(__name__)

# Sentinel budget for requests tracked only for reasoning loop breaking: large
# enough that the budget countdown never trips, so the existing budget state
# machine runs unchanged and only a loop detection can force the end sequence.
_LOOP_BREAK_ONLY_BUDGET = 1 << 62


def maybe_create_thinking_budget_state_holder(
    reasoning_config: "ReasoningConfig | None",
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
) -> "ThinkingBudgetStateHolder | None":
    if reasoning_config is None:
        return None
    return ThinkingBudgetStateHolder(
        reasoning_config, max_num_seqs, num_spec_tokens, device, is_pin_memory
    )


class ThinkingBudgetStateHolder:
    """Tracks thinking sections and forces end tokens when budget is exceeded."""

    think_start_token_ids: list[int]
    think_end_token_ids: list[int]
    natural_think_end_token_ids: list[int]

    def __init__(
        self,
        reasoning_config: "ReasoningConfig | None",
        max_num_seqs: int,
        num_spec_tokens: int,
        device: torch.device,
        is_pin_memory: bool,
    ):
        _ = is_pin_memory  # API parity with logits processors
        max_num_reqs = max_num_seqs
        self.in_spec_mode = num_spec_tokens > 0
        self.num_spec_tokens = num_spec_tokens

        # No separate enable flag: a non-``None`` ``reasoning_config`` is the switch.
        self.is_enabled = reasoning_config is not None

        if reasoning_config is None:
            self.think_start_token_ids = []
            self.think_end_token_ids = []
            self.natural_think_end_token_ids = []
        else:
            rs = reasoning_config.reasoning_start_token_ids
            re = reasoning_config.reasoning_end_token_ids
            natural_re = getattr(
                reasoning_config, "natural_reasoning_end_token_ids", None
            )
            self.think_start_token_ids = rs if rs else []
            self.think_end_token_ids = re if re else []
            # ``reasoning_end_str`` may prepend a transition phrase to the
            # parser's own end marker, so a natural exit emits a shorter
            # sequence than the one forcing writes.
            self.natural_think_end_token_ids = (
                natural_re if natural_re and natural_re != re else []
            )

        self.loop_break_params: RepetitionDetectionParams | None = None
        self.loop_break_min_reasoning_tokens = 256
        self.loop_break_check_interval = 16
        if (
            reasoning_config is not None
            and getattr(reasoning_config, "loop_break_max_pattern_size", 0) > 0
        ):
            # __post_init__ validates the combination at engine-init time.
            self.loop_break_params = RepetitionDetectionParams(
                max_pattern_size=reasoning_config.loop_break_max_pattern_size,
                min_pattern_size=reasoning_config.loop_break_min_pattern_size,
                min_count=reasoning_config.loop_break_min_count,
            )
            self.loop_break_min_reasoning_tokens = getattr(
                reasoning_config, "loop_break_min_reasoning_tokens", 256
            )
            self.loop_break_check_interval = max(
                1, getattr(reasoning_config, "loop_break_check_interval", 16)
            )

        self.device = device
        self._state: dict[int, dict[str, Any]] = {}
        self.cu_num_tokens: dict[int, int] = {}

        if self.num_spec_tokens > 0:
            self._mask_capacity = max_num_reqs * (self.num_spec_tokens + 1)
        else:
            self._mask_capacity = max_num_reqs

    def has_tracked_requests(self) -> bool:
        """True when ``sync_batch`` has state for a ``thinking_token_budget`` row.

        Used to decide whether sampling needs output-token rows and spec combining;
        distinct from merely having a holder instance (reasoning may be on with no
        budgeted requests in this batch).
        """
        return bool(self._state)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Add/remove/move per-request state only (no _update_think_state)."""
        if not self.is_enabled or not batch_update:
            return
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            thinking_token_budget = params.thinking_token_budget
            loop_break = self._loop_break_active_for(params)
            if thinking_token_budget is not None or loop_break:
                effective_budget = (
                    thinking_token_budget
                    if thinking_token_budget is not None
                    else _LOOP_BREAK_ONLY_BUDGET
                )
                entry = self._init_state_entry(prompt_tok_ids, effective_budget)
                entry["output_tok_ids"] = output_tok_ids
                entry["spec_token_ids"] = []
                entry["loop_break"] = loop_break
                if loop_break:
                    entry["lb_scan_pos"] = 0
                    entry["lb_in_think"] = entry["in_think"]
                    # None: the section began in the prompt, so every output
                    # token so far belongs to it.
                    entry["lb_section_begin"] = None
                    entry["lb_think_len"] = entry["think_count"]
                    entry["lb_last_check_len"] = 0
                    entry["lb_fired"] = False
                self._state[index] = entry
            else:
                self._state.pop(index, None)

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                state1 = self._state.pop(i1, None)
                state2 = self._state.pop(i2, None)
                if state1 is not None:
                    self._state[i2] = state1
                if state2 is not None:
                    self._state[i1] = state2
            else:
                state = self._state.pop(i1, None)
                if state is not None:
                    self._state[i2] = state

    def update_state(
        self,
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None,
        repeat_indices: torch.Tensor | None = None,
    ) -> None:
        """Refresh output/spec from sampling rows and recompute think state."""
        if not self.is_enabled or not self._state:
            return

        spec_lists = spec_token_ids or []
        last_row_for_req: dict[int, int] | None = None
        if repeat_indices is not None:
            last_row_for_req = {}
            rpt = repeat_indices.cpu().tolist()
            for batch_row, req_i in enumerate(rpt):
                last_row_for_req[req_i] = batch_row

        for seq_idx, state in list(self._state.items()):
            if last_row_for_req is not None:
                output_row: int | None = last_row_for_req.get(seq_idx)
                if output_row is None or output_row >= len(output_token_ids):
                    continue
                state["output_tok_ids"] = output_token_ids[output_row]
            elif seq_idx >= len(output_token_ids):
                continue
            else:
                state["output_tok_ids"] = output_token_ids[seq_idx]
            if seq_idx < len(spec_lists):
                state["spec_token_ids"] = list(spec_lists[seq_idx])
            else:
                state["spec_token_ids"] = []
            state["in_spec_mode"] = self.in_spec_mode
            state["force_index"] = []
            self._update_think_state(state)
            if state.get("loop_break"):
                self._update_loop_break_state(state)

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        """Mask and bump logits for forced end-of-thinking tokens."""
        if not self.is_enabled or not self._state:
            return logits
        spec_lists = spec_token_ids or []
        return self._apply_forcing_to_logits(logits, predict_bonus_token, spec_lists)

    def _loop_break_active_for(self, params: Any) -> bool:
        """Server-configured loop breaking, minus a per-request opt-out.

        ``thinking_loop_break=False`` opts a request out; ``None`` (the
        default) follows the server configuration. ``True`` cannot enable
        loop breaking on a server that has not configured it, because the
        detection parameters live in ``ReasoningConfig``.
        """
        if self.loop_break_params is None:
            return False
        override = getattr(params, "thinking_loop_break", None)
        if override is None:
            return True
        return bool(override)

    def _update_loop_break_state(self, state: dict[str, Any]) -> None:
        """Track the current reasoning section and break exact loops.

        Runs after ``_update_think_state`` so a detection this step cannot be
        undone by the budget machinery's rejected-end recovery, and keeps its
        own lightweight section tracking (``lb_*`` keys) because the budget
        path skips section bookkeeping while comfortably under budget.

        On detection the request is flipped into the exact ``in_end`` state
        the budget-exhaustion transition produces, so all forced-end
        enforcement (spec decode, bonus-token double call, platform paths)
        is shared with the thinking-budget feature.
        """
        output = state.get("output_tok_ids", [])
        current_length = len(output)
        scan_pos = state.get("lb_scan_pos", 0)
        if current_length <= scan_pos:
            return

        start_ids = self.think_start_token_ids
        end_ids = self.think_end_token_ids
        natural_end_ids = self.natural_think_end_token_ids
        max_marker = max(len(start_ids), len(end_ids), len(natural_end_ids), 1)
        # Overlap by marker-length - 1 so a marker sequence spanning the
        # previous scan edge is still seen.
        window_begin = max(0, scan_pos - (max_marker - 1))
        window = output[window_begin:]
        last_start = self._find_last_sequence_index(window, start_ids)
        # A natural exit emits only the parser's end marker, which is a
        # shorter sequence than ``reasoning_end_str`` when that carries a
        # transition phrase; miss it and answer tokens keep counting as
        # reasoning, so a repetitive answer can force another end sequence.
        last_end = max(
            self._find_last_sequence_index(window, end_ids),
            self._find_last_sequence_index(window, natural_end_ids),
        )

        if last_end > last_start:
            # The section ended (naturally or forced); re-arm for the next.
            state["lb_in_think"] = False
            state["lb_section_begin"] = None
            state["lb_think_len"] = 0
            state["lb_last_check_len"] = 0
            state["lb_fired"] = False
        elif last_start > last_end:
            state["lb_in_think"] = True
            state["lb_section_begin"] = window_begin + last_start + len(start_ids)
            state["lb_think_len"] = current_length - state["lb_section_begin"]
        elif state.get("lb_in_think"):
            state["lb_think_len"] = state.get("lb_think_len", 0) + (
                current_length - scan_pos
            )
        state["lb_scan_pos"] = current_length

        if state.get("lb_fired", False):
            # Keep forcing until the end sequence actually lands: under spec
            # decode a forced end token can be rejected, and the budget
            # machinery's rejected-end recovery then flips the request back
            # to in_think. Re-assert the forced end every step while the
            # section is still open; the tracker above clears ``lb_fired``
            # once the end sequence appears in the accepted output.
            if state.get("lb_in_think") and not state.get("in_end", False):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["bonus_token_forced"] = False
                state["force_index"] = [0]
            return

        if not state.get("lb_in_think") or state.get("in_end", False):
            return
        think_len = state.get("lb_think_len", 0)
        if think_len < self.loop_break_min_reasoning_tokens:
            return
        if (
            think_len - state.get("lb_last_check_len", 0)
            < self.loop_break_check_interval
        ):
            return
        state["lb_last_check_len"] = think_len

        params = self.loop_break_params
        assert params is not None
        # ``check_sequence_repetition`` anchors at the sequence end and never
        # indexes back more than max_pattern_size * min_count tokens; slice
        # just that tail, clamped to the section start so a pattern cannot
        # span into the prompt or a previous section.
        need = params.max_pattern_size * params.min_count
        section_begin = state.get("lb_section_begin") or 0
        tail_begin = max(section_begin, current_length - need)
        if check_sequence_repetition(output[tail_begin:], params):
            state["lb_fired"] = True
            state["in_think"] = False
            state["in_end"] = True
            state["end_count"] = 0
            state["bonus_token_forced"] = False
            state["force_index"] = [0]
            logger.info(
                "Breaking a repeating reasoning loop after %d reasoning "
                "tokens; forcing the reasoning end sequence.",
                think_len,
            )

    def _scan_markers(self, state: dict[str, Any]) -> None:
        """Locate the reasoning markers, examining each output token once.

        ``start_thinking``/``end_thinking`` are sticky until the section is
        reset, and a marker that is absent from the tokens seen so far stays
        absent, so only tokens appended since the last call need searching.
        Rescanning ``output[scan_offset:]`` every step would cost O(n) per
        token while a section is open -- O(n^2) over a long reasoning trace.
        """
        output = state.get("output_tok_ids", [])
        scan_offset = state.get("scan_offset", 0)
        if state["start_thinking"] == -1 or state["end_thinking"] == -1:
            overlap = (
                max(
                    len(self.think_start_token_ids),
                    len(self.think_end_token_ids),
                    len(self.natural_think_end_token_ids),
                    1,
                )
                - 1
            )
            # ``scan_offset`` wins after a section reset, which moves it past
            # the cursor; the overlap keeps a marker straddling the previous
            # window edge visible.
            window_begin = max(scan_offset, state.get("marker_scan_pos", 0) - overlap)
            window = output[window_begin:]
            if state["start_thinking"] == -1:
                found = self._find_last_sequence_index(
                    window, self.think_start_token_ids
                )
                if found >= 0:
                    state["start_thinking"] = window_begin + found
            if state["end_thinking"] == -1:
                found = self._find_last_end_index(window)
                if found >= 0:
                    state["end_thinking"] = window_begin + found
        state["marker_scan_pos"] = len(output)

    def _find_last_end_index(self, target_list: list[int]) -> int:
        """Start of the last reasoning end, whether forced or natural.

        ``reasoning_end_str`` may carry a transition phrase before the parser's
        own marker, so a section the model closed itself ends with the shorter
        natural sequence and a search for the forced one alone misses it.
        """
        return max(
            self._find_last_sequence_index(target_list, self.think_end_token_ids),
            self._find_last_sequence_index(
                target_list, self.natural_think_end_token_ids
            ),
        )

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    def _init_state_entry(
        self, prompt_tok_ids: list[int] | None, thinking_token_budget: int
    ) -> dict[str, Any]:
        if prompt_tok_ids is None:
            last_start = -1
            last_end = -1
            in_think = False
            think_count = 0
            start_thinking = -1
            countdown = thinking_token_budget
            continue_thinking = False
            in_end = False
        else:
            start_thinking = -1
            countdown = thinking_token_budget
            continue_thinking = False
            in_end = False
            last_start = self._find_last_sequence_index(
                prompt_tok_ids, self.think_start_token_ids
            )
            last_end = self._find_last_end_index(prompt_tok_ids)
            in_think = last_start > last_end
            # load metrics such as think count, start thinking
            # if request is in thinking mode, already
            if in_think:
                think_count = len(prompt_tok_ids) - (
                    last_start + len(self.think_start_token_ids)
                )
                start_thinking = len(prompt_tok_ids) - think_count - 1
                countdown -= think_count
                continue_thinking = True
                # check if the token is exhausted within prompt
                token_exhausted = thinking_token_budget - think_count
                in_end = token_exhausted <= 0
            else:
                think_count = 0

        return {
            "in_think": in_think,
            "in_end": in_end,
            "check_count_down": countdown,
            "think_count": think_count,
            "end_count": 0,
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget": thinking_token_budget,
            "prev_output_length": 0,
            "spec_token_ids": [],
            "force_index": [],
            "start_thinking": start_thinking,
            "end_thinking": -1,
            "in_spec_mode": False,
            "bonus_token_forced": False,
            "continue_thinking": continue_thinking,
            "scan_offset": 0,
            "marker_scan_pos": 0,
        }

    def _update_think_state(self, state: dict[str, Any]) -> None:
        if state.get("thinking_token_budget", -1) == -1:
            return
        if len(self.think_end_token_ids) == 0:
            state["thinking_token_budget"] = -1
            state["in_end"] = False
            state["force_index"] = []
            return

        self._scan_markers(state)

        if (
            not state.get("in_end", False)
            and state["start_thinking"] >= 0
            and state["end_thinking"] >= 0
            and state["end_thinking"] > state["start_thinking"]
            and not state.get("continue_thinking", False)
        ):
            state["in_think"] = False
            state["think_count"] = 0
            state["continue_thinking"] = False
            state["start_thinking"] = -1
            state["end_thinking"] = -1
            state["scan_offset"] = len(state.get("output_tok_ids", []))
            state["check_count_down"] = state["thinking_token_budget"]
            return

        if state["start_thinking"] == -1:
            return

        if state["continue_thinking"]:
            sampled_tokens_from_previous_step = len(
                state.get("output_tok_ids", [])
            ) - state.get("prev_output_length", 0)
        else:
            if state["prev_output_length"] == 0:
                sampled_tokens_from_previous_step = len(
                    state.get("output_tok_ids", [])
                ) - len(self.think_start_token_ids)
            else:
                sampled_tokens_from_previous_step = (
                    len(state.get("output_tok_ids", [])) - state["prev_output_length"]
                )
        current_step_countdown = (
            state["check_count_down"] - sampled_tokens_from_previous_step
        )
        predicted_countdown = current_step_countdown - len(state["spec_token_ids"]) - 1
        # We only proceed further if we have counted down the thinking budget
        # to 0 or less and when we are in the "in think" mode.
        # Exception: when continue_thinking=True and a natural </think> is
        # detected (end_thinking != -1), fall through to handle the exit —
        # even if the budget hasn't expired yet. For continue_thinking=False,
        # the early natural-end detection block above already handles it.
        natural_end_with_continue = (
            state.get("continue_thinking", False) and state["end_thinking"] != -1
        )
        if (
            not state.get("in_end", False)
            and predicted_countdown >= 0
            and state["start_thinking"] > -1
            and not natural_end_with_continue
        ):
            state["check_count_down"] = current_step_countdown
            state["prev_output_length"] = len(state.get("output_tok_ids", []))
            return
        output = state.get("output_tok_ids", [])
        if not output:
            # When in_end was set at init (budget=0, prompt already in think),
            # we must force the first generated token to be the end token;
            # otherwise apply() sees in_end=True but force_index=[] and
            # allows an extra thinking token.
            if state.get("in_end", False):
                state["force_index"] = [0]
            return

        # Track previous output length for incremental processing
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            if state.get("in_end", False):
                remaining_budget = state["thinking_token_budget"] - state["think_count"]
                spec_len = len(state["spec_token_ids"])
                if spec_len > 0:
                    if 0 < remaining_budget < spec_len:
                        state["force_index"] = [remaining_budget]
                    elif remaining_budget <= 0:
                        state["force_index"] = [0]
                    else:
                        state["force_index"] = [spec_len]
                else:
                    state["force_index"] = [0]
            return

        state["prev_output_length"] = current_length

        start_len = len(self.think_start_token_ids)
        absolute_start_pos = state["start_thinking"]

        if state["continue_thinking"] and state["end_thinking"] > -1:
            absolute_end_pos = state["end_thinking"] + len(
                state.get("prompt_tok_ids") or []
            )
        else:
            absolute_end_pos = state["end_thinking"]
        # Update state based on recent sequences
        # This is the case where we are in end mode, but the rejection sampler
        # rejected a token before the end token,
        # so we need to go back to think mode and wait for the next end token
        # eg with 999: [2,4,5,999] -> [3,-1,-1,-1]
        if state["in_end"] and state["end_count"] == 0:
            new_tokens = output[prev_length:]
            stopping_thinking = (
                self.think_end_token_ids[state["end_count"]] in new_tokens
            )
            if not stopping_thinking:
                state["in_think"] = True
                state["in_end"] = False
                state["end_count"] = 0
                state["bonus_token_forced"] = False

        if not state["in_end"]:
            if absolute_start_pos >= 0 and absolute_end_pos >= 0:
                # Case: ...<end>...<start>... - entering think mode
                if absolute_start_pos > absolute_end_pos:
                    new_think_count = current_length - (absolute_start_pos + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # Case: ...<start>...<end>... - exiting think mode
                    state["in_think"] = False
                    state["think_count"] = 0
                    state["continue_thinking"] = False
                    state["start_thinking"] = -1
                    state["end_thinking"] = -1
                    state["scan_offset"] = len(state.get("output_tok_ids", []))

            elif absolute_start_pos >= 0 and not state["continue_thinking"]:
                # Found think start - entering think mode
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count

            elif absolute_end_pos >= 0:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0
                state["continue_thinking"] = False
                state["start_thinking"] = -1
                state["end_thinking"] = -1
                state["scan_offset"] = len(state.get("output_tok_ids", []))

            elif state["in_think"]:
                # Continue thinking mode, increment count by new tokens
                prompt_tok_ids = state.get("prompt_tok_ids") or []
                think_tokens_in_prompt = len(prompt_tok_ids) - (
                    absolute_start_pos + start_len
                )
                state["think_count"] = (
                    len(state["output_tok_ids"]) + think_tokens_in_prompt
                )
            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = remaining_budget
            else:
                state["check_count_down"] = state["thinking_token_budget"]

            total_thinking_tokens = (
                state["think_count"] + len(state["spec_token_ids"]) + 1
            )
            # Check if need to transition to end mode
            # If we have more thinking tokens than the budget,
            # we need to transition to end mode
            if (
                state["in_think"]
                and total_thinking_tokens > state["thinking_token_budget"]
            ):
                # Calculate force_index: position within spec_token_ids where
                # forcing starts. If we're already over budget without spec
                # tokens, force from position 0. Force from the position
                # where budget is exceeded.
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
                remaining_budget = state["thinking_token_budget"] - state["think_count"]
                spec_len = len(state["spec_token_ids"])
                if 0 < remaining_budget < spec_len:
                    state["force_index"] = [remaining_budget]

                elif remaining_budget <= 0:
                    state["force_index"] = [0]

                else:
                    # remaining_budget >= spec_len: all spec tokens are within
                    # budget; force the bonus token position
                    state["force_index"] = [len(state["spec_token_ids"])]

        else:
            state["force_index"] = []
            if len(state["spec_token_ids"]) > 0:
                for i, token_id in enumerate(state["spec_token_ids"]):
                    if state["end_count"] + 1 < len(self.think_end_token_ids):
                        if token_id == self.think_end_token_ids[state["end_count"] + 1]:
                            state["end_count"] += 1
                        else:
                            state["end_count"] += 1
                            state["force_index"] = [i]
                            break
                    else:
                        state["end_count"] += 1
                if len(state["force_index"]) == 0:
                    state["end_count"] += 1
                    state["force_index"] = [len(state["spec_token_ids"])]
            else:
                state["end_count"] += 1
                state["force_index"] = [0]
            if state["end_count"] >= len(self.think_end_token_ids):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget"],
                        "start_thinking": -1,
                        "end_thinking": -1,
                        "think_count": 0,
                        "continue_thinking": False,
                        "scan_offset": len(state.get("output_tok_ids", [])),
                    }
                )

    def _apply_forcing_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> torch.Tensor:
        cumulative_total = 0
        self.cu_num_tokens.clear()

        n_layout = len(spec_token_ids_for_layout)
        if self._state:
            n_layout = max(n_layout, max(self._state.keys()) + 1)

        for index in range(n_layout):
            self.cu_num_tokens[index] = cumulative_total
            spec_tokens = (
                spec_token_ids_for_layout[index]
                if index < len(spec_token_ids_for_layout)
                else []
            )
            if self.in_spec_mode:
                cumulative_total += len(spec_tokens) if not predict_bonus_token else 1
            else:
                cumulative_total += 1

        # Build the active index / forced-token lists entirely on CPU so we
        # avoid per-iteration scalar sync writes to GPU tensors.
        active_indices_cpu: list[int] = []
        force_tokens_cpu: list[int] = []

        for seq_idx in sorted(self._state.keys()):
            if seq_idx not in self.cu_num_tokens:
                continue
            state = self._state[seq_idx]
            if state.get("in_end", False):
                # logits processor in spec mode are called twice
                # once for bonus token logits and
                # second time for the target logits
                # in case the force index is bonus token index
                # we change the force index to 0
                if predict_bonus_token:
                    if state.get("force_index") and state["force_index"][0] < len(
                        state["spec_token_ids"]
                    ):
                        continue
                    else:
                        state["force_index"] = [0]
                # continue enforcing the end thinking tokens
                if state["end_count"] > 0:
                    state["bonus_token_forced"] = False
                if state and not state["bonus_token_forced"]:
                    force_index = state.get("force_index", [])
                    if len(force_index) == 0:
                        continue
                    end_count = state.get("end_count", 0)
                    for force_idx in force_index:
                        if end_count < len(self.think_end_token_ids):
                            mask_idx = self.cu_num_tokens[seq_idx] + force_idx
                            if (
                                mask_idx < self._mask_capacity
                                and mask_idx < logits.shape[0]
                            ):
                                active_indices_cpu.append(mask_idx)
                                force_tokens_cpu.append(
                                    self.think_end_token_ids[end_count]
                                )
                            if predict_bonus_token:
                                if state["end_count"] > 0:
                                    state["bonus_token_forced"] = False
                                    state["force_index"] = []
                                else:
                                    state["bonus_token_forced"] = True

        if active_indices_cpu:
            device = logits.device
            if current_platform.is_rocm() and logits.is_contiguous():
                # Flattened index_fill avoids ROCm faults seen with 2-D
                # advanced-indexing writes on the thinking-budget path.
                vocab_size = logits.shape[1]
                flat_indices_cpu = [
                    row * vocab_size + token
                    for row, token in zip(active_indices_cpu, force_tokens_cpu)
                ]
                flat_indices = async_tensor_h2d(
                    flat_indices_cpu, dtype=torch.long, device=device
                )
                logits.view(-1).index_fill_(0, flat_indices, 1e9)
            elif current_platform.is_rocm():
                fill = logits.new_tensor(1e9)
                for row, token in zip(active_indices_cpu, force_tokens_cpu):
                    logits[row, token] = fill
            else:
                active_indices = async_tensor_h2d(
                    active_indices_cpu, dtype=torch.long, device=device
                )
                force_tokens = async_tensor_h2d(
                    force_tokens_cpu, dtype=torch.long, device=device
                )
                # Avoid CPU->GPU sync.
                fill = logits.new_full((len(active_indices_cpu),), 1e9)
                logits.index_put_((active_indices, force_tokens), fill)

        return logits
