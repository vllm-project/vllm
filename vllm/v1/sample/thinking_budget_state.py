# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch thinking token budget state; applied after penalties at sample time."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig
    from vllm.sampling_params import SamplingParams


def _should_track_request(params: "SamplingParams") -> bool:
    return (
        params.thinking_token_budget is not None
        or params.reasoning_eos_policy == "force_end"
    )


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
            self.implicit_end_token_ids: list[int] = []
        else:
            rs = reasoning_config.reasoning_start_token_ids
            re = reasoning_config.reasoning_end_token_ids
            implicit = getattr(
                reasoning_config, "implicit_reasoning_end_token_ids", None
            )
            self.think_start_token_ids = rs if rs else []
            self.think_end_token_ids = re if re else []
            self.implicit_end_token_ids = implicit if implicit else []

        self.device = device
        self._state: dict[int, dict[str, Any]] = {}
        self.cu_num_tokens: dict[int, int] = {}

        if self.num_spec_tokens > 0:
            self._mask_capacity = max_num_reqs * (self.num_spec_tokens + 1)
        else:
            self._mask_capacity = max_num_reqs

    def has_tracked_requests(self) -> bool:
        """True when ``sync_batch`` has state for a budget or force-end row.

        Used to decide whether sampling needs output-token rows and spec combining;
        distinct from merely having a holder instance (reasoning may be on with no
        tracked requests in this batch).
        """
        return bool(self._state)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Add/remove/move per-request state only (no _update_think_state)."""
        if not self.is_enabled or not batch_update:
            return
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            if _should_track_request(params):
                self._state[index] = self._init_state_entry(
                    prompt_tok_ids,
                    params.thinking_token_budget,
                    params.reasoning_eos_policy,
                    params.stop_token_ids_that_finish_request(),
                )
                self._state[index]["output_tok_ids"] = output_tok_ids
                self._state[index]["spec_token_ids"] = []
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
        self._prepare_cu_num_tokens(predict_bonus_token, spec_lists)
        self._apply_eos_policy_to_logits(logits, predict_bonus_token, spec_lists)
        return self._apply_forcing_to_logits(logits, predict_bonus_token, spec_lists)

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    def _find_last_end_index(self, tokens: list[int], scan_offset: int = 0) -> int:
        """Latest of forced/natural ``</think>`` and implicit ends (tool call)."""
        output_slice = tokens[scan_offset:]
        end = self._find_last_sequence_index(output_slice, self.think_end_token_ids)
        impl = (
            self._find_last_sequence_index(output_slice, self.implicit_end_token_ids)
            if self.implicit_end_token_ids
            else -1
        )
        chosen = max(end, impl)
        return chosen + scan_offset if chosen >= 0 else -1

    def _init_state_entry(
        self,
        prompt_tok_ids: list[int] | None,
        thinking_token_budget: int | None,
        reasoning_eos_policy: str = "stop",
        stop_token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        has_budget = thinking_token_budget is not None
        countdown = thinking_token_budget if has_budget else 0
        if prompt_tok_ids is None:
            last_start = -1
            last_end = -1
            in_think = False
            think_count = 0
            start_thinking = -1
            continue_thinking = False
            in_end = False
        else:
            start_thinking = -1
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
                if has_budget:
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
            "reasoning_eos_policy": reasoning_eos_policy,
            "stop_token_ids": stop_token_ids or [],
            "prev_output_length": 0,
            "spec_token_ids": [],
            "force_index": [],
            "start_thinking": start_thinking,
            "end_thinking": -1,
            "in_spec_mode": False,
            "bonus_token_forced": False,
            "continue_thinking": continue_thinking,
            "scan_offset": 0,
        }

    @staticmethod
    def _has_token_budget(state: dict[str, Any]) -> bool:
        budget = state.get("thinking_token_budget")
        return isinstance(budget, int) and budget >= 0

    @staticmethod
    def _reset_countdown(state: dict[str, Any]) -> None:
        budget = state.get("thinking_token_budget")
        state["check_count_down"] = (
            budget if isinstance(budget, int) and budget >= 0 else 0
        )

    def _spec_contains_stop_token(self, state: dict[str, Any]) -> bool:
        stop_ids = state.get("stop_token_ids") or []
        if not stop_ids:
            return False
        stop_set = set(stop_ids)
        return any(tok in stop_set for tok in state.get("spec_token_ids") or [])

    @staticmethod
    def _sequence_at(tokens: list[int], index: int, sequence: list[int]) -> bool:
        n = len(sequence)
        return bool(n) and tokens[index : index + n] == sequence

    def _spec_reasoning_exit_index(self, spec: list[int]) -> int | None:
        """First spec index that leaves think (implicit end or ``</think>``)."""
        for i in range(len(spec)):
            if self._sequence_at(spec, i, self.implicit_end_token_ids):
                return i
            if self._sequence_at(spec, i, self.think_end_token_ids):
                return i
        return None

    def _maybe_force_end_from_spec_eos(self, state: dict[str, Any]) -> None:
        if (
            state.get("reasoning_eos_policy") != "force_end"
            or not state.get("in_think")
            or state.get("in_end")
        ):
            return
        stop_ids = set(state.get("stop_token_ids") or [])
        if not stop_ids:
            return
        spec = state.get("spec_token_ids") or []
        exit_at = self._spec_reasoning_exit_index(spec)
        limit = len(spec) if exit_at is None else exit_at
        for i in range(limit):
            if spec[i] in stop_ids:
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["force_index"] = [i]
                return

    def _update_think_state(self, state: dict[str, Any]) -> None:
        if state.get("thinking_token_budget", -1) == -1 and (
            state.get("reasoning_eos_policy") != "force_end"
        ):
            return
        if len(self.think_end_token_ids) == 0:
            state["thinking_token_budget"] = -1
            state["in_end"] = False
            state["force_index"] = []
            return

        if state["start_thinking"] == -1:
            scan_offset = state.get("scan_offset", 0)
            output_slice = state.get("output_tok_ids", [])[scan_offset:]
            start_thinking = self._find_last_sequence_index(
                output_slice, self.think_start_token_ids
            )
            if start_thinking >= 0:
                start_thinking += scan_offset
            state["start_thinking"] = start_thinking
        if state["end_thinking"] == -1:
            state["end_thinking"] = self._find_last_end_index(
                state.get("output_tok_ids", []),
                state.get("scan_offset", 0),
            )

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
            self._reset_countdown(state)
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
        # force_end also falls through when a speculative token is EOS so we
        # can substitute reasoning_end before the budget expires.
        natural_end_with_continue = (
            state.get("continue_thinking", False) and state["end_thinking"] != -1
        )
        spec_eos_pending = state.get(
            "reasoning_eos_policy"
        ) == "force_end" and self._spec_contains_stop_token(state)
        if (
            not state.get("in_end", False)
            and predicted_countdown >= 0
            and state["start_thinking"] > -1
            and not natural_end_with_continue
            and not spec_eos_pending
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
                spec_len = len(state["spec_token_ids"])
                if self._has_token_budget(state):
                    remaining_budget = (
                        state["thinking_token_budget"] - state["think_count"]
                    )
                    if spec_len > 0:
                        if 0 < remaining_budget < spec_len:
                            state["force_index"] = [remaining_budget]
                        elif remaining_budget <= 0:
                            state["force_index"] = [0]
                        else:
                            state["force_index"] = [spec_len]
                    else:
                        state["force_index"] = [0]
                else:
                    existing = state.get("force_index")
                    state["force_index"] = [0] if spec_len == 0 else existing or [0]
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
            if state["in_think"] and self._has_token_budget(state):
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = remaining_budget
            elif not state["in_think"]:
                self._reset_countdown(state)

            # Same placement as budget exhaustion: set in_end here so the
            # rejected-end-token reset above does not undo it this step.
            self._maybe_force_end_from_spec_eos(state)

            total_thinking_tokens = (
                state["think_count"] + len(state["spec_token_ids"]) + 1
            )
            # Check if need to transition to end mode
            # If we have more thinking tokens than the budget,
            # we need to transition to end mode
            if (
                state["in_think"]
                and self._has_token_budget(state)
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
                        "start_thinking": -1,
                        "end_thinking": -1,
                        "think_count": 0,
                        "continue_thinking": False,
                        "scan_offset": len(state.get("output_tok_ids", [])),
                    }
                )
                self._reset_countdown(state)

    def _prepare_cu_num_tokens(
        self,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> None:
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

    def _request_row_count(
        self,
        seq_idx: int,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> int:
        if not self.in_spec_mode or predict_bonus_token:
            return 1
        if seq_idx >= len(spec_token_ids_for_layout):
            return 0
        spec = spec_token_ids_for_layout[seq_idx]
        n_rows = len(spec)
        exit_at = self._spec_reasoning_exit_index(spec)
        if exit_at is not None:
            n_rows = min(n_rows, exit_at)
        return n_rows

    def _apply_eos_policy_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> None:
        """Mask stop tokens in-think and force reasoning_end when EOS is argmax."""
        if not self.think_end_token_ids:
            return

        entries: list[tuple[int, int, list[int]]] = []
        for seq_idx in sorted(self._state.keys()):
            if seq_idx not in self.cu_num_tokens:
                continue
            state = self._state[seq_idx]
            if state.get("reasoning_eos_policy") != "force_end":
                continue
            if not state.get("in_think") and not state.get("in_end"):
                continue
            stop_ids = state.get("stop_token_ids") or []
            if not stop_ids:
                continue
            n_rows = self._request_row_count(
                seq_idx, predict_bonus_token, spec_token_ids_for_layout
            )
            base = self.cu_num_tokens[seq_idx]
            for offset in range(n_rows):
                row = base + offset
                if row >= logits.shape[0]:
                    break
                entries.append((seq_idx, row, stop_ids))

        if not entries:
            return

        device = logits.device
        vocab = logits.shape[1]
        rows = async_tensor_h2d(
            [row for _, row, _ in entries], dtype=torch.long, device=device
        )
        row_logits = logits.index_select(0, rows)
        row_max = row_logits.max(dim=-1).values
        max_stops = max(len(stops) for _, _, stops in entries)
        stop_cpu = torch.zeros((len(entries), max_stops), dtype=torch.long)
        valid_cpu = torch.zeros((len(entries), max_stops), dtype=torch.bool)
        for i, (_, _, stops) in enumerate(entries):
            clipped = [s for s in stops if 0 <= s < vocab]
            if clipped:
                stop_cpu[i, : len(clipped)] = torch.as_tensor(clipped, dtype=torch.long)
                valid_cpu[i, : len(clipped)] = True
        stop_ids_t = stop_cpu.to(device, non_blocking=True)
        valid = valid_cpu.to(device, non_blocking=True)
        gathered = row_logits.gather(1, stop_ids_t)
        gathered = gathered.masked_fill(~valid, float("-inf"))
        stop_max = gathered.max(dim=-1).values
        is_eos_argmax = (stop_max >= row_max).tolist()

        first_eos_row: dict[int, int] = {}
        mask_rows: list[int] = []
        mask_cols: list[int] = []
        for i, (seq_idx, row, stops) in enumerate(entries):
            for stop_id in stops:
                if 0 <= stop_id < logits.shape[1]:
                    mask_rows.append(row)
                    mask_cols.append(stop_id)
            if is_eos_argmax[i] and seq_idx not in first_eos_row:
                first_eos_row[seq_idx] = row - self.cu_num_tokens[seq_idx]

        if mask_rows:
            mask_row_t = async_tensor_h2d(mask_rows, dtype=torch.long, device=device)
            mask_col_t = async_tensor_h2d(mask_cols, dtype=torch.long, device=device)
            logits.index_put_(
                (mask_row_t, mask_col_t),
                logits.new_full((len(mask_rows),), float("-inf")),
            )

        for seq_idx, force_offset in first_eos_row.items():
            state = self._state[seq_idx]
            if state.get("in_end"):
                continue
            state["in_think"] = False
            state["in_end"] = True
            state["end_count"] = 0
            state["force_index"] = [force_offset]

    def _apply_forcing_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids_for_layout: list[list[int]],
    ) -> torch.Tensor:
        self._prepare_cu_num_tokens(predict_bonus_token, spec_token_ids_for_layout)

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
