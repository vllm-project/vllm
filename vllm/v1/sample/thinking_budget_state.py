# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch thinking token budget state; applied after penalties at sample time."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig


def maybe_create_thinking_budget_state_holder(
    reasoning_config: "ReasoningConfig | None",
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
) -> "ThinkingBudgetStateHolder | None":
    if reasoning_config is None:
        return None
    return ThinkingBudgetStateHolder(
        reasoning_config, max_num_seqs, num_spec_tokens, device, PIN_MEMORY
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
        else:
            rs = reasoning_config.reasoning_start_token_ids
            re = reasoning_config.reasoning_end_token_ids
            self.think_start_token_ids = rs if rs else []
            self.think_end_token_ids = re if re else []

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
            if thinking_token_budget is not None:
                self._state[index] = self._init_state_entry(
                    prompt_tok_ids, thinking_token_budget
                )
                self._state[index]["output_tok_ids"] = output_tok_ids
                self._state[index]["spec_token_ids"] = []
            else:
                self._state.pop(index, None)

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                state1 = self._state.get(i1)
                state2 = self._state.get(i2)
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
            # Strip draft suffix only when callers pass combined output+spec rows
            # (penalties / bad words). TB update_state uses committed outputs only.
            out = state["output_tok_ids"]
            spec = state["spec_token_ids"]
            spec_len = len(spec)
            if (
                spec_len > 0
                and len(out) >= spec_len
                and out[-spec_len:] == spec
            ):
                state["output_tok_ids"] = out[:-spec_len]
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
        return self._apply_forcing_to_logits(logits, predict_bonus_token, spec_lists)

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    @staticmethod
    def _find_last_sequence_index_from(
        target_list: list[int], token_ids: list[int], search_start: int
    ) -> int:
        """Last occurrence of ``token_ids`` at or after ``search_start``."""
        if not token_ids:
            return -1
        lo = max(0, search_start)
        for i in range(len(target_list) - len(token_ids), lo - 1, -1):
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
            last_end = self._find_last_sequence_index(
                prompt_tok_ids, self.think_end_token_ids
            )
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
            "start_search_pos": 0,
            "end_search_pos": 0,
            "in_spec_mode": False,
            "bonus_token_forced": False,
            "continue_thinking": continue_thinking,
            "scan_offset": 0,
        }

    def _update_think_state(self, state: dict[str, Any]) -> None:
        if state.get("thinking_token_budget", -1) == -1:
            return
        if len(self.think_end_token_ids) == 0:
            state["thinking_token_budget"] = -1
            state["in_end"] = False
            state["force_index"] = []
            return

        output_tok_ids = state.get("output_tok_ids", [])
        if state["start_thinking"] == -1:
            seq_len = len(self.think_start_token_ids)
            scan_offset = state.get("scan_offset", 0)
            start_thinking = self._find_last_sequence_index_from(
                output_tok_ids,
                self.think_start_token_ids,
                max(scan_offset, state["start_search_pos"] - (seq_len - 1)),
            )
            if start_thinking >= 0 and scan_offset > 0:
                # Re-entry after a forced end: budget was already exhausted
                # in a prior block, so immediately force-close this one.
                # scan_offset > 0 is only set after forced-end completion
                # (never after natural end), so this won't block legitimate
                # re-entries where budget remains.
                state["start_thinking"] = start_thinking
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["force_index"] = [0]
                return
            state["start_thinking"] = start_thinking
            if start_thinking == -1:
                state["start_search_pos"] = len(output_tok_ids)
        if state["end_thinking"] == -1:
            seq_len = len(self.think_end_token_ids)
            scan_offset = state.get("scan_offset", 0)
            end_thinking = self._find_last_sequence_index_from(
                output_tok_ids,
                self.think_end_token_ids,
                max(scan_offset, state["end_search_pos"] - (seq_len - 1)),
            )
            state["end_thinking"] = end_thinking
            if end_thinking == -1:
                state["end_search_pos"] = len(output_tok_ids)

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
        if (
            not state.get("in_end", False)
            and predicted_countdown >= 0
            and state["start_thinking"] > -1
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

            elif absolute_start_pos >= 0 and not state["continue_thinking"]:
                # Found think start - entering think mode
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count

            elif absolute_end_pos >= 0:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0

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
