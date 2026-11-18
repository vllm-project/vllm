# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Kimi protocol envelope around a JSON grammar.

Reasoning may reopen before the response body. Once JSON starts, only its
continuation, an optional response/message closing envelope, and EOS are legal.
The same transitions drive speculative validation, masks, and actual commits.
"""

from collections import deque
from dataclasses import dataclass

import torch

from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


@dataclass(frozen=True)
class _State:
    phase: str
    continuations: tuple[tuple[tuple[int, ...], str], ...] = ()


class KimiK3ProtocolGrammar(StructuredOutputGrammar):
    def __init__(
        self,
        body: XgrammarGrammar,
        tokenizer,
        prompt_token_ids: list[int],
        thinking: bool,
        max_rollback_tokens: int,
        stop_token_ids: set[int],
    ):
        self.body = body
        self.vocab_size = body.vocab_size
        self.stop_ids = frozenset(stop_token_ids or {tokenizer.eos_token_id})
        self.special_ids = (
            frozenset(tokenizer.all_special_ids)
            | frozenset(getattr(tokenizer, "special_tokens", {}).values())
            | self.stop_ids
        )

        def encode(text: str) -> tuple[int, ...]:
            ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            if not ids:
                raise ValueError(f"Empty Kimi protocol marker: {text!r}")
            return ids

        self.open_ids = encode("<|open|>")
        self.close_ids = encode("<|close|>")
        # These are reserved atomic tokens; channel names may span many tokens.
        if len(self.open_ids) != 1 or len(self.close_ids) != 1:
            raise ValueError("Kimi open/close delimiters must be atomic tokens")
        self.open_id, self.close_id = self.open_ids[0], self.close_ids[0]
        self.special_ids |= frozenset(
            self.open_ids + self.close_ids + encode("<|sep|>")
        )
        self._think_mask = torch.full(
            ((self.vocab_size + 31) // 32,), -1, dtype=torch.int32
        )
        for token in self.special_ids:
            if 0 <= token < self.vocab_size:
                self._set(self._think_mask, token, False)
        self._body_mask = self._think_mask.clone()
        for token in self.stop_ids:
            self._set(self._body_mask, token, True)
        self._set(self._think_mask, self.close_id, True)
        self.think = encode("think<|sep|>")
        self.response = encode("response<|sep|>")
        self.message = encode("message<|sep|>")
        self.thinking = thinking
        self._initial_state = _State("think" if thinking else "response")
        self.state = self._initial_state
        self.max_rollback_tokens = max(1, max_rollback_tokens)
        self.history: deque[tuple[_State, bool]] = deque(
            maxlen=self.max_rollback_tokens
        )
        # Initialize once, from the most recent explicit generation channel.
        # Historical turns are not scanned again for each speculative row.
        start = None
        for i in range(len(prompt_token_ids) - 1, -1, -1):
            if prompt_token_ids[i] != self.open_id:
                continue
            tail = tuple(prompt_token_ids[i + 1 :])
            if any(
                tail[: min(len(tail), len(channel))]
                == channel[: min(len(tail), len(channel))]
                for channel in (self.think, self.response)
            ):
                start = i
                break
        if start is not None:
            self.state = _State("between")
            for token in prompt_token_ids[start:]:
                if not self._accept(token):
                    raise ValueError("Unsupported Kimi structured-output prompt prefix")
        self._prompt_prefix = (
            list(prompt_token_ids[start:]) if start is not None else []
        )
        self._initial_state = (
            _State("between") if start is not None else self._initial_state
        )
        self.history.clear()

    def _header(self, alternatives) -> _State:
        return _State("header", tuple(alternatives))

    def _complete_body(self) -> bool:
        return self.body.matcher.is_completed()

    def _accept(self, token: int) -> bool:
        if not 0 <= token < self.vocab_size:
            return False
        old = self.state
        phase = old.phase
        body_advanced = False
        if phase == "header":
            matches = [
                (ids[1:], dest) for ids, dest in old.continuations if ids[0] == token
            ]
            if not matches:
                return False
            complete = [dest for ids, dest in matches if not ids]
            self.state = _State(complete[0]) if complete else self._header(matches)
        elif phase == "think":
            if token == self.close_id:
                self.state = self._header(((self.think, "between"),))
            elif token in self.special_ids:
                return False
        elif phase == "between" and token == self.open_id:
            channels = [(self.response, "response")]
            if self.thinking:
                channels.append((self.think, "think"))
            self.state = self._header(channels)
        elif phase in ("between", "response"):
            if phase == "response" and token == self.close_id and self._complete_body():
                self.state = self._header(((self.response, "after_response"),))
            else:
                if token in self.special_ids and token not in self.stop_ids:
                    return False
                if self.body.validate_tokens([token]) != [token]:
                    return False
                if not self.body.accept_tokens("kimi-protocol", [token]):
                    return False
                body_advanced = True
                self.state = _State(
                    "ended" if self.body.is_terminated() else "response"
                )
        elif phase == "after_response":
            if token in self.stop_ids:
                self.state = _State("ended")
            elif token == self.close_id:
                self.state = self._header(((self.message, "after_message"),))
            else:
                return False
        elif phase == "after_message" and token in self.stop_ids:
            self.state = _State("ended")
        else:
            return False
        self.history.append((old, body_advanced))
        return True

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        return all(self._accept(token) for token in tokens)

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        if len(tokens) > self.max_rollback_tokens:
            raise ValueError("Kimi validation exceeds the speculative window")
        accepted = []
        try:
            for token in tokens:
                if not self._accept(token):
                    break
                accepted.append(token)
        finally:
            self.rollback(len(accepted))
        return accepted

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > len(self.history):
            raise ValueError("Kimi protocol rollback exceeds retained history")
        for _ in range(num_tokens):
            self.state, body_advanced = self.history.pop()
            if body_advanced:
                self.body.rollback(1)

    @staticmethod
    def _set(row: torch.Tensor, token: int, allowed: bool) -> None:
        word, bit = divmod(token, 32)
        value = int(row[word]) & 0xFFFFFFFF
        value = value | (1 << bit) if allowed else value & ~(1 << bit)
        row[word] = value if value < (1 << 31) else value - (1 << 32)

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        row = bitmask[idx]
        phase = self.state.phase
        if phase == "think":
            row.copy_(self._think_mask)
        elif phase in ("between", "response"):
            self.body.fill_bitmask(bitmask, idx)
            row.bitwise_and_(self._body_mask)
            if phase == "between":
                self._set(row, self.open_id, True)
            elif self._complete_body():
                self._set(row, self.close_id, True)
        else:
            row.zero_()
            if phase == "header":
                allowed = {ids[0] for ids, _ in self.state.continuations}
            elif phase in ("after_response", "after_message", "ended"):
                allowed = set(self.stop_ids)
                if phase == "after_response":
                    allowed.add(self.close_id)
            else:
                raise AssertionError(phase)
            for token in allowed:
                self._set(row, token, True)

    def is_terminated(self) -> bool:
        return self.state.phase == "ended"

    def reset(self) -> None:
        self.body.reset()
        self.state = self._initial_state
        for token in self._prompt_prefix:
            assert self._accept(token)
        self.history.clear()
