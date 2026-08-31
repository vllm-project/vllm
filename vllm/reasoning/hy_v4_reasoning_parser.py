# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypedDict

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

# Must match the chat template's own default for ``reasoning_effort``.
DEFAULT_REASONING_EFFORT = "high"


class ReasoningDelta(TypedDict):
    """One streaming reasoning delta. Either/both fields may be None."""

    reasoning: str | None
    content: str | None


# NOTE: mirrored in ``vllm.tool_parsers.hy_v4_tool_parser`` (same pattern as
# ``gemma4_utils``) so neither package depends on the other.
def detect_token_suffix(tokenizer: TokenizerLike) -> str:
    """Detect the per-checkpoint suffix used by Hunyuan structural tokens.

    Args:
        tokenizer: Tokenizer of the served checkpoint.

    Returns:
        The suffix including its leading colon (e.g. ``":6124c78e"``), or ``""``
        when the checkpoint uses unsuffixed tokens.

    Raises:
        RuntimeError: The tokenizer declares the structural tokens through
            ``model_specific_special_tokens``, which transformers 5 no longer
            round-trips.
    """

    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        think_begin_as_special = init_kwargs.get(
            "model_specific_special_tokens", {}
        ).get("think_begin_token", "")
        if think_begin_as_special:
            raise RuntimeError(
                "This checkpoint declares HYV4 structural tokens (think_begin_token"
                "/toolcalls_begin_token/argkey_begin_token) in "
                "tokenizer_config.json, which transformers 5 no longer supports. "
                "Remove those fields and keep the tokens in the tokenizer's own "
                "token definitions so the suffix can be read from the vocab."
            )

    structural_token_re = re.compile(
        r"<(?:think|tool_calls|tool_call|arg_key|arg_value)(:[^\s>]+)?>"
    )
    for token in tokenizer.get_vocab():
        match = structural_token_re.fullmatch(token)
        if match:
            return match.group(1) or ""

    return ""


class HYV4ReasoningExtractor:
    """Reasoning extraction for HYV4, on plain data.

    ``thinking`` selects the mode:
    - ``True``: parse ``<think>...</think>``; the start token is injected at the
      end of the prompt and is therefore usually absent from the output.
    - ``False``: ``no_think`` -- the whole output is content, no reasoning.
    """

    def __init__(self, vocab: dict[str, int], token_suffix: str, thinking: bool = True):
        self.thinking = thinking
        self.start_token: str = f"<think{token_suffix}>"
        self.end_token: str = f"</think{token_suffix}>"

        start_token_id = vocab.get(self.start_token)
        end_token_id = vocab.get(self.end_token)
        if thinking and (start_token_id is None or end_token_id is None):
            raise RuntimeError(
                "HYV4 reasoning extractor could not locate think "
                "start/end tokens in the tokenizer!"
            )
        # In no_think mode the ids are unused (kept for symmetry / debugging).
        self.start_token_id: int | None = start_token_id
        self.end_token_id: int | None = end_token_id

    def has_reasoning_ended(self, token_ids: Sequence[int]) -> bool:
        """Idempotent STATE query: has ``</think>`` been emitted yet?

        Monotonic -- once the end token is present this stays True. Use for
        "has reasoning finished, given everything so far". (no_think is always
        ended.) Contrast with ``reasoning_ended_in_delta`` (edge detector).
        """
        if not self.thinking:
            return True
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] == self.start_token_id:
                return False
            if token_ids[i] == self.end_token_id:
                return True
        return False

    def reasoning_ended_in_delta(
        self, token_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Edge detector (NOT idempotent): did reasoning end *in this* step?

        True only on the step whose ``delta_ids`` contains ``</think>``. Callers
        use it once behind their own "already ended" latch, so it need not be
        monotonic; it only checks the new delta (cheaper than scanning the whole
        sequence). Contrast with ``has_reasoning_ended`` (cumulative state).
        """
        if not self.thinking:
            return True
        end_token_id = self.end_token_id
        if end_token_id is None:
            return False
        return end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return the token ids after ``</think>`` (all of them in no_think)."""
        if not self.thinking:
            return input_ids
        head = input_ids[: max(0, len(input_ids) - 1)]
        if self.end_token_id not in head:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning tokens (0 in no_think).

        The ``<think>`` start token is injected at the END of the prompt, so it
        is normally absent from ``token_ids``. Only skip a leading start token
        when it actually appears first (legacy / no prompt-injection); otherwise
        count every token up to ``</think>``.
        """
        if not self.thinking:
            return 0
        start = 1 if (token_ids and token_ids[0] == self.start_token_id) else 0
        try:
            return token_ids.index(self.end_token_id, start) - start
        except ValueError:
            # No end token -- reasoning was truncated; everything counts.
            return len(token_ids) - start

    def extract_reasoning(self, model_output: str) -> tuple[str | None, str | None]:
        """Split a complete output into ``(reasoning, content)``.

        In no_think mode reasoning is None and the whole output is content. The
        start token may be absent (injected in the prompt), so reasoning is
        assumed to begin at the start of the output.
        """
        if not self.thinking:
            return None, model_output

        parts = model_output.partition(self.start_token)
        model_output = parts[2] if parts[1] else parts[0]

        if self.end_token not in model_output:
            return model_output, None
        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> ReasoningDelta | None:
        """Classify a streaming delta as reasoning and/or content.

        In no_think mode every delta is content. Otherwise: same logic as vLLM's
        generic thinking parser, except the final branch (no start token seen
        anywhere) treats output as reasoning-then-content, because HYV4 injects
        ``<think>`` in the prompt (absent from the output).
        """
        if not self.thinking:
            if not delta_text:
                return None
            return ReasoningDelta(reasoning=None, content=delta_text)

        # Skip a delta that is a single lone start/end special token.
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id,
            self.end_token_id,
        ):
            return None

        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return ReasoningDelta(reasoning=reasoning, content=content or None)
            elif self.end_token_id in previous_token_ids:
                return ReasoningDelta(reasoning=None, content=delta_text)
            else:
                return ReasoningDelta(reasoning=delta_text, content=None)
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return ReasoningDelta(reasoning=reasoning, content=content or None)
            else:
                return ReasoningDelta(reasoning=delta_text, content=None)
        else:
            # HYV4: <think> was injected at the end of the prompt, so it never
            # appears in the output. Treat the stream as reasoning until
            # </think>, then content.
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return ReasoningDelta(reasoning=reasoning, content=content or None)
            elif self.end_token_id in previous_token_ids:
                return ReasoningDelta(reasoning=None, content=delta_text)
            else:
                return ReasoningDelta(reasoning=delta_text, content=None)


def build_reasoning_extractor(
    tokenizer: TokenizerLike, *, thinking: bool = True
) -> HYV4ReasoningExtractor:
    return HYV4ReasoningExtractor(
        tokenizer.get_vocab(), detect_token_suffix(tokenizer), thinking
    )


class HYV4ReasoningParser(ReasoningParser):
    """vLLM adapter around :class:`HYV4ReasoningExtractor`.

    Decides ``thinking`` from the request's ``reasoning_effort`` and converts
    the extractor's ``ReasoningDelta`` to ``DeltaMessage``.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Decide thinking vs no_think from the request; the extractor handles
        # both modes internally. Prefer chat_template_kwargs.reasoning_effort,
        # then the top-level reasoning_effort. The default must match the chat
        # template, which sets reasoning_effort='high' when the request omits
        # it -- defaulting to no_think here would return the whole
        # chain-of-thought (and a stray ``</think:SUF>``) as content.
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        reasoning_effort = chat_kwargs.get("reasoning_effort", None)
        if not reasoning_effort:
            reasoning_effort = (
                kwargs.get("reasoning_effort") or DEFAULT_REASONING_EFFORT
            )
        logger.debug("reasoning_effort for choosing parser: %s", reasoning_effort)
        thinking = reasoning_effort != "no_think"

        self._extractor = build_reasoning_extractor(tokenizer, thinking=thinking)

    # Reported regardless of the per-request thinking mode: ``ReasoningConfig``
    # is engine-wide and needs the checkpoint's delimiters to derive the token
    # ids used for thinking budget enforcement.
    @property
    def reasoning_start_str(self) -> str:
        return self._extractor.start_token

    @property
    def reasoning_end_str(self) -> str:
        return self._extractor.end_token

    def is_reasoning_end(self, input_ids) -> bool:
        return self._extractor.has_reasoning_ended(input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self._extractor.reasoning_ended_in_delta(input_ids, delta_ids)

    def extract_content_ids(self, input_ids) -> list[int]:
        return self._extractor.extract_content_ids(input_ids)

    def count_reasoning_tokens(self, token_ids) -> int:
        return self._extractor.count_reasoning_tokens(token_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        return self._extractor.extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
    ) -> DeltaMessage | None:
        delta = self._extractor.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if delta is None:
            return None
        return DeltaMessage(reasoning=delta["reasoning"], content=delta["content"])
