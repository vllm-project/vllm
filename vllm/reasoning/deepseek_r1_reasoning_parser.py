# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import List

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.

    DeepSeek R1 uses a byte-level BPE tokenizer. When reasoning content
    contains non-ASCII characters (e.g. CJK), the incremental detokenizer
    may emit raw byte-level glyphs (e.g. "Ä±" instead of "ı") because each
    token maps to a single UTF-8 byte. To avoid this we re-decode the
    *cumulative* reasoning token IDs (with special tokens skipped) on each
    streaming step and return only the newly decoded suffix as the delta.

    The number of characters already emitted is tracked in
    ``_prev_reasoning_decoded_len`` so only the fresh suffix is returned
    each turn.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        # Tracks how many characters of reasoning text have already been
        # emitted in previous streaming steps (for the re-decode approach).
        self._prev_reasoning_decoded_len: int = 0

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def _decode_ids(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs to a UTF-8 string, skipping special tokens.

        Using skip_special_tokens=True ensures that <think>/<｀/think> markers
        are not included in the returned text and that multi-byte UTF-8
        sequences spanning token boundaries are assembled correctly.
        """
        return self.model_tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
        )

    def _reasoning_token_ids(self, token_ids: Sequence[int]) -> List[int]:
        """Extract only the token IDs inside <think>...</think>.

        Handles models that may omit the start token (treats everything before
        </think> as reasoning in that case).
        """
        ids = list(token_ids)
        if self.start_token_id in ids:
            start_pos = ids.index(self.start_token_id)
            try:
                end_pos = ids.index(self.end_token_id, start_pos + 1)
                return ids[start_pos + 1 : end_pos]
            except ValueError:
                return ids[start_pos + 1 :]
        else:
            # No start token: treat everything before end token as reasoning.
            try:
                end_pos = ids.index(self.end_token_id)
                return ids[:end_pos]
            except ValueError:
                return ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract reasoning/content from a streaming delta.

        Overrides the base implementation to re-decode cumulative reasoning
        token IDs via the tokenizer so that multi-byte UTF-8 sequences
        (e.g. CJK characters) are assembled correctly instead of being
        emitted as raw byte-level BPE glyphs.
        """
        # Skip a lone special token (start/end marker with no payload yet).
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id,
            self.end_token_id,
        ):
            return None

        start_in_prev = self.start_token_id in previous_token_ids
        start_in_delta = self.start_token_id in delta_token_ids
        end_in_prev = self.end_token_id in previous_token_ids
        end_in_delta = self.end_token_id in delta_token_ids

        # Reasoning fully ended in a prior step → this delta is content.
        # Use delta_text directly; content is emitted by the standard path.
        if end_in_prev and not start_in_delta:
            return DeltaMessage(content=delta_text)

        # We are inside the reasoning span (or it just ended this step).
        in_reasoning = (
            start_in_prev
            or start_in_delta
            or (not end_in_prev and not end_in_delta)
        )
        if in_reasoning or end_in_delta:
            # Re-decode the full reasoning span to date (UTF-8-safe).
            reasoning_ids = self._reasoning_token_ids(current_token_ids)
            full_reasoning = self._decode_ids(reasoning_ids) if reasoning_ids else ""

            # Slice off only the portion not yet sent to the client.
            fresh = full_reasoning[self._prev_reasoning_decoded_len :]
            self._prev_reasoning_decoded_len = len(full_reasoning)

            if end_in_delta:
                # </think> arrived this step; any ids after it are content.
                ids = list(current_token_ids)
                end_pos = ids.index(self.end_token_id)
                content_ids = ids[end_pos + 1 :]
                content_text = self._decode_ids(content_ids) if content_ids else None
                return DeltaMessage(
                    reasoning=fresh or None,
                    content=content_text or None,
                )

            return DeltaMessage(reasoning=fresh or None)

        # Fallback: no thinking markers seen → pass through as content.
        return DeltaMessage(content=delta_text)
