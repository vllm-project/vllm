# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import  Union

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

logger = init_logger(__name__)


@ReasoningParserManager.register_module("qwen3")
class Qwen3ReasoningParser(DeepSeekR1ReasoningParser):


    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
                self.start_token_id, self.end_token_id
        ]):
            return None

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.start_token):end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # No <think> in previous or delta, also need to check for </think>.
            # Because the model may have generated </think> without <think>
            # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
            if self.end_token_id in delta_token_ids:
                # </think> in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token_id in previous_token_ids:
                # </think> in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no </think> in previous or delta, content continues
                return DeltaMessage(content=delta_text)

