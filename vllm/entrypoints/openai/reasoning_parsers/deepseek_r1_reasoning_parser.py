import re
from typing import Optional, Sequence, Tuple, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ReasoningParserManager.register_module("deepseek_r1")
class DeepSeekR1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning 
    text. This parser extracts the reasoning content from the model output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        self.reasoning_regex = re.compile(
            rf"{self.think_start_token}(.*?){self.think_end_token}", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if (self.think_start_token_id is None
                or self.think_end_token_id is None):
            raise RuntimeError(
                "DeepSeek R1 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")

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
                self.think_start_token_id, self.think_end_token_id
        ]):
            return None

        if self.think_start_token_id in previous_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content,
                                    content=content if content else None)
            elif self.think_end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.think_start_token_id in delta_token_ids:
            logger.info(delta_text)
            if self.think_end_token_id in delta_token_ids:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.think_start_token
                                                   ):end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content,
                                    content=content if content else None)
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # No <think> in previous or delta, reasoning content continues.
            return DeltaMessage(content=delta_text)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:

        # Check if the model output contains the <think> tokens.
        if (self.think_start_token not in model_output
                or self.think_end_token not in model_output):
            return None, model_output
        else:
            # Use a regex to find the reasoning content
            reasoning_content = self.reasoning_regex.findall(model_output)[0]

            # Remove the reasoning content from the model output
            # Although deepseek's <think> token is always at the
            # beginning of the line, we cannot guarantee that the
            # other models will follow this convention.
            # Therefore, we need to add :start_index.
            start_index = model_output.find(self.think_start_token)
            if start_index != -1:
                end_index = start_index + len(
                    f"{self.think_start_token}{reasoning_content}{self.think_end_token}"
                )
                model_output = model_output[:start_index] + \
                                model_output[end_index:]

                if len(model_output) == 0:
                    return reasoning_content, None

            return reasoning_content, model_output
