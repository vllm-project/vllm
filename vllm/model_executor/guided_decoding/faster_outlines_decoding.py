import math
import torch
from re import escape as regex_escape
from json import dumps as json_dumps
from typing import Dict, List, Union

from faster_outlines.fsm import (
    LazyVLLMRegexGuide, 
    Write,
    Generate,
    TokenVocabulary
)
from transformers import PreTrainedTokenizerBase
from vllm.sampling_params import GuidedDecodingParams
from outlines.fsm.json_schema import build_regex_from_schema

TOKENIZER_CACHE: Dict[str, TokenVocabulary] = {}

class BaseLogitsProcessor:

    def __init__(self, guide):
        self._guide = guide
        self.state = 0
    
    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""
        if len(input_ids) > 0:
            self.state = self._guide.get_next_state(
                state=self.state,
                token_id=input_ids[-1]
            )

        instruction = self._guide.get_next_instruction(
            state=self.state)
        if type(instruction) == Generate:  # noqa: E721
            allowed_tokens = instruction.tokens
        elif type(instruction) == Write:  # noqa: E721
            # TODO: support fast forward tokens
            allowed_tokens = [instruction.tokens[0]]
        else:
            raise TypeError(
                f"Unsupported instruction type {type(instruction)}")

        mask = torch.full((scores.shape[-1], ),
                          -math.inf,
                          device=scores.device)
        mask[allowed_tokens] = 0
        scores.add_(mask)
        return scores


class RegexLogitsProcessor(BaseLogitsProcessor):

    @classmethod
    def _get_guide(cls, regex_string: str,
                   tokenizer: PreTrainedTokenizerBase):
        vocab = _adapt_tokenizer(tokenizer)
        return LazyVLLMRegexGuide(regex_string, vocab)

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            The model's tokenizer

        """
        super().__init__(
            RegexLogitsProcessor._get_guide(regex_string, tokenizer))

def _adapt_tokenizer(tokenizer: PreTrainedTokenizerBase):
    """
    Adapt VLLM's tokenizer into a TokenVocabulary, readable by Rust.
    """
    if TOKENIZER_CACHE.get(tokenizer.name_or_path) is not None:

        return TOKENIZER_CACHE[tokenizer.name_or_path]

    token_vocab = TokenVocabulary(
        tokenizer.get_vocab(),
        tokenizer.eos_token_id,
        set(tokenizer.all_special_tokens)
    )

    TOKENIZER_CACHE[tokenizer.name_or_path] = token_vocab

    return token_vocab

def get_local_faster_outlines_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, 
    tokenizer: PreTrainedTokenizerBase
) -> Union[RegexLogitsProcessor, None]:
    regex = _get_regex(guided_params)

    if not regex:
        return None
    
    return RegexLogitsProcessor(regex, tokenizer)

def _get_regex(
    guided_params: GuidedDecodingParams
) -> Union[str, None]:
    if guided_params.json:
        if isinstance(guided_params.json, dict):
            # turn dict into hashable string
            json = build_regex_from_schema(json_dumps(guided_params.json))
        else:
            json = build_regex_from_schema(guided_params.json)
        return json
    elif guided_params.regex:
        return guided_params.regex
    elif guided_params.choice:
        # choice just uses regex
        choices = [
            regex_escape(str(choice)) for choice in guided_params.choice
        ]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex
    return None
