# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import AnyTokenizer

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: Optional[LoRARequest] = None
    cum_logprob: float = 0.0
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
    sequences: list[BeamSearchSequence]


class BeamSearchInstance:

    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: Optional[LoRARequest] = None,
        logprobs: Optional[list[dict[int, Logprob]]] = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    stop_token_ids: list[int],
    stop_tokens: list[str],
    tokenizer: Optional[AnyTokenizer] = None,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] in stop_token_ids or (
            tokenizer is not None
            and tokenizer.decode([tokens[-1]]).strip() in stop_tokens):
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(stop_token_ids: Union[int, list[int]],
                                   length_penalty: float,
                                   stop_tokens: Optional[list[str]] = None,
                                   tokenizer: Optional[AnyTokenizer] = None):
    stop_token_ids = [stop_token_ids] if isinstance(stop_token_ids,
                                                    int) else stop_token_ids
    stop_tokens = stop_tokens or []

    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(x.tokens, x.cum_logprob, stop_token_ids,
                                     stop_tokens, tokenizer, length_penalty)

    return sort_beams_key
