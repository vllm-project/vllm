from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens includes the prompt.
    tokens: List[int]
    cum_logprob: float = 0.0
    text: Optional[str] = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
    sequences: List[BeamSearchSequence]


class BeamSearchInstance:

    def __init__(self, prompt_tokens: List[int]):
        self.beams: List[BeamSearchSequence] = [
            BeamSearchSequence(tokens=prompt_tokens)
        ]
        self.completed: List[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: List[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):

    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(x.tokens, x.cum_logprob, eos_token_id,
                                     length_penalty)

    return sort_beams_key
