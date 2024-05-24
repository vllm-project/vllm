from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .sequence import Logprob, Sequence


class SequenceController:
    """Callback for generation control for a single sequence group.
    
    This can be part of SamplingParams and gets callbacks for various
    steps. It is to be used together with LogitsProcessor.
    """

    def scheduled(self, seq: 'Sequence'):
        """
        Called whenever the current sequence is scheduled to be run
        in the next step.
        """
        pass

    @staticmethod
    def forward_started():
        """
        Called when all sequences for the current step have been queued.
        """
        pass

    def sampled(self, seq: 'Sequence', token_id: int,
                logprobs: Dict[int, 'Logprob']) -> Tuple[int, List[int], bool]:
        """
        Informs the controller a given token has been sampled.
        Returns the number of tokens to backtrack, the tokens to append,
        and whether to stop.
        """
        if token_id == seq.eos_token_id:
            return 0, [], True
        return 0, [token_id], False

    def free(self, seq: 'Sequence'):
        """
        Called when the sequence is stopped, and deallocated.
        .scheduled() will not be called again for this sequence.
        """
        pass
