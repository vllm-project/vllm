from abc import ABC, abstractmethod
from typing import List, Callable, Iterable

from vllm.config import SchedulerConfig
from vllm.sequence import SequenceGroup, SequenceGroupOutput, Sequence
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.stop_checker import StopChecker


class SequenceGroupOutputProcessor(ABC):
    """Interface for logic that processes new token ids in sequence groups,
    managing detokenization, stop checking, and freeing/forking sequences with
    the scheduler.

    This is highly coupled with the LLMEngine and should be seen as an extension
    of it. The logic is separated out to simplify the LLMEngine class and to
    allow a beam search implementation (which handles forking, etc) and a block
    decode implementation (which handles decoding >1 token per step).
    """

    @staticmethod
    def create_output_processor(
        scheduler_config: SchedulerConfig,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Iterable[int],
        get_tokenizer_for_seq: Callable[[Sequence], PreTrainedTokenizer],
        stop_checker: "StopChecker",
    ):
        """Create an output processor.

        This returns an output processor compatible with beam search if the
        scheduler is not configured to scheduler lookahead slots. Otherwise, it
        returns an output processor that is incompatible with beam search but
        which supports decoding more than one token per scheduling invocation.
        """
        if scheduler_config.num_lookahead_slots == 0:
            # Importing here to avoid cycle.
            from vllm.engine.output_processor.beam_search import (
                BeamSearchOutputProcessor)
            return BeamSearchOutputProcessor(
                scheduler_config,
                detokenizer,
                scheduler,
                seq_counter,
                stop_checker,
            )
        else:
            # Importing here to avoid cycle.
            from vllm.engine.output_processor.block_decode import (
                BlockDecodeOutputProcessor)
            return BlockDecodeOutputProcessor(
                detokenizer,
                scheduler,
                seq_counter,
                get_tokenizer_for_seq,
                stop_checker,
            )

    @abstractmethod
    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> None:
        """Process new token ids for the sequence group. Handles logic such as
        detokenization, stop checking, and freeing/forking sequences in the
        scheduler.
        """
        pass
