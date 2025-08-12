from abc import ABC, abstractmethod
from typing import Callable, Iterable, List

from transformers import PreTrainedTokenizer

from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import Sequence, SequenceGroup, SequenceGroupOutput
from vllm.transformers_utils.detokenizer import Detokenizer


class SequenceGroupOutputProcessor(ABC):
    """Interface for logic that processes new token ids in sequence groups,
    managing detokenization, stop checking, and freeing/forking sequences with
    the scheduler.

    This is highly coupled with the LLMEngine and should be seen as an extension
    of it. The logic is separated to simplify the LLMEngine class and allow
    separate implementations for single-step decoding (which supports beam
    search sequence forking) and multi-step decoding (which does not support
    beam search, but does support speculative decoding).
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

        This returns a single-step output processor if num_lookahead_slots is
        zero, else returns a multi-step output processor.
        """
        if scheduler_config.num_lookahead_slots == 0:
            # Importing here to avoid cycle.
            from vllm.engine.output_processor.single_step import (
                SingleStepOutputProcessor)
            return SingleStepOutputProcessor(
                scheduler_config,
                detokenizer,
                scheduler,
                seq_counter,
                stop_checker,
            )
        else:
            # Importing here to avoid cycle.
            from vllm.engine.output_processor.multi_step import (
                MultiStepOutputProcessor)
            return MultiStepOutputProcessor(
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
