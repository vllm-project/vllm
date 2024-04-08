from abc import ABC, abstractmethod
from vllm.config import SchedulerConfig
from vllm.sequence import SequenceGroup, SequenceGroupOutput
from typing import List

class SequenceGroupOutputProcessor(ABC):
    
    @staticmethod
    def create_output_processor(
        scheduler_config: SchedulerConfig,
        detokenizer,
        scheduler,
        seq_counter,
        get_tokenizer_for_seq,
        stop_checker,
    ):
        if scheduler_config.num_lookahead_slots == 0:
            from vllm.engine.output_processor.beam_search import BeamSearchOutputProcessor
            return BeamSearchOutputProcessor(
                scheduler_config,
                detokenizer,
                scheduler,
                seq_counter,
                stop_checker,
            )
        else:
            from vllm.engine.output_processor.block_decode import BlockDecodeOutputProcessor
            return BlockDecodeOutputProcessor(
                detokenizer,
                scheduler,
                seq_counter,
                get_tokenizer_for_seq,
                stop_checker,
            )

    @abstractmethod
    def process_outputs(self, sequence_group: SequenceGroup, outputs: List[SequenceGroupOutput]) -> None:
        pass
