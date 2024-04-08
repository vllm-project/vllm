from typing import List

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceStatus

logger = init_logger(__name__)
_LOCAL_LOGGING_INTERVAL_SEC = 5


class StopChecker:

    def __init__(self, scheduler, scheduler_config, get_tokenizer_for_seq):
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.get_tokenizer_for_seq = get_tokenizer_for_seq

    def maybe_stop_sequence(self, seq: Sequence,
                            sampling_params: SamplingParams,
                            new_token_ids: List[int]) -> None:
        """Stop the finished sequences."""
        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        if sampling_params.detokenize:
            for stop_str in sampling_params.stop:
                if seq.output_text.endswith(stop_str):
                    self._finalize_sequence(seq, sampling_params, stop_str)
                    seq.status = SequenceStatus.FINISHED_STOPPED
                    seq.stop_reason = stop_str
                    return

        # Determine if any stop_token_ids are in new_token_ids.
        intersection = set(new_token_ids).intersection(
            sampling_params.stop_token_ids)
        if intersection:
            # Get arbitrary token id that caused the stop.
            stop_token_id = next(iter(intersection))

            stop_str = self.get_tokenizer_for_seq(seq).convert_ids_to_tokens(
                stop_token_id)
            self._finalize_sequence(seq, sampling_params, stop_str)
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_token_id
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.eos_token_id in new_token_ids):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _finalize_sequence(self, seq: Sequence,
                           sampling_params: SamplingParams,
                           stop_string: str) -> None:
        if sampling_params.include_stop_str_in_output:
            return

        if stop_string and seq.output_text.endswith(stop_string):
            # Truncate the output text so that the stop string is
            # not included in the output.
            seq.output_text = seq.output_text[:-len(stop_string)]
