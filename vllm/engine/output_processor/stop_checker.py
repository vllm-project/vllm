from typing import Callable, Optional

from transformers import PreTrainedTokenizer

from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceStatus


class StopChecker:
    """LLMEngine helper class which separates out the logic involving stop
    checking. This checks things such as: whether the eos token was emitted,
    whether the max_tokens has been consumed, whether a stop string has been
    emitted, or if we have exceeded the max model len.
    """

    def __init__(self, max_model_len: int,
                 get_tokenizer_for_seq: Callable[[Sequence],
                                                 PreTrainedTokenizer]):
        self.max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_tokenizer_for_seq

    def maybe_stop_sequence(self, seq: Sequence, new_char_count: int,
                            sampling_params: SamplingParams) -> None:
        """Stop the finished sequences.

       new_char_count is the number of chars added to the
           sequence's output text for the newly generated token
        """

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = seq.get_last_token_id()
        if last_token_id in sampling_params.stop_token_ids:
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = last_token_id
            return

        # Check if any stop strings are matched.
        stop_str = self._check_stop_strings(seq, new_char_count,
                                            sampling_params)
        if stop_str is not None:
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_str
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

    @staticmethod
    def _check_stop_strings(seq: Sequence, new_char_count: int,
                            sampling_params: SamplingParams) -> Optional[str]:
        """Check if any stop strings are matched and truncate sequence
        output text accordingly.

        Returns the stop string if matched or else None.
        """
        if not new_char_count:
            return None

        for stop_str in sampling_params.stop:
            stop_string_len = len(stop_str)
            # Avoid searching already-searched text.
            stop_index = seq.output_text.find(
                stop_str, -new_char_count - stop_string_len)
            if stop_index == -1:
                continue

            if sampling_params.include_stop_str_in_output:
                # Truncate to end of stop string.
                stop_index += stop_string_len
                if stop_index >= len(seq.output_text):
                    # No truncation required.
                    return stop_str

            # Truncate the output text to either the beginning
            # or end of the stop string.
            seq.output_text = seq.output_text[:stop_index]
            return stop_str
        return None
