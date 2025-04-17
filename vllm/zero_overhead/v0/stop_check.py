

from typing import Optional
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceStatus
from vllm.zero_overhead.v0.sequence import ZeroOverheadSequence


class ZeroOverheadStopChecker(StopChecker):
    def __init__(self, max_model_len, get_tokenizer_for_seq):
        super().__init__(max_model_len, get_tokenizer_for_seq)
    
    
    def maybe_stop_sequence(
        self,
        seq: ZeroOverheadSequence,
        new_char_count: int,
        sampling_params: SamplingParams,
        lora_req: Optional[LoRARequest] = None,
    ) -> None:
        """Stop the finished sequences.

       new_char_count is the number of chars added to the
           sequence's output text for the newly generated token
        """

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.zero_overhead_get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.zero_overhead_get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = seq.zero_overhead_get_last_token_id()
        if last_token_id in (sampling_params.stop_token_ids or ()):
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = last_token_id
            return

        # Check if any stop strings are matched.
        stop = self.check_stop_strings(
            seq.output_text, new_char_count, sampling_params.stop,
            sampling_params.include_stop_str_in_output)
        if stop is not None:
            stop_str, truncate_to = stop
            if truncate_to != -1:
                seq.output_text = seq.output_text[:truncate_to]
            seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_str
            return

        # Check if the sequence has reached max_model_len.
        if seq.zero_overhead_get_len() > self._get_max_model_len(lora_req):
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.zero_overhead_get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return