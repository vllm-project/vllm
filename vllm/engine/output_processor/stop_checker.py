from typing import Callable, Optional

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceStatus
from vllm.transformers_utils.tokenizer import AnyTokenizer


class StopChecker:
    """LLMEngine helper class which separates out the logic involving stop
    checking. This checks things such as: whether the eos token was emitted,
    whether the max_tokens has been consumed, whether a stop string has been
    emitted, or if we have exceeded the max model len.
    """

    def __init__(self, max_model_len: int,
                 get_tokenizer_for_seq: Callable[[Sequence], AnyTokenizer]):
        # Do not use it directly, but use `self._get_max_model_len`.
        self._max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_tokenizer_for_seq

        # the position to start checking for repetition
        self.repeat_start_from = 0

        # the number of tokens repeated
        self.repeated_count = 0
        # the gap between the repeated tokens
        self.repeated_gap = 0
        # the repeated ngram that we already generated
        self.repeated_total = 0

    def _get_max_model_len(self, lora_req: Optional[LoRARequest]):
        if lora_req and lora_req.long_lora_max_len:
            return lora_req.long_lora_max_len
        else:
            return self._max_model_len

    def maybe_stop_sequence(
        self,
        seq: Sequence,
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
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
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
        if seq.get_len() > self._get_max_model_len(lora_req):
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the last ngram is repeated in the output text.
        last_token = seq.output_text[-new_char_count:]
        # start checking for repetition after the first 32 tokens
        if seq.get_output_len() > 32 and self.check_ngram_repetition(seq, sampling_params, last_token):
            seq.status = SequenceStatus.FINISHED_REPEATED
            return

    def check_ngram_repetition(self, seq: Sequence, sampling_params: SamplingParams, last_token: str) -> bool:
        """Check if the last ngram is repeated in the output text.
        """

        is_done = False
        output_ids = seq.get_output_token_ids()
        last_token_id = seq.get_last_token_id()
        output_len = seq.get_output_len()

        repeated_at = None
        repeated_gap = None

        for i, token in enumerate(output_ids[self.repeat_start_from:-1]):
            if token == last_token_id:
                repeated_at = self.repeat_start_from + i
                repeated_gap = output_len - repeated_at

        if repeated_at is not None:
            self.repeated_count += 1
            # token_str = self.tokenizer.convert_ids_to_tokens([last_token])[0]
            # print(
            #     f"\n==> token ({last_token}) at {output_len}\n"
            #     f"==> repeat_at: {repeated_at}\n"
            #     f"==> repeated_count: {self.repeated_count}\n"
            #     f"==> repeated_gap: {repeated_gap}\n"
            #     f"==> repeate_start_from: {self.repeat_start_from}"
            # )

            self.repeat_start_from = repeated_at

        if repeated_at is None or repeated_gap != self.repeated_gap:
            self.repeated_count = 0
            self.repeated_gap = 0
            self.repeated_total = 0

        if repeated_gap is not None:
            self.repeated_gap = repeated_gap

        if self.repeated_count == self.repeated_gap and self.repeated_gap:
            self.repeated_total += 1
            self.repeated_count = 0

            # print(f"==> repeated_total: {self.repeated_total}")

            repeate_ngram_size = self.repeated_gap
            # print(f'==> repeate_ngram_size: {repeate_ngram_size}')

            if repeate_ngram_size == 1:
                # single token repetition
                is_done = self.repeated_total > 64
            elif repeate_ngram_size > 64:
                # paragraph repetition
                is_done = self.repeated_total >= 4
            else:
                # short ngram repetition?
                is_done = self.repeated_total >= 8

        return is_done



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
