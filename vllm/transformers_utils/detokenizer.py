from typing import Dict, List, Optional

from vllm.sequence import (VLLM_INVALID_TOKEN_ID, Logprob, SamplingParams,
                           Sequence, SequenceGroup)

from .detokenizer_utils import (convert_prompt_ids_to_tokens,
                                detokenize_incrementally)
from .tokenizer import AnyTokenizer
from .tokenizer_group import BaseTokenizerGroup


class Detokenizer:
    """Provides methods to decode the output of a model into text."""

    def __init__(self, tokenizer_group: BaseTokenizerGroup):
        self.tokenizer_group = tokenizer_group

    def get_tokenizer_for_seq(self, sequence: Sequence) -> AnyTokenizer:
        """Returns the HF tokenizer to use for a given sequence."""
        return self.tokenizer_group.get_lora_tokenizer(sequence.lora_request)

    def decode_prompt_logprobs_inplace(self, seq_group: SequenceGroup,
                                       prompt_logprobs: List[Optional[Dict[
                                           int, Logprob]]],
                                       position_offset: int) -> None:
        """Decodes the logprobs for the prompt of a sequence group.

        Args:
            seq_group: The sequence group to decode.
            prompt_logprobs: The logprobs to decode.
            position_offset: Offset of the first index of the logprobs 
                relative to the start of the sequence (for chunked prefill).
        
        Returns:
            The prompt logprobs with the decoded tokens.
        """
        prms = seq_group.sampling_params
        assert prms is not None

        # We can pick any sequence for the prompt.
        seq = seq_group.get_seqs()[0]
        # Only prompt, without the generated token.
        all_token_ids = seq.get_token_ids()
        prompt_token_ids = all_token_ids[:-1]
        tokenizer = self.get_tokenizer_for_seq(seq)
        prefix_offset = 0
        read_offset = 0
        next_iter_prefix_offset = 0
        next_iter_read_offset = 0
        next_iter_tokens: List[str] = []
        prev_tokens = None

        for token_position_in_logprob, prompt_logprobs_for_token in enumerate(
                prompt_logprobs):

            # Absolute token position equals the index in the logprobs
            # list plus the offset of the entire logprobs list relative
            # to the start of the sequence.
            token_position = token_position_in_logprob + position_offset
            if not prompt_logprobs_for_token:
                continue
            for token_id, sample_logprob in prompt_logprobs_for_token.items():
                if (sample_logprob.decoded_token is None
                        and token_id != VLLM_INVALID_TOKEN_ID):
                    prompt_token_ids_with_token = (
                        prompt_token_ids[:token_position] + [token_id])
                    (new_tokens, new_text, new_prefix_offset,
                     new_read_offset) = detokenize_incrementally(
                         tokenizer=tokenizer,
                         all_input_ids=prompt_token_ids_with_token,
                         prev_tokens=prev_tokens,
                         prefix_offset=prefix_offset,
                         read_offset=read_offset,
                         skip_special_tokens=prms.skip_special_tokens,
                         spaces_between_special_tokens=prms.
                         spaces_between_special_tokens,
                     )

                    sample_logprob.decoded_token = new_text

                    # Use the offsets & prev tokens corresponding to
                    # real tokens to ensure detokenization is consistent
                    # actual with prompt.
                    if token_id == all_token_ids[token_position]:
                        next_iter_prefix_offset = new_prefix_offset
                        next_iter_read_offset = new_read_offset
                        next_iter_tokens = new_tokens

            # Advance to the next token position.
            prefix_offset = next_iter_prefix_offset
            read_offset = next_iter_read_offset
            if prev_tokens is None:
                prev_tokens = next_iter_tokens.copy()
            else:
                prev_tokens.extend(next_iter_tokens)

    def decode_sequence_inplace(self, seq: Sequence,
                                prms: SamplingParams) -> int:
        """Decodes the new token for a sequence. In-place operation.

        Args:
            seq: The sequence to decode.
            prms: The sampling parameters used to generate the sequence.

        Returns:
            The number of characters added to the output text.
        """
        all_input_ids = seq.get_token_ids()
        token_id_generated_this_iteration = all_input_ids[-1]
        tokenizer = self.get_tokenizer_for_seq(seq)

        # Convert prompt token IDs to tokens if necessary.
        # Do it here so that we don't have to repeat this
        # computation for each logprob.
        if seq.tokens is None:
            (seq.tokens, seq.prefix_offset,
             seq.read_offset) = convert_prompt_ids_to_tokens(
                 tokenizer=tokenizer,
                 prompt_ids=all_input_ids[:-1],
                 skip_special_tokens=prms.skip_special_tokens,
             )

        (new_tokens, new_decoded_token_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             tokenizer=tokenizer,
             all_input_ids=all_input_ids,
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )

        # Decode logprobs
        logprobs = seq.output_logprobs[-1]
        if logprobs:
            previous_tokens = all_input_ids[:-1]
            for token_id, sample_logprob in logprobs.items():
                # If the token was generated this iteration,
                # use the provided text.
                if token_id == token_id_generated_this_iteration:
                    sample_logprob.decoded_token = new_decoded_token_text
                    continue

                if (sample_logprob.decoded_token is None
                        and token_id != VLLM_INVALID_TOKEN_ID):
                    all_input_ids_with_logprob = previous_tokens + [token_id]
                    (_, new_text, _, _) = detokenize_incrementally(
                        tokenizer=tokenizer,
                        all_input_ids=all_input_ids_with_logprob,
                        prev_tokens=seq.tokens,
                        prefix_offset=seq.prefix_offset,
                        read_offset=seq.read_offset,
                        skip_special_tokens=prms.skip_special_tokens,
                        spaces_between_special_tokens=prms.
                        spaces_between_special_tokens,
                    )
                    sample_logprob.decoded_token = new_text

        seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_decoded_token_text

        return len(new_decoded_token_text)
