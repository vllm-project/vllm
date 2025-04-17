

from vllm.sampling_params import SamplingParams
from vllm.sequence import VLLM_INVALID_TOKEN_ID
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.detokenizer_utils import convert_prompt_ids_to_tokens, detokenize_incrementally
from vllm.zero_overhead.v0.sequence import ZeroOverheadSequence


class ZeroOverheadDetokenizer(Detokenizer):
    def __init__(self, tokenizer_group):
        super().__init__(tokenizer_group)
    
    def decode_sequence_inplace(self, seq: ZeroOverheadSequence,
                                prms: SamplingParams) -> int:
        """Decodes the new token for a sequence. In-place operation.

        Args:
            seq: The sequence to decode.
            prms: The sampling parameters used to generate the sequence.

        Returns:
            The number of characters added to the output text.
        """   
        eff_length = seq.get_prompt_len() + seq.effective_output_len
        all_input_ids = seq.get_token_ids()[ : eff_length]

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