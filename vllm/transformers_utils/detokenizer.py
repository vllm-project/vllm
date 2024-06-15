from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.sequence import Logprob, SamplingParams, Sequence, SequenceGroup
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
    BaseTokenizerGroup)

# Used eg. for marking rejected tokens in spec decoding.
INVALID_TOKEN_ID = -1


class Detokenizer:
    """Provides methods to decode the output of a model into text."""

    def __init__(self, tokenizer_group: BaseTokenizerGroup):
        self.tokenizer_group = tokenizer_group

    def get_tokenizer_for_seq(self,
                              sequence: Sequence) -> "PreTrainedTokenizer":
        """Returns the HF tokenizer to use for a given sequence."""
        return self.tokenizer_group.get_lora_tokenizer(sequence.lora_request)

    def decode_prompt_logprobs_inplace(
            self, seq_group: SequenceGroup,
            prompt_logprobs: List[Optional[Dict[int, Logprob]]]) -> None:
        """Decodes the logprobs for the prompt of a sequence group.

        Args:
            seq_group: The sequence group to decode.
            prompt_logprobs: The logprobs to decode.
        
        Returns:
            The prompt logprobs with the decoded tokens.
        """
        prms = seq_group.sampling_params
        # We can pick any sequence for the prompt.
        seq = next(iter(seq_group.seqs_dict.values()))
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

        for token_position, prompt_logprobs_for_token in enumerate(
                prompt_logprobs):
            if not prompt_logprobs_for_token:
                continue
            for token_id, sample_logprob in prompt_logprobs_for_token.items():
                if (sample_logprob.decoded_token is None
                        and token_id != INVALID_TOKEN_ID):
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
                prev_tokens = next_iter_tokens
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
                        and token_id != INVALID_TOKEN_ID):
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


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    output_tokens: List[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts: List[str] = []
    current_sub_text: List[str] = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    else:
        return "".join(sub_texts)


# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5


def convert_prompt_ids_to_tokens(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt_ids: List[int],
    skip_special_tokens: bool = False,
) -> Tuple[List[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2:],
        skip_special_tokens=skip_special_tokens)
    read_offset = len(new_tokens)
    prefix_offset = max(
        read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    return new_tokens, prefix_offset, read_offset


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> Tuple[List[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.
    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset,
         read_offset) = convert_prompt_ids_to_tokens(
             tokenizer,
             all_input_ids[:-1],
             skip_special_tokens=skip_special_tokens)
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if new_token_id >= len(tokenizer):
        new_tokens = [""]
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens)
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
    output_tokens = prev_tokens + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text):]
    return new_tokens, new_text, read_offset, len(output_tokens)
