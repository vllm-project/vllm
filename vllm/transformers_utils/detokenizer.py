from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple, TypeVar

from vllm.logprobs import Logprob
from vllm.sampling_params import SamplingParams
from vllm.utils import ensure_list

from .tokenizer import AnyTokenizer

# Used eg. for marking rejected tokens in spec decoding.
INVALID_TOKEN_ID = -1

T = TypeVar("T")


class IncrementalDetokenizer:

    def __init__(self):
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None

    def decode_sequence_inplace(self, all_input_ids: GenericSequence[int],
                                output_logprobs: Optional[List[Dict[int,
                                                                    Logprob]]],
                                params: SamplingParams,
                                tokenizer: AnyTokenizer) -> str:
        """Decodes the new token for a sequence. In-place operation.

        Args:
            all_input_ids: The sequence to decode.
            output_logprobs: The current list of output logprobs.
            params: The sampling parameters used to generate the sequence.
            tokenizer: The tokenizer to use.

        Returns:
            The number of characters added to the output text.
        """
        if not all_input_ids:
            return ""

        token_id_generated_this_iteration = all_input_ids[-1]

        # Convert prompt token IDs to tokens if necessary.
        # Do it here so that we don't have to repeat this
        # computation for each logprob.
        if self.tokens is None:
            (self.tokens, self.prefix_offset,
             self.read_offset) = convert_prompt_ids_to_tokens(
                 tokenizer=tokenizer,
                 prompt_ids=all_input_ids[:-1],
                 skip_special_tokens=params.skip_special_tokens,
             )

        (new_tokens, new_decoded_token_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             tokenizer=tokenizer,
             all_input_ids=all_input_ids,
             prev_tokens=self.tokens,
             prefix_offset=self.prefix_offset,
             read_offset=self.read_offset,
             skip_special_tokens=params.skip_special_tokens,
             spaces_between_special_tokens=params.
             spaces_between_special_tokens,
         )

        # Decode logprobs
        if output_logprobs and (logprobs := output_logprobs[-1]):
            previous_tokens = ensure_list(all_input_ids[:-1])
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
                        prev_tokens=self.tokens,
                        prefix_offset=self.prefix_offset,
                        read_offset=self.read_offset,
                        skip_special_tokens=params.skip_special_tokens,
                        spaces_between_special_tokens=params.
                        spaces_between_special_tokens,
                    )
                    sample_logprob.decoded_token = new_text

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return new_decoded_token_text


def decode_output_tokens(
    output_token_ids: GenericSequence[int],
    prompt_token_ids: List[int],
    tokenizer: AnyTokenizer,
    skip_special_tokens: bool,
) -> str:
    if not prompt_token_ids:
        return tokenizer.decode(output_token_ids,
                                skip_special_tokens=skip_special_tokens)

    # Ensure spaces are handled as if the output tokens were decoded as
    # a full list of ids including the prompt.
    token_ids = prompt_token_ids[-4:]
    prefix_ids_str = tokenizer.decode(token_ids,
                                      skip_special_tokens=skip_special_tokens)
    token_ids.extend(output_token_ids)
    all_ids_str = tokenizer.decode(token_ids,
                                   skip_special_tokens=skip_special_tokens)
    return all_ids_str[len(prefix_ids_str):]


def decode_prompt_logprobs_inplace(all_token_ids: List[int],
                                   prompt_logprobs: List[Optional[Dict[
                                       int, Logprob]]], position_offset: int,
                                   params: SamplingParams,
                                   tokenizer: AnyTokenizer) -> None:
    """Decodes the logprobs for the prompt of a sequence group.

    Args:
        seq_group: The sequence group to decode.
        prompt_logprobs: The logprobs to decode.
        position_offset: Offset of the first index of the logprobs
            relative to the start of the sequence (for chunked prefill).

    Returns:
        The prompt logprobs with the decoded tokens.
    """

    # # Only prompt, without the generated token.
    prompt_token_ids = all_token_ids[:-1]

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
                     skip_special_tokens=params.skip_special_tokens,
                     spaces_between_special_tokens=params.
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


def _replace_none_with_empty(tokens: List[Optional[str]]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: AnyTokenizer,
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
    tokenizer: AnyTokenizer,
    prompt_ids: GenericSequence[int],
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
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)
    return new_tokens, prefix_offset, read_offset


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: AnyTokenizer,
    all_input_ids: GenericSequence[int],
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
