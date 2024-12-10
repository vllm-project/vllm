from typing import Dict, List, Optional, Tuple, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.sequence import Logprob, SamplingParams, Sequence, SequenceGroup

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1


class Tokenizer(object):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer_kwargs = kwargs

        # layzer_load
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(tokenizer_name=self.tokenizer_name,
                                            **self.tokenizer_kwargs)

        return self._tokenizer

    @classmethod
    def from_engine(cls, engine):
        init_kwargs = dict(
            tokenizer_name=engine.engine_config.model_config.tokenizer,
            tokenizer_mode=engine.engine_config.model_config.tokenizer_mode,
            trust_remote_code=engine.engine_config.model_config.
            trust_remote_code,
            revision=engine.engine_config.model_config.tokenizer_revision)

        return cls(**init_kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

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


def get_cached_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer with cached properties.

    This will patch the tokenizer object in place.

    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown. This
    function caches these properties for faster access."""

    tokenizer_all_special_ids = set(tokenizer.all_special_ids)
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)
    tokenizer_all_special_tokens = set(tokenizer.all_special_tokens)
    tokenizer_len = len(tokenizer)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore

        @property
        def all_special_ids(self):
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self):
            return tokenizer_all_special_tokens

        @property
        def all_special_tokens_extended(self):
            return tokenizer_all_special_tokens_extended

        def __len__(self):
            return tokenizer_len

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    tokenizer.__class__ = CachedTokenizer
    return tokenizer


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope.
    """
    if VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        import os

        import huggingface_hub
        from modelscope.hub.snapshot_download import snapshot_download

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not os.path.exists(tokenizer_name):
            tokenizer_path = snapshot_download(
                model_id=tokenizer_name,
                cache_dir=download_dir,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                # Ignore weights - we only need the tokenizer.
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])
            tokenizer_name = tokenizer_path

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if (not trust_remote_code and
            ("does not exist or is not currently imported." in str(e)
             or "requires you to execute the tokenizer file" in str(e))):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    except AttributeError as e:
        if "BaichuanTokenizer" in str(e):
            # This is for the error "'BaichuanTokenizer' object has no
            # attribute 'sp_model'".
            from vllm.transformers_utils.tokenizers import BaichuanTokenizer
            tokenizer = BaichuanTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs)
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return get_cached_tokenizer(tokenizer)


def _replace_none_with_empty(tokens: List[Optional[str]]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


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
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)
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
