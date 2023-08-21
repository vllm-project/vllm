from typing import List, Tuple, Union, Dict

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        logger.info(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer.")
        raise RuntimeError(err_msg) from e
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

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return tokenizer


def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prev_output_tokens: List[str],
    new_token_id: int,
    skip_special_tokens: bool,
) -> Tuple[str, str]:
    """Detokenizes the new token in conjunction with the previous output tokens.

    NOTE: This function does not update prev_output_tokens.

    Returns:
        new_token: The new token as a string.
        output_text: The new output text as a string.
    """
    if skip_special_tokens and (new_token_id in tokenizer.all_special_ids):
        return None, prev_output_tokens
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    output_tokens = prev_output_tokens + [new_token]

    # Convert the tokens to a string.
    # Optimization: If the tokenizer does not have `added_tokens_encoder`,
    # then we can directly use `convert_tokens_to_string`.
    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(output_tokens)
        return new_token, output_text

    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    for token in output_tokens:
        if skip_special_tokens and token in tokenizer.all_special_tokens:
            continue
        if token in tokenizer.added_tokens_encoder:
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
    output_text = " ".join(sub_texts)
    return new_token, output_text


class SequenceDetokenizeState:
    def __init__(self, seq_id: int, stop_strings: List[str]) -> None:
        self.seq_id: int = seq_id
        self.stop_strings: List[str] = stop_strings
        self.output_text: str = ""
        self.output_tokens: List[str] = []
        self.stop_string_matched = False


class Detokenizer:
    def __init__(
        self, 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.tokenizer = tokenizer
        self.decoding_sequences: Dict[int, SequenceDetokenizeState] = dict()

    def add_sequence(self, seq_id: int, stop_strings: List[str]) -> None:
        self.decoding_sequences[seq_id] = SequenceDetokenizeState(seq_id, stop_strings) 

    def detokenize_last_token(self, seq_id: int, last_token_id: int) -> None:
        assert seq_id in self.decoding_sequences, f"{self.decoding_sequences.keys()}"
        state = self.decoding_sequences[seq_id]

        new_token, new_output_text = detokenize_incrementally(
            self.tokenizer,
            state.output_tokens,
            last_token_id,
            skip_special_tokens=True,
        )
        if new_token is None:
            return
        for stop_str in state.stop_strings:
            if new_output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                new_output_text = new_output_text[:-len(stop_str)]
                # TODO: propogate stop_string_matched state
                state.stop_string_matched = True
                break

        state.output_tokens.append(new_token)
        state.output_text = new_output_text

    def get_output_text(self, seq_id: int) -> str:
        if seq_id in self.decoding_sequences:
            return self.decoding_sequences[seq_id].output_text
        return ""

    def free_sequence(self, seq_id: int) -> None:
        self.decoding_sequences.pop(seq_id)

    def stop_string_matched(self, seq_id: int) -> bool:
        assert seq_id in self.decoding_sequences
        return self.decoding_sequences[seq_id].stop_string_matched
