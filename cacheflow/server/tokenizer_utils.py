from typing import List, Tuple, Union

from transformers import (AutoConfig, AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

_MODEL_TYPES_WITH_SLOW_TOKENIZER = [
    # LLaMA fast tokenizer has a bug related to protobuf.
    # See https://github.com/WoosukKwon/cacheflow/issues/80#issue-1698550554
    "llama",
]


def get_tokenizer(
    model_name: str,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type in _MODEL_TYPES_WITH_SLOW_TOKENIZER:
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(model_name, *args, **kwargs)


def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prev_output_tokens: List[str],
    new_token_id: int,
    skip_special_tokens: bool,
) -> Tuple[str, str]:
    """Detokenizes the new token in conjuction with the previous output tokens.

    NOTE: This function does not update prev_output_tokens.

    Returns:
        new_token: The new token as a string.
        output_text: The new output text as a string.
    """
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    output_tokens = prev_output_tokens + [new_token]

    # Convert the tokens to a string.
    # We optimize tokenizer._decode() by assuming that the tokenizer does not
    # have added_tokens_encoder.
    if hasattr(tokenizer, "added_tokens_encoder"):
        assert not tokenizer.added_tokens_encoder
    output_text = tokenizer.convert_tokens_to_string(output_tokens)
    return new_token, output_text
