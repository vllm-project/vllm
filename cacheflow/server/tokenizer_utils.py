from typing import Union

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
