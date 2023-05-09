from typing import Union

from transformers import (AutoConfig, AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)


def get_tokenizer(
    model_name: str,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "llama":
        # LLaMA fast tokenizer has a bug related to protobuf.
        # See https://github.com/WoosukKwon/cacheflow/issues/80#issue-1698550554
        if "use_fast" in kwargs:
            kwargs.pop("use_fast")
        return AutoTokenizer.from_pretrained(
            model_name, use_fast=False, *args, **kwargs)
    return AutoTokenizer.from_pretrained(model_name, *args, **kwargs)
