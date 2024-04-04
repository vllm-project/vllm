from typing import Optional, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizers import *
from vllm.utils import make_async

logger = init_logger(__name__)


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

    class CachedTokenizer(tokenizer.__class__):

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
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
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
            tokenizer = BaichuanTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return get_cached_tokenizer(tokenizer)


def get_lora_tokenizer(lora_request: LoRARequest, *args,
                       **kwargs) -> Optional[PreTrainedTokenizer]:
    if lora_request is None:
        return None
    try:
        tokenizer = get_tokenizer(lora_request.lora_local_path, *args,
                                  **kwargs)
    except OSError as e:
        # No tokenizer was found in the LoRA folder,
        # use base model tokenizer
        logger.warning(
            f"No tokenizer found in {lora_request.lora_local_path}, "
            "using base model tokenizer instead. "
            f"(Exception: {str(e)})")
        tokenizer = None
    return tokenizer


get_lora_tokenizer_async = make_async(get_lora_tokenizer)
