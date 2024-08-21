import os
import re
from pathlib import Path
from typing import Optional, Union, List

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizers import BaichuanTokenizer
from vllm.utils import make_async

logger = init_logger(__name__)

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def download_mistral_tokenizer_from_hf(tokenizer_name: str, revision: str) -> str:
    api = HfApi()
    repo_info = api.model_info(tokenizer_name)
    files = [s.rfilename for s in repo_info.siblings]
    pattern = re.compile(r'^tokenizer\.model\..*$|^tekken\.json$')

    matched_files = [file for file in files if pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(f"Found {len(matched_files)} files matching the pattern: {matched_files}. Make sure only one Mistral tokenizer is present in {tokenizer_name}.")
    elif len(matched_files) == 0: 
        raise OSError(f"Found {len(matched_files)} files matching the pattern: {matched_files}. Make sure that a Mistral tokenizer is present in {tokenizer_name}.")

    tokenizer_file = hf_hub_download(tokenizer_name, filename=matched_files[0], revision=revision)
    return tokenizer_file



def get_cached_tokenizer(tokenizer: AnyTokenizer) -> AnyTokenizer:
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


class VLLMMistralTokenizer:

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        self.mistral = tokenizer
        self.instruct = tokenizer.instruct_tokenizer
        self.tokenizer = tokenizer.instruct_tokenizer.tokenizer

        self.vocab_size = len(self.tokenizer.vocab())
        self.is_mistral = True

    def encode(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt, bos=False, eos=False)

    def convert_tokens_to_string(self, ids: List[str]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_id

    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> List[str]:
        # TODO(Patrick) - potentially allow special tokens to not be skipped
        assert skip_special_tokens, "Skipping special tokens is not supported for Mistral tokenizers."
        return [self.tokenizer.id_to_piece(id) for id in ids]

    def __len__(self):
        return self.vocab_size


def get_tokenizer(
    tokenizer_name: Union[str, Path],
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope.
    """
    if VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
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

    # Separate model folder from file path for GGUF models
    is_gguf = Path(tokenizer_name).is_file() and Path(
        tokenizer_name).suffix == ".gguf"
    if is_gguf:
        kwargs["gguf_file"] = Path(tokenizer_name).name
        tokenizer_name = Path(tokenizer_name).parent

    if tokenizer_mode == "mistral":
        tokenizer_file = download_mistral_tokenizer_from_hf(tokenizer_name, revision)

        mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)
        tokenizer = VLLMMistralTokenizer(mistral_tokenizer)
    else:
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
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer


def get_lora_tokenizer(lora_request: LoRARequest, *args,
                       **kwargs) -> Optional[AnyTokenizer]:
    if lora_request is None:
        return None
    try:
        tokenizer = get_tokenizer(lora_request.lora_path, *args, **kwargs)
    except OSError as e:
        # No tokenizer was found in the LoRA folder,
        # use base model tokenizer
        logger.warning(
            "No tokenizer found in %s, using base model tokenizer instead. "
            "(Exception: %s)", lora_request.lora_path, e)
        tokenizer = None
    return tokenizer


get_lora_tokenizer_async = make_async(get_lora_tokenizer)
