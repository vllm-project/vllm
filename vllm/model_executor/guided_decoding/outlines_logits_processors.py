# SPDX-License-Identifier: Apache-2.0

# Copyright 2024- the Outlines developers
# This file is adapted from
# https://github.com/outlines-dev/outlines/blob/main/outlines/serve/vllm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata
import json
import os
import re
from typing import Dict, List, Optional, Union

import diskcache as dc
import torch
from cachetools import LRUCache
from diskcache import Cache
from outlines_core import Guide, Index, Vocabulary
from outlines_core.json_schema import build_regex_from_schema
from outlines_core.kernels.torch import (_apply_token_bitmask_inplace_kernel,
                                         allocate_token_bitmask)
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding.reasoner import Reasoner
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

CACHE = None


def get_disk_cache() -> dc.Cache:
    """Get the context object that contains previously-computed return values"""
    outlines_cache_dir = os.getenv("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    home_dir = os.path.expanduser("~")
    outlines_version = importlib.metadata.version("outlines_core")

    if outlines_cache_dir:
        # OUTLINES_CACHE_DIR takes precendence
        cache_dir = outlines_cache_dir
    elif xdg_cache_home:
        cache_dir = os.path.join(xdg_cache_home, ".cache", "outlines")
    elif home_dir != "/":
        cache_dir = os.path.join(home_dir, ".cache", "outlines")
    else:
        import tempfile

        # home_dir may be / inside a docker container without existing user
        tempdir = tempfile.gettempdir()
        cache_dir = os.path.join(tempdir, ".cache", "outlines")

    memory = Cache(cache_dir, eviction_policy="none", cull_limit=0)

    # Ensure if a version upgrade occurs, old cache is pruned
    cached_version = memory.get('__version__', None)
    if cached_version != outlines_version:
        memory.clear()
    memory.set('__version__', outlines_version)

    return memory


if envs.VLLM_V0_USE_OUTLINES_CACHE:
    logger.warning("Enabling outlines cache. This is an unbounded on-disk "
                   "cache. It may consume a lot of disk space and should "
                   "not be used with untrusted clients.")
    CACHE = get_disk_cache()
else:
    CACHE = LRUCache(maxsize=128)


class BaseLogitsProcessor:

    def __init__(self, guide: Guide, vocab_size: int,
                 reasoner: Optional[Reasoner]):
        self._guide: Guide = guide
        self._reasoner: Optional[Reasoner] = reasoner
        self._mask = allocate_token_bitmask(vocab_size)

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        # Skip the structured logits processing if reasoning is not finished.
        # reasoner is not None only when `--enable-reasoning` is set.
        if self._reasoner is not None:
            if not self._reasoner.is_reasoning_end(input_ids):
                return scores
            else:
                # Remove the reasoning tokens from the input_ids
                # We need this because our implementation relies on the
                # input_ids sequence to store the FSM state.
                input_ids = self._reasoner.extract_content(input_ids)

        if len(input_ids) > 0:
            self._guide.advance(token_id=input_ids[-1], return_tokens=False)

        self._guide.write_mask_into(data_ptr=self._mask.data_ptr(),
                                    numel=self._mask.numel(),
                                    element_size=self._mask.element_size())

        # Any allowed tokens beyond the length of the scores will
        # be ignored by the kernel, taking care of the issue with
        # models such as Llama 3.2 Vision with an `<|image|>` token
        # with id 128256, but scores.shape == torch.Size([128256])
        _apply_token_bitmask_inplace_kernel(
            logits=scores.unsqueeze(dim=0),
            # mask must be on same device
            mask=self._mask.to(scores.device))
        self._mask.to("cpu")

        return scores


class RegexLogitsProcessor(BaseLogitsProcessor):

    @classmethod
    def _get_guide(cls, regex_string: str,
                   tokenizer: PreTrainedTokenizerBase) -> Guide:
        vocabulary = get_vocabulary(tokenizer)  # type: ignore[arg-type]
        cache_key = f"{vocabulary._hash}_{regex_string}"
        if CACHE is not None and cache_key in CACHE:
            return Guide(CACHE[cache_key])

        index = Index(regex_string, vocabulary)

        if CACHE is not None:
            CACHE[cache_key] = index

        return Guide(index)

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase,
                 reasoner: Optional[Reasoner], vocab_size: int) -> None:
        super().__init__(guide=RegexLogitsProcessor._get_guide(
            regex_string, tokenizer),
                         vocab_size=vocab_size,
                         reasoner=reasoner)


class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, schema: Union[str, Dict, BaseModel],
                 tokenizer: PreTrainedTokenizerBase,
                 whitespace_pattern: Union[str, None],
                 reasoner: Optional[Reasoner], vocab_size: int) -> None:

        if isinstance(schema, type(BaseModel)):
            schema_str = json.dumps(schema.model_json_schema())
        elif isinstance(schema, Dict):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            raise ValueError(
                f"Cannot parse schema {schema}. The schema must be either "
                f"a Pydantic object, a dictionary or a string that contains "
                f"the JSON Schema specification")

        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        super().__init__(regex_string, tokenizer, reasoner, vocab_size)


def _reduced_vocabulary(tokenizer: AnyTokenizer,
                        eos_token_id: int) -> Dict[str | bytes, List[int]]:
    """Create a map from vocabulary tokens to lists of equivalent token ids.
    
    Returns:
        A Dict of token string -> equivalent token ids
    """
    unicode_to_bytes = {v: k for k, v in bytes_to_unicode().items()}

    re_llama_byte_token = re.compile(r"^<0x[0-9A-F]{2}>$")
    re_replacement_seq = re.compile(r"^▁* +\.*$")

    def byte_symbol(byte: int) -> str:
        return f"\x00{byte:02X}" if byte >= 0x80 else chr(byte)

    def convert_token_to_string(token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if (type(token) is str and token.startswith(SPIECE_UNDERLINE)
                or token == "<0x20>"):
            return " " + string

        return string

    vocabulary: dict[str | bytes, list[int]] = {}
    empty_token_ids: list[int] = []
    for token, token_idx in tokenizer.get_vocab().items():
        if token in tokenizer.special_tokens:  # type: ignore
            continue

        token_str = convert_token_to_string(token)

        if token_str:
            if isinstance(token, bytes):
                # For BPE tokenizers where tokens are stored as bytes.
                token_str = "".join(byte_symbol(b)
                                    for b in token)  # type: ignore
            elif "\ufffd" in token_str and not re_replacement_seq.match(token):
                # Handle tokens with invalid UTF-8 sequences.
                if re_llama_byte_token.match(token):
                    # Llama-like tokenizers use <0xXX> for incomplete sequences.
                    token_bytes = [int(token[3:5], 16)]
                else:
                    # GPT2 tokenizers: map each byte back using unicode_to_bytes
                    token_bytes = [unicode_to_bytes.get(c) for c in token]
                    if None in token_bytes:
                        raise RuntimeError(
                            f"Cannot convert token `{token}`"
                            f" ({token_idx}) to bytes: {token_str}"
                        )
                token_str = "".join(byte_symbol(b) # type: ignore[arg-type]
                                    for b in token_bytes)

            if token_idx != eos_token_id:
                vocabulary.setdefault(token_str, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)

    return vocabulary


def get_vocabulary(tokenizer: AnyTokenizer) -> Vocabulary:
    """Get the `Vocabulary` object for a given tokenizer.
    """
    if hasattr(tokenizer, "_outlines_vocabulary") is not None:
        return tokenizer._outlines_vocabulary

    try:
        if hasattr(
                tokenizer,
                "eos_token_id",
        ) and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                f"Error during guided decoding setup: Tokenizer"
                f" ({type(tokenizer)}) has no `eos_token_id` property, "
                "but `eos_token_id` is required for guided decoding"
                " to work properly."
            )

        reduced_vocab = _reduced_vocabulary(
            tokenizer,
            eos_token_id  #type: ignore
        )
        vocabulary = Vocabulary(eos_token_id, reduced_vocab)
        vocabulary._hash = hash(vocabulary.__repr__())
        tokenizer._outlines_vocabulary = vocabulary

        return vocabulary
    except AttributeError as e:
        raise ValueError(f"Cannot get the vocabulary of the tokenizer "
                         f"({type(tokenizer)}). The tokenizer should have a "
                         "get_vocab method.") from e
