# SPDX-License-Identifier: Apache-2.0

# Copyright 2024- the Outlines developers
# This file is adapted from
# https://github.com/outlines-dev/outlines/blob/main/outlines/serve/vllm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import json
from diskcache import Cache
import diskcache as dc
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

import torch
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from outlines_core.outlines_core import __version__ as outlines_version
from outlines_core.json_schema import build_regex_from_schema
from outlines_core import Vocabulary, Index, Guide
from outlines_core.kernels.torch import (
    allocate_token_bitmask,
    _apply_token_bitmask_inplace_kernel
)
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding.reasoner import Reasoner

logger = init_logger(__name__)

CACHE = None

def get_cache() -> dc.Cache:
    """Get the context object that contains previously-computed return values."""
    outlines_cache_dir = os.getenv("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    home_dir = os.path.expanduser("~")


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

cache = get_cache()

if envs.VLLM_V0_USE_OUTLINES_CACHE:
    logger.warning("Enabling outlines cache. This is an unbounded on-disk "
                   "cache. It may consume a lot of disk space and should "
                   "not be used with untrusted clients.")
    CACHE = get_cache()
else:
    pass


class BaseLogitsProcessor:

    def __init__(self, guide: Guide, vocab_size: int, reasoner: Optional[Reasoner]):
        self._guide: Guide = guide
        self._reasoner: Optional[Reasoner] = reasoner
        self._mask = allocate_token_bitmask(vocab_size)

    
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
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
            if len(input_ids) == 1:
                logger.info(f"advancing with token {input_ids[-1]} from initial state")
            self._guide.advance(token_id=input_ids[-1])
        
        self._guide.write_mask_into(
            data_ptr=self._mask.data_ptr(),
            numel=self._mask.numel(),
            element_size=self._mask.element_size()
        )

        # Any allowed tokens beyond the length of the scores will
        # be ignored by the kernel, taking care of the issue with 
        # models such as Llama 3.2 Vision with an `<|image|>` token
        # with id 128256, but scores.shape == torch.Size([128256])
        _apply_token_bitmask_inplace_kernel(
            logits=scores.unsqueeze(dim=0),
            # mask must be on same device
            mask=self._mask.to(scores.device)
        )
        self._mask.to("cpu")

        return scores

class RegexLogitsProcessor(BaseLogitsProcessor):

    @classmethod
    def _get_guide(cls, regex_string: str, tokenizer: PreTrainedTokenizerBase) -> Guide:
        cached_index = CACHE.get(regex_string) if CACHE is not None else None
        if cached_index is not None:
            return Guide(cached_index)

        vocabulary = get_vocabulary(tokenizer)
        index = Index(regex_string, vocabulary)

        if CACHE is not None:
            CACHE[regex_string] = index

        return Guide(index)

    def __init__(
        self,
        regex_string: str,
        tokenizer: PreTrainedTokenizerBase,
        reasoner: Optional[Reasoner],
        vocab_size: int
    ) -> None:
        super().__init__(
            guide=RegexLogitsProcessor._get_guide(regex_string, tokenizer), 
            vocab_size=vocab_size, 
            reasoner=reasoner)

class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, schema: Union[str, Dict, BaseModel],
                 tokenizer: PreTrainedTokenizerBase,
                 whitespace_pattern: Union[str, None],
                 reasoner: Optional[Reasoner],
                 vocab_size: int) -> None:
        
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


def _reduced_vocabulary(tokenizer: PreTrainedTokenizerBase, eos_token_id: int) -> Dict[str | bytes, List[int]]:
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids.
    
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
        if token in tokenizer.special_tokens: # type: ignore
            continue

        token_str = convert_token_to_string(token)

        if token_str:
            if isinstance(token, bytes):
                # For BPE tokenizers where tokens are stored as bytes.
                token_str = "".join(byte_symbol(b) for b in token) # type: ignore
            elif "\ufffd" in token_str and not re_replacement_seq.match(token):
                # Handle tokens with invalid UTF-8 sequences.
                if re_llama_byte_token.match(token):
                    # Llama-like tokenizers use <0xXX> for incomplete sequences.
                    token_bytes = [int(token[3:5], 16)]
                else:
                    # GPT2-like tokenizers: map each byte back using unicode_to_bytes.
                    token_bytes = [unicode_to_bytes.get(c) for c in token]
                    if None in token_bytes:
                        raise RuntimeError(
                            f"Cannot convert token `{token}` ({token_idx}) to bytes: {token_str}"
                        )
                token_str = "".join(byte_symbol(b) for b in token_bytes) # type: ignore

            if token_idx != eos_token_id:
                vocabulary.setdefault(token_str, []).append(token_idx)
        else:
            empty_token_ids.append(token_idx)

    return vocabulary


def get_vocabulary(tokenizer: PreTrainedTokenizerBase) -> Vocabulary:
    """Get the `Vocabulary` object for a given tokenizer.
    """
    if getattr(tokenizer, "_outlines_vocabulary", False) is not None:
        return tokenizer._outlines_vocabulary

    eos_token_id = None
    if hasattr(
                tokenizer,
                "eos_token_id",
        ) and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
    else:
        raise ValueError(
            f"Error during guided decoding setup: Tokenizer ({type(tokenizer)}) has no `eos_token_id`" 
            " property, but `eos_token_id` is required for guided decoding to work properly.")

    reduced = _reduced_vocabulary(tokenizer, eos_token_id) # type: ignore
    vocabulary = Vocabulary(eos_token_id, reduced)
    setattr(tokenizer, "_outlines_vocabulary", vocabulary)

    return vocabulary
