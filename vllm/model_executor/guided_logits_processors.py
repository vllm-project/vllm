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
import json
import math
from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Optional, Union

import torch
from outlines.fsm.fsm import CFGFSM, RegexFSM
from outlines.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase


class BaseLogitsProcessor:

    def adapt_tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        """Adapt vLLM's tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. The decoder of outlines, returns a list whereas
        the decode of vLLM returns an str. To sync the vLLM decoder with
        outlines internal api, the decoder should be adapted. In addition
        we need to handle the missing spaces to Llama's tokenizer to be
        able to compile FSMs for this model.

        """
        if getattr(tokenizer, "_outlines_adapted", False):
            return tokenizer

        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE

            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        def change_decoder(
            decoder: Callable[[List[int]], str]
        ) -> Callable[[List[int]], List[str]]:
            """Sync vLLM's decoder with the outlines by returning list."""

            def new_decoder(inp_tokens: List[int]) -> List[str]:
                return [decoder(inp_tokens)]

            return new_decoder

        tokenizer.convert_token_to_string = convert_token_to_string
        tokenizer.decode = change_decoder(tokenizer.decode)
        setattr(tokenizer, "_outlines_adapted", True)  # noqa: B010

        return tokenizer

    def init_state(self):
        """Initialize the FSM states."""
        self.fsm_state: DefaultDict[int, int] = defaultdict(int)

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""

        seq_id = hash(tuple(input_ids))

        if len(input_ids) == 0:
            self.init_state()
        else:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self.fsm_state[seq_id] = self.fsm.next_state(
                self.fsm_state[last_seq_id], last_token)

        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state[seq_id])

        mask = torch.full((scores.shape[-1], ),
                          -math.inf,
                          device=scores.device)
        mask[allowed_tokens] = 0
        scores.add_(mask)

        return scores


class RegexLogitsProcessor(BaseLogitsProcessor):

    def __init__(self, regex_string: str, tokenizer: PreTrainedTokenizerBase):
        """Compile the FSM that drives the regex-structured generation.

        Parameters
        ----------
        regex_string
            A string that represents a regular expression
        tokenizer
            The model's tokenizer

        """
        tokenizer = self.adapt_tokenizer(tokenizer)
        fsm = RegexFSM(regex_string, tokenizer)
        self.fsm = fsm


class JSONLogitsProcessor(RegexLogitsProcessor):

    def __init__(self,
                 schema: Union[str, Dict, BaseModel],
                 tokenizer: PreTrainedTokenizerBase,
                 whitespace_pattern: Optional[str] = None):
        """Compile the FSM that drives the JSON-guided generation.

        Parameters
        ----------
        schema
            A JSON schema that encodes the structure we want the model to
            generate
        tokenizer
            The model's tokenizer
        whitespace_pattern
            Pattern to use for JSON syntactic whitespace (doesn't impact
            string literals)
            Example: allow only a single space or newline with
            `whitespace_pattern=r"[\n ]?"`
        """
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
        super().__init__(regex_string, tokenizer)


class CFGLogitsProcessor(BaseLogitsProcessor):

    def __init__(self, cfg: str, tokenizer: PreTrainedTokenizerBase):
        """Compile the FSM that drives the context free grammar generation.

        Parameters
        ----------
        cfg
            A string that represents a context-free grammar
        tokenizer
            The model's tokenizer

        """
        tokenizer = self.adapt_tokenizer(tokenizer)
        fsm = CFGFSM(cfg, tokenizer)
        self.fsm = fsm
