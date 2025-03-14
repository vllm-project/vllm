# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import Any, List, Type, Union

import llguidance  # type: ignore[import-untyped]
import llguidance.hf
import llguidance.torch
import torch
from llguidance.gbnf_to_lark import any_to_lark  # type: ignore[import-untyped]
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase


class GuidanceLogitsProcessor:
    """Base Guidance Logits Processor"""

    cached_tokenizers: dict[str, Any] = {}

    def __init__(
        self,
        mode: str,
        guide: Union[dict, Type[BaseModel], str],
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Union[str, None] = None,
    ) -> None:
        """Base Guidance Logits Processor

        Args:
            mode (str)
                guided generation mode. 
                Must be one of "json", "regex", "choice", "grammar"
            guide (Union[dict, Type[BaseModel], str])
                guide for guided generation
            tokenizer (PreTrainedTokenizerBase)
                model's tokenizer
            whitespace_pattern (Union[str, None], optional)
                Json-string to indicate pattern to use \
                    for JSON syntactic whitespace
                Example: '{"whitespace_flexible":true}'
        """
        self.mode = mode
        self.guide = guide
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer.name_or_path
        self.whitespace_pattern = whitespace_pattern

        self.pending_ff_tokens: list[int] = []
        self.new_sampling = False
        self.initialized = False

    def _get_serialized_grammar(self):
        if self.mode.lower() in ["json", "json_object"]:
            if self.mode.lower() == "json_object":
                schema = "{}"
            if isinstance(self.guide, dict):
                schema = json.dumps(self.guide)
            elif isinstance(self.guide, BaseModel):
                schema = json.dumps(self.guide.model_json_schema())
            else:
                schema = str(self.guide)

            whitespaces_config = {}
            if isinstance(self.whitespace_pattern, str):
                whitespaces_config = json.loads(self.whitespace_pattern)

            whitespace_flexible = whitespaces_config.get(
                "whitespace_flexible", False)
            compiler = llguidance.JsonCompiler(
                whitespace_flexible=whitespace_flexible)
            return compiler.compile(schema)
        elif self.mode.lower() in ["regex", "choice"]:
            compiler = llguidance.RegexCompiler()
            return compiler.compile(regex=self.guide)
        elif self.mode.lower() == "grammar":
            # grammar can be in EBNF or LARK syntax
            compiler = llguidance.LarkCompiler()
            return compiler.compile(any_to_lark(self.guide))

        raise ValueError(f"Invalid mode: {self.mode}")

    def _initialize(self):
        if self.initialized:
            return

        self.serialized_grammar = self._get_serialized_grammar()
        ll_tokenizer = self.cached_tokenizers.get(self.tokenizer.name_or_path,
                                                  None)
        if ll_tokenizer is None:
            ll_tokenizer = llguidance.hf.from_tokenizer(self.tokenizer, None)
            self.cached_tokenizers[self.tokenizer.name_or_path] = ll_tokenizer

        self.ll_tokenizer = ll_tokenizer
        self.ll_interpreter = llguidance.LLInterpreter(
            self.ll_tokenizer,
            self.serialized_grammar,
            enable_backtrack=False,
            enable_ff_tokens=False,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        # create reusable bitmask
        self.bitmask = llguidance.torch.allocate_token_bitmask(
            1, self.ll_tokenizer.vocab_size)

        self.initialized = True

    def __call__(
        self,
        input_ids: List[int],
        scores: torch.Tensor,
    ) -> torch.Tensor:
        # we initialize the guidance model here
        # to avoid pickling ll_tokenizer and ll_interpreter
        self._initialize()

        if self.ll_interpreter.has_pending_stop():
            if self.ll_tokenizer.eos_token is not None:
                scores.add_(-scores)
                scores[self.ll_tokenizer.eos_token] = 200.0

            return scores

        if self.new_sampling and len(input_ids) > 0:
            backtrack, ff_tokens = self.ll_interpreter.commit_token(
                input_ids[-1])
            if len(ff_tokens) > 0 and backtrack == 0:
                # first token is last generated token
                ff_tokens = ff_tokens[1:]
            self.pending_ff_tokens.extend(ff_tokens)
            self.new_sampling = False

        if len(self.pending_ff_tokens) > 0:
            # if we have pending fast-forward tokens,
            # just return them immediately
            ff_token = self.pending_ff_tokens.pop(0)
            scores.add_(-scores)
            scores[ff_token] = 200.0
            return scores

        llguidance.torch.fill_next_token_bitmask(self.ll_interpreter,
                                                 self.bitmask, 0)
        llguidance.torch.apply_token_bitmask_inplace(
            scores, self.bitmask.to(scores.device))
        self.new_sampling = True

        return scores


class JsonGuidanceLogitsProcessor(GuidanceLogitsProcessor):
    """Json Guidance Logits Processor"""

    def __init__(
        self,
        guide: Union[dict, Type[BaseModel], str],
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Union[str, None] = None,
    ):
        super().__init__("json", guide, tokenizer, whitespace_pattern)


class RegexGuidanceLogitsProcessor(GuidanceLogitsProcessor):
    """Regex Guidance Logits Processor"""

    def __init__(
        self,
        guide: str,
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Union[str, None] = None,
    ):
        super().__init__("regex", guide, tokenizer, whitespace_pattern)


class ChoiceGuidanceLogitsProcessor(GuidanceLogitsProcessor):
    """Choice Guidance Logits Processor"""

    def __init__(
        self,
        guide: str,
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Union[str, None] = None,
    ):
        super().__init__("choice", guide, tokenizer, whitespace_pattern)


class GrammarGuidanceLogitsProcessor(GuidanceLogitsProcessor):
    """Grammar Guidance Logits Processor"""

    def __init__(
        self,
        guide: str,
        tokenizer: PreTrainedTokenizerBase,
        whitespace_pattern: Union[str, None] = None,
    ):
        super().__init__("grammar", guide, tokenizer, whitespace_pattern)
