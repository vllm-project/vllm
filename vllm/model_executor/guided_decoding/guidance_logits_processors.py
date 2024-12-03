import json
import os
from typing import Any, List, Type, Union

import llguidance  # type: ignore[import-untyped]
import numpy as np
import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from vllm.model_executor.guided_decoding.guidance_utils import (
    LLInterpreterResponse, TransformersTokenizer)


class GuidanceLogitsProcessor:
    """Base Guidance Logits Processor"""
    metadata: dict[str, Any] = {}

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

        self.is_stopped = False
        self.pending_ff_tokens: list[int] = []
        self.new_sampling = False
        self.initialized = False

    def _initialize(self):
        if self.initialized:
            return

        if self.mode.lower() == "json":
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
            self.serialized_grammar = compiler.compile(schema)
        elif self.mode.lower() in ["regex", "choice"]:
            compiler = llguidance.RegexCompiler()
            self.serialized_grammar = compiler.compile(regex=self.guide)
        elif self.mode.lower() == "grammar":
            serialized_grammar = self.guide
            if isinstance(self.guide, dict):
                serialized_grammar = json.dumps(self.guide)
            self.serialized_grammar = serialized_grammar

        if f"guidance_tokenizer_{self.tokenizer_name}" not in self.metadata:
            self.metadata[
                f"guidance_tokenizer_{self.tokenizer_name}"] = \
                    TransformersTokenizer( \
                        model=self.tokenizer.name_or_path,
                        transformers_tokenizer=self.tokenizer)
        self.guidance_tokenizer = self.metadata[
            f"guidance_tokenizer_{self.tokenizer_name}"]

        if f"ll_tokenizer_{self.tokenizer_name}" not in self.metadata:
            self.metadata[
                f"ll_tokenizer_{self.tokenizer_name}"] = llguidance.LLTokenizer(
                    llguidance.TokenizerWrapper(self.guidance_tokenizer))
        self.ll_tokenizer = self.metadata[
            f"ll_tokenizer_{self.tokenizer_name}"]

        self.ll_interpreter = llguidance.LLInterpreter(
            self.ll_tokenizer,
            self.serialized_grammar,
            enable_backtrack=False,
            enable_ff_tokens=False,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        self.initialized = True

    def __call__(
        self,
        prompt_tokens_ids: List[int],
        past_tokens_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # we initialize the guidance model here
        # to avoid pickling ll_tokenizer and ll_interpreter
        self._initialize()

        if self.is_stopped:
            return logits

        if self.new_sampling and len(past_tokens_ids) > 0:
            backtrack, ff_tokens = self.ll_interpreter.commit_token(
                past_tokens_ids[-1])
            if len(ff_tokens) > 0 and backtrack == 0:
                # first token is last generated token
                ff_tokens = ff_tokens[1:]
            self.pending_ff_tokens.extend(ff_tokens)
            self.new_sampling = False

        if len(self.pending_ff_tokens) > 0:
            # if we have pending fast-forward tokens,
            # just return them immediately
            ff_token = self.pending_ff_tokens.pop(0)
            logits.add_(-logits)
            logits[ff_token] = 200.0
            return logits

        mask, resp = self.ll_interpreter.compute_mask()
        r = LLInterpreterResponse.model_validate_json(resp)

        if r.stop:
            mask = np.zeros(logits.shape[-1], dtype=np.uint8)
            if self.guidance_tokenizer.eos_token_id is not None:
                mask[self.guidance_tokenizer.eos_token_id] = 200
            self.is_stopped = True
        elif mask is None:
            # NOTE: mask should not be None unless r.stop is True
            # However, we are handling this case just in case
            # llguidance allows free-style generation
            mask = np.zeros(logits.shape[-1], dtype=np.uint8)
        else:
            mask = np.frombuffer(mask, dtype=np.uint8)

        # Force all invalid tokens to have 0 value
        logits.add_(-torch.min(logits))
        zero_indices = np.where(mask == 0)[0]
        logits[zero_indices] = 0.0
        non_zero_indices = np.nonzero(mask)[0]
        logits[non_zero_indices] += 200.0
        # set special tokens not in vocab to 0
        logits[mask.shape[0]:] = 0.0
        self.new_sampling = True

        return logits
