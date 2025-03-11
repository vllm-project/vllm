# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(),
                                  "llguidance.torch")

logger = init_logger(__name__)


class GuidanceBackend(StructuredOutputBackend):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        tokenizer_group = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)  # type: ignore[arg-type]
        tokenizer_group.ping()
        self.vllm_config = vllm_config
        self.vocab_size = vllm_config.model_config.get_vocab_size()

        tokenizer = tokenizer_group.get_lora_tokenizer(None)
        self.ll_tokenizer = llguidance_hf.from_tokenizer(tokenizer, None)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:

        if request_type == StructuredOutputOptions.JSON:
            if isinstance(grammar_spec, dict):
                schema = json.dumps(grammar_spec)
            else:
                schema = str(grammar_spec)

            # TODO: make whitespace_flexible configurable
            compiler = llguidance.JsonCompiler(whitespace_flexible=False)
            self.serialized_grammar = compiler.compile(schema)
        elif (request_type == StructuredOutputOptions.REGEX
              or request_type == StructuredOutputOptions.CHOICE):
            compiler = llguidance.RegexCompiler()
            self.serialized_grammar = compiler.compile(regex=grammar_spec)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            if isinstance(grammar_spec, dict):
                self.serialized_grammar = json.dumps(grammar_spec)
            else:
                self.serialized_grammar = str(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        ll_interpreter = llguidance.LLInterpreter(
            self.ll_tokenizer,
            self.serialized_grammar,
            enable_backtrack=False,
            enable_ff_tokens=False,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        return GuidanceGrammar(
            ll_interpreter=ll_interpreter,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size)


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):

    ll_interpreter: llguidance.LLInterpreter
    ll_tokenizer: llguidance_hf.LLTokenizer
    vocab_size: int
    stopped: bool = False

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """

        if self.stopped:
            return True

        for token in tokens:
            # TODO - Add jump decoding support in the future.
            # For now we turn this off when creating the LLInterpreter.
            #backtrack, ff_tokens = self.ll_interpreter.commit_token(token)
            self.ll_interpreter.commit_token(token)

        return True

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        if self.ll_interpreter.has_pending_stop():
            # fill bitmask with eos token before is_terminated() return True
            eos_token = self.ll_tokenizer.eos_token
            bitmask[idx, :] = 0
            bitmask[idx, eos_token // 32] = 1 << (eos_token % 32)
            self.stopped = True
        else:
            llguidance_torch.fill_next_token_bitmask(self.ll_interpreter,
                                                     bitmask, idx)

    def is_terminated(self) -> bool:
        return self.stopped

    def reset(self):
        # This method may be not needed anymore? TODO
        pass
