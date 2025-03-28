# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class XgrammarBackend(StructuredOutputBackend):

    def __post_init__(self):
        self.disable_any_whitespace = (
            "disable-any-whitespace"
            in self.vllm_config.decoding_config.guided_decoding_backend)

        if isinstance(self.tokenizer, MistralTokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            try:
                encoded_vocab = [
                    token for token, _ in sorted(
                        self.tokenizer.get_vocab().items(),
                        key=lambda x: x[1],
                    )
                ]
                stop_token_ids = None
                if hasattr(
                        self.tokenizer,
                        "eos_token_id",
                ) and self.tokenizer.eos_token_id is not None:
                    stop_token_ids = [self.tokenizer.eos_token_id]
            except AttributeError as e:
                raise ValueError(
                    f"Cannot get the vocabulary of the tokenizer "
                    f"{type(self.tokenizer)}. The tokenizer should have a "
                    "get_vocab method.") from e
            tokenizer_info = xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=encoded_vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                vocab_type=xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_builtin_json_grammar()
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        return XgrammarGrammar(
            # NOTE: conservatively fixed this value for now
            # given that the length of jf-forward string in theory
            # won't have a limit
            matcher=xgr.GrammarMatcher(ctx, max_rollback_tokens=50),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def encode_with_jump(
        self,
        output_token_ids: list[int],
        jump_forward_string: str,
    ) -> list[int]:
        ...


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.", request_id, token)
                return False
        return True

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self.matcher.is_terminated()

    def jump_forward_string(self) -> str | None:
        jf_string = self.matcher.find_jump_forward_string()
        return jf_string if jf_string else None

    def find_token_divergence(
        self,
        request_id: str,
        prev_tokens: list[int],
        combined_tokens: list[int],
    ) -> int:
        min_len = min(len(prev_tokens), len(combined_tokens))
        k = sum(1 for _ in itertools.takewhile(
            lambda x: x[0] == x[1],
            zip(prev_tokens[:(min_len)], combined_tokens[:min_len]),
        ))

        # We have to rollback the tokens to the divergence point
        if k < len(prev_tokens):
            self.matcher.rollback(len(prev_tokens) - k)
        assert self.accept_tokens(request_id, combined_tokens[k:])
        return k
