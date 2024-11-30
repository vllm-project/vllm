from __future__ import annotations

import json, torch

from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Any, Optional, List

try:
    import xgrammar as xgr
    from xgrammar.base import _core as xgr_core
except ImportError:
    pass

if TYPE_CHECKING:
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.config import ModelConfig
    from transformers import PreTrainedTokenizer


# TODO: passing batch size to max threads here
def get_local_xgrammar_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        max_threads=8):
    config = GrammarConfig.from_guided_params(guided_params=guided_params,
                                              model_config=model_config,
                                              tokenizer=tokenizer,
                                              max_threads=max_threads)
    return XGrammarLogitsProcessor(config)


@dataclass
class GrammarConfig:
    """Serializable configuration for grammar compilation"""
    vocab_size: int = 0
    max_threads: int = 8
    json_str: Optional[str] = None
    grammar_str: Optional[str] = None
    encoded_vocab: Optional[Dict[str, int]] = None
    stop_token_ids: Optional[List[int]] = None
    backend_str: str = ""

    @classmethod
    def from_guided_params(cls,
                           guided_params: GuidedDecodingParams,
                           model_config: ModelConfig,
                           tokenizer: PreTrainedTokenizer,
                           max_threads: int = 8) -> GrammarConfig:

        # Vendorred from xgrammar logics
        try:
            encoded_vocab = tokenizer.get_vocab()
            encoded_vocab = [
                token for token, _ in sorted(encoded_vocab.items(),
                                             key=lambda x: x[1])
            ]
        except AttributeError as e:
            msg = (
                f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer "
                "should have a get_vocab method.")
            raise ValueError(msg) from e

        stop_token_ids = None
        backend_str = xgr.VocabType.RAW
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            # huggingface fast tokenizer
            # - the vocabulary is directly obtained from tokenizer.get_vocab()
            #   (tokenizer.backend_tokenizer.to_str() may not contain the full vocab, special
            #   tokens may be omitted)
            # - the vocab size is obtained from len(tokenizer.get_vocab()) or provided by user
            # - the vocab type and prepend_space_in_tokenization are obtained from
            #   tokenizer.backend_tokenizer.to_str()
            # - stop token id is provided by user, or auto detected.
            backend_str = tokenizer.backend_tokenizer.to_str()
            if stop_token_ids is None:
                if hasattr(
                        tokenizer,
                        "eos_token_id") and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]
                else:
                    logger.warning(
                        "When constructing TokenizerInfo from a huggingface tokenizer, "
                        "stop_token_ids is neither provided by user nor found from the tokenizer. "
                        "It will be automatically detected.")

        if guided_params.json:
            if not isinstance(guided_params.json, str):
                json_str = json.dumps(guided_params.json)
            else:
                json_str = guided_params.json
            return cls(json_str=json_str,
                       vocab_size=model_config.hf_config.vocab_size,
                       max_threads=max_threads,
                       encoded_vocab=encoded_vocab,
                       stop_token_ids=stop_token_ids,
                       backend_str=backend_str)
        elif guided_params.grammar:
            return cls(grammar_str=guided_params.grammar,
                       vocab_size=model_config.hf_config.vocab_size,
                       max_threads=max_threads,
                       encoded_vocab=encoded_vocab,
                       stop_token_ids=stop_token_ids,
                       backend_str=backend_str)
        else:
            raise ValueError(
                "Currently only support JSON and EBNF grammar mode for xgrammar"
            )

    def create_tokenizer_info(self):
        return xgr.TokenizerInfo._create_from_handle(
            xgr_core.TokenizerInfo.from_huggingface(self.encoded_vocab,
                                                    self.backend_str,
                                                    self.vocab_size,
                                                    self.stop_token_ids))


@dataclass
class XGrammarLogitsProcessor:
    """Wrapper class to support pickle protocol"""
    config: GrammarConfig

    ctx: Optional[xgr.CompiledGrammar] = None
    matchers: List[xgr.GrammarMatcher] = field(default_factory=list)
    batch_size: int = 1
    token_bitmask: Optional[torch.Tensor]  = None
    prefilled: boolean = False

    def __getstate__(self) -> Dict[str, Any]:
        return {'config': self.config}

    def __setstate__(self, state: Dict[str, Any]):
        self.config = state['config']

        self.ctx = None
        self.matchers = []
        self.batch_size = 1
        self.token_bitmask = None
        self.prefilled = False

    def _ensure_ctx(self):
        """Lazily initialize the processor in the worker process"""
        if self.ctx is None:
            compiler = xgr.GrammarCompiler(self.config.create_tokenizer_info(),
                                           max_threads=self.config.max_threads)

            if self.config.json_str is not None:
                self.ctx = compiler.compile_json_schema(self.config.json_str)
            else:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        if self.ctx is None: self._ensure_ctx()
        if len(self.matchers) == 0:
            self.matchers = [
                xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)
            ]
            self.token_bitmask = xgr.allocate_token_bitmask(
                self.batch_size, self.config.vocab_size)

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            for i, matcher in enumerate(self.matchers):
                if not matcher.is_terminated():
                    sampled_token = input_ids[-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i, matcher in enumerate(self.matchers):
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.token_bitmask, i)

        device_type = scores.device.type
        if device_type != "cuda": scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores,
                                        self.token_bitmask.to(scores.device))
        if device_type != "cuda": scores = scores.to(device_type)

        return scores
