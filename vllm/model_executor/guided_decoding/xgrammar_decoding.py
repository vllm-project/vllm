from __future__ import annotations

import json

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, Optional

try:
    import xgrammar as xgr
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
    config = GrammarConfig.from_guided_params(
        guided_params, model_config, max_threads)
    return XGrammarLogitsProcessor(config, tokenizer)

@dataclass
class GrammarConfig:
    """Serializable configuration for grammar compilation"""
    json_str: Optional[str] = None
    grammar_str: Optional[str] = None
    vocab_size: int = 0
    max_threads: int = 8

    @classmethod
    def from_guided_params(cls,
                          guided_params: GuidedDecodingParams,
                          model_config: ModelConfig,
                          max_threads: int = 8) -> GrammarConfig:
        if guided_params.json:
            if not isinstance(guided_params.json, str):
                json_str = json.dumps(guided_params.json)
            else:
                json_str = guided_params.json
            return cls(
                json_str=json_str,
                vocab_size=model_config.hf_config.vocab_size,
                max_threads=max_threads
            )
        elif guided_params.grammar:
            return cls(
                grammar_str=guided_params.grammar,
                vocab_size=model_config.hf_config.vocab_size,
                max_threads=max_threads
            )
        else:
            raise ValueError("Currently only support JSON and EBNF grammar mode for xgrammar")

class XGrammarLogitsProcessor:
    """Wrapper class that rebuilds CompiledGrammar in each worker process"""
    def __init__(self, config: GrammarConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._processor = None

    def __getstate__(self) -> Dict[str, Any]: return {'config': self.config}

    def __setstate__(self, state: Dict[str, Any]):
        self.config = state['config']
        self._processor = None

    def _ensure_processor(self):
        """Lazily initialize the processor in the worker process"""
        if self._processor is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=self.config.vocab_size)
            compiler = xgr.GrammarCompiler(
                tokenizer_info, max_threads=self.config.max_threads)

            if self.config.json_str is not None:
                ctx = compiler.compile_json_schema(self.config.json_str)
            else:
                ctx = compiler.compile_grammar(self.config.grammar_str)

            self._processor = xgr.contrib.hf.LogitsProcessor(ctx)

    def __call__(self, *args, **kwargs):
        """Delegate to underlying processor after ensuring it's initialized"""
        if self._processor is None: self._ensure_processor()
        return self._processor(*args, **kwargs)
