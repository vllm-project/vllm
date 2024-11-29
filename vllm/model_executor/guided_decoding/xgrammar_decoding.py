from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
    full_vocab_size = model_config.hf_config.vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=full_vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=max_threads)

    if guided_params.json:
        if not isinstance(guided_params.json, str):
            json_str = json.dumps(guided_params.json)
        else:
            json_str = guided_params.json
        ctx = compiler.compile_json_schema(json_str)
    elif guided_params.grammar:
        ctx = compiler.compile_grammar(guided_params.grammar)
    else:
        raise ValueError(
            "Currently only support JSON and EBNF grammar mode for xgrammar")

    return xgr.contrib.hf.LogitsProcessor(ctx)
