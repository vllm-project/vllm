# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry, get_tokenizer


def test_rwkv_tokenizer_matches_world_vocab_golden_ids():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.encode("Hello world") == [33155, 40213]
    assert tokenizer.encode("你好") == [10464, 11685]
    assert tokenizer.encode(" 42") == [3515]
    assert tokenizer.decode([33155, 40213]) == "Hello world"
    assert tokenizer.decode([10464, 11685]) == "你好"


def test_rwkv_tokenizer_decode_replaces_invalid_utf8_tokens():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.decode([129]) == "\ufffd"
    assert tokenizer.decode([196]) == "\ufffd"
    assert tokenizer.decode([256]) == "\ufffd"
    assert tokenizer.decode([129, 196, 256]) == "\ufffd\ufffd\ufffd"


def test_rwkv_tokenizer_registry_entry():
    tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("rwkv")

    assert tokenizer_cls.__name__ == "RWKVTokenizer"


def test_rwkv_tokenizer_exposes_cached_metadata():
    tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("rwkv")
    tokenizer = tokenizer_cls.from_pretrained("BlinkDL/rwkv7-g1")

    assert tokenizer.name_or_path == "BlinkDL/rwkv7-g1"
    cached_max_chars = tokenizer.max_chars_per_token
    tokenizer.idx2token.append(b"x" * (cached_max_chars + 1))
    assert tokenizer.max_chars_per_token == cached_max_chars


def test_rwkv_renderer_registry_entry():
    renderer_cls = RENDERER_REGISTRY.load_renderer_cls("rwkv")

    assert renderer_cls.__name__ == "HfRenderer"
