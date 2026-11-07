# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib.util
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.renderers import TokenizeParams
from vllm.renderers.cohere import CohereRenderer
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer, DeepseekV41Renderer
from vllm.renderers.deepseek_v32 import DeepseekV32Renderer
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inkling import InklingRenderer
from vllm.renderers.inkling_encoding import SPECIAL_TOKEN_SPELLINGS
from vllm.renderers.kimi_k3 import KimiK3Renderer
from vllm.tokenizers.hf import ThreadSafeHFTokenizerMixin, get_cached_tokenizer

RENDERERS = [
    HfRenderer,
    pytest.param(
        CohereRenderer,
        marks=pytest.mark.skipif(
            importlib.util.find_spec("cohere_melody") is None,
            reason="cohere_melody is not installed",
        ),
    ),
    DeepseekV32Renderer,
    DeepseekV4Renderer,
    DeepseekV41Renderer,
    InklingRenderer,
    KimiK3Renderer,
]
PROMPT = " ".join(f"w{i}" for i in range(300))
SHORT, LONG = 128, 256


@dataclass
class MockHFConfig:
    model_type: str = "llama"


@dataclass
class MockModelConfig:
    runner_type: str = "generate"
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    multimodal_config: Any = None
    encoder_config: Any = None
    allowed_local_media_path: str = ""
    allowed_media_domains: Any = None
    enable_prompt_embeds: bool = False
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 2
    max_model_len: int = 4096


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig = field(default_factory=MockParallelConfig)


def _tokenizer():
    special = [spellings[0] for spellings in SPECIAL_TOKEN_SPELLINGS.values()]
    words = ["[UNK]", *(f"w{i}" for i in range(512)), *special]
    backend = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return get_cached_tokenizer(
        TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]")
    )


def _params(max_length: int) -> TokenizeParams:
    return TokenizeParams(max_total_tokens=1024, truncate_prompt_tokens=max_length)


@pytest.fixture
def interleaved_truncation(monkeypatch):
    """Order two concurrent encodes: the LONG request sets its truncation, the
    SHORT request sets its own, and only then does the LONG request encode.

    A fast tokenizer stores the truncation on its Rust backend right before
    encoding, so a backend shared by both requests truncates the LONG one to
    SHORT; per-request backends are unaffected.
    """
    long_set, short_set = threading.Event(), threading.Event()
    original = TokenizersBackend.set_truncation_and_padding

    def ordered(self, *args, **kwargs):
        max_length = kwargs.get("max_length", args[2] if len(args) > 2 else None)
        if max_length == SHORT:
            long_set.wait(timeout=10)
        result = original(self, *args, **kwargs)
        if max_length == LONG:
            long_set.set()
            short_set.wait(timeout=10)
        elif max_length == SHORT:
            short_set.set()
        return result

    monkeypatch.setattr(TokenizersBackend, "set_truncation_and_padding", ordered)
    return long_set, short_set


@pytest.mark.parametrize("renderer_cls", RENDERERS)
def test_concurrent_truncation_is_per_request(renderer_cls, interleaved_truncation):
    renderer = renderer_cls(MockVllmConfig(MockModelConfig()), _tokenizer())

    async def run():
        return await asyncio.gather(
            renderer.tokenize_prompt_async({"prompt": PROMPT}, _params(LONG)),
            renderer.tokenize_prompt_async({"prompt": PROMPT}, _params(SHORT)),
        )

    long_out, short_out = asyncio.run(run())

    assert all(event.is_set() for event in interleaved_truncation)
    assert len(short_out["prompt_token_ids"]) == SHORT
    assert len(long_out["prompt_token_ids"]) == LONG


@pytest.mark.parametrize("renderer_cls", RENDERERS)
def test_pooled_tokenizer_leaves_cached_tokenizer_unchanged(renderer_cls):
    tokenizer = _tokenizer()
    tokenizer_cls = type(tokenizer)
    renderer = renderer_cls(MockVllmConfig(MockModelConfig()), tokenizer)

    assert type(tokenizer) is tokenizer_cls
    assert not isinstance(tokenizer, ThreadSafeHFTokenizerMixin)
    pooled = renderer.get_tokenizer()
    assert pooled.encode(PROMPT) == tokenizer.encode(PROMPT)
    assert pooled.decode(tokenizer.encode(PROMPT)) == tokenizer.decode(
        tokenizer.encode(PROMPT)
    )
