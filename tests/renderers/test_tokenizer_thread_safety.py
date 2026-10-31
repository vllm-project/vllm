# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.hf import ThreadSafeHFTokenizerMixin


@dataclass
class _MockModelConfig:
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 4
    hf_config: object | None = None


@dataclass
class _MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _MockVllmConfig:
    model_config: _MockModelConfig
    parallel_config: _MockParallelConfig


def _make_fast_tokenizer() -> PreTrainedTokenizerFast:
    # Build a tokenizer locally so the test does not depend on the Hub.
    vocab = {"hello": 0, "world": 1, "[UNK]": 2}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")


@pytest.mark.parametrize("renderer_cls", [HfRenderer, DeepseekV4Renderer])
def test_renderer_pools_hf_fast_tokenizer(renderer_cls):
    """Every renderer must make its fast tokenizer safe for the shared
    renderer worker pool."""
    tokenizer = _make_fast_tokenizer()
    renderer = renderer_cls(
        _MockVllmConfig(_MockModelConfig(), _MockParallelConfig()),
        tokenizer,
    )

    assert isinstance(renderer.tokenizer, ThreadSafeHFTokenizerMixin)
    assert renderer.tokenizer is not tokenizer
    assert renderer.tokenizer.encode("hello world") == tokenizer.encode("hello world")
