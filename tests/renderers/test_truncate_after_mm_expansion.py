# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truncation must bound the prompt *after* multimodal placeholders expand.

``TokenizeParams.apply_post_tokenization`` truncates the unexpanded prompt, so
one placeholder token can expand into many and push the result back over
``truncate_prompt_tokens`` (vllm-project/vllm#57038).
"""

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import PlaceholderRange
from vllm.renderers import renderer_from_config
from vllm.renderers.base import BaseRenderer
from vllm.renderers.hf import _ensure_prompt_embeds_placeholder_token
from vllm.renderers.params import TokenizeParams


def _renderer():
    """A BaseRenderer stub: only the tokenizer attribute is needed here."""

    class _StubRenderer(BaseRenderer):
        def __init__(self):
            self.tokenizer = None

        def render_messages(self, *args, **kwargs):
            raise NotImplementedError

    return _StubRenderer()


def _mm_input(num_tokens: int, offset: int, length: int, **extra):
    return {
        "type": "multimodal",
        "prompt_token_ids": list(range(num_tokens)),
        "mm_kwargs": {},
        "mm_hashes": {},
        "mm_placeholders": {"audio": [PlaceholderRange(offset=offset, length=length)]},
        **extra,
    }


def _params(limit: int | None, side: str = "right"):
    return TokenizeParams(
        max_total_tokens=4096,
        truncate_prompt_tokens=limit,
        truncation_side=side,
    )


def test_expanded_prompt_is_bounded():
    """The reported case: audio fits, trailing text is cut to the limit."""
    engine_input = _mm_input(num_tokens=147, offset=1, length=52)

    _renderer()._truncate_expanded_prompt(engine_input, _params(96))

    assert engine_input["prompt_token_ids"] == list(range(96))
    assert engine_input["mm_placeholders"]["audio"][0] == PlaceholderRange(
        offset=1, length=52
    )


def test_prompt_within_limit_is_untouched():
    engine_input = _mm_input(num_tokens=64, offset=1, length=52)

    _renderer()._truncate_expanded_prompt(engine_input, _params(96))

    assert engine_input["prompt_token_ids"] == list(range(64))


def test_split_multimodal_span_is_rejected():
    """Cutting mid-span would desync features from token ids, so reject."""
    engine_input = _mm_input(num_tokens=147, offset=80, length=50)

    with pytest.raises(VLLMValidationError, match="truncate_prompt_tokens"):
        _renderer()._truncate_expanded_prompt(engine_input, _params(96))


def test_dropped_multimodal_span_is_rejected():
    """A span entirely past the cut would be discarded silently."""
    engine_input = _mm_input(num_tokens=147, offset=100, length=20)

    with pytest.raises(VLLMValidationError, match="truncate_prompt_tokens"):
        _renderer()._truncate_expanded_prompt(engine_input, _params(96))


def test_left_truncation_shifts_placeholder_offsets():
    engine_input = _mm_input(num_tokens=147, offset=100, length=20)

    _renderer()._truncate_expanded_prompt(engine_input, _params(96, side="left"))

    assert engine_input["prompt_token_ids"] == list(range(51, 147))
    # The span started at 100 and 51 leading tokens were dropped.
    assert engine_input["mm_placeholders"]["audio"][0] == PlaceholderRange(
        offset=49, length=20
    )


def test_text_only_input_is_untouched():
    """Only multimodal prompts grow after tokenization, so nothing else runs."""
    engine_input = {
        "type": "token",
        "prompt_token_ids": list(range(147)),
        "prompt_token_offsets": [(i, i + 1) for i in range(147)],
    }

    _renderer()._truncate_expanded_prompt(engine_input, _params(96))

    assert engine_input["prompt_token_ids"] == list(range(147))
    assert len(engine_input["prompt_token_offsets"]) == 147


def test_no_truncation_requested_is_noop():
    engine_input = _mm_input(num_tokens=147, offset=1, length=52)

    _renderer()._truncate_expanded_prompt(engine_input, _params(None))

    assert engine_input["prompt_token_ids"] == list(range(147))


def test_mixed_prompt_embeds_truncation_covers_embeds_spans():
    """Bound in _process_singleton, where prompt_embeds spans are already attached."""
    model_config = ModelConfig(
        "Qwen/Qwen3-ASR-0.6B-hf",
        revision="7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        dtype="float32",
        max_model_len=4096,
        mm_processor_cache_gb=0,
        enable_prompt_embeds=True,
    )
    renderer = renderer_from_config(VllmConfig(model_config=model_config))
    tok = renderer.tokenizer
    pe_id = _ensure_prompt_embeds_placeholder_token(tok)
    hidden = model_config.get_hidden_size()
    audio = np.zeros(4 * 16000, dtype=np.float32)

    pe_len = 20
    tensors = [torch.zeros(pe_len, hidden, dtype=torch.float32)]

    def _build_prompt():
        text = "<|audio_start|><|audio_pad|><|audio_end|>"
        ids = tok.encode(text) + [pe_id] + tok.encode(" tail" * 40)
        return {
            "prompt_token_ids": ids,
            "multi_modal_data": {"audio": (audio, 16000)},
            "_prompt_embeds": (tensors, pe_id),
        }

    # A limit past every span (audio ends at 52, prompt_embeds at 74):
    # truncation should succeed and bound the prompt.
    out = renderer._process_singleton(
        _build_prompt(),
        tok_params=_params(79),
    )
    assert len(out["prompt_token_ids"]) == 79

    # A limit inside the prompt_embeds span (54..74): must be rejected.
    with pytest.raises(VLLMValidationError, match="truncate_prompt_tokens"):
        renderer._process_singleton(
            _build_prompt(),
            tok_params=_params(64),
        )
