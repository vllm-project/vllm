# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering vector plumbing through OpenAI-compatible protocol models.

Verifies that ChatCompletionRequest and CompletionRequest accept all three
steering fields (steering_vectors, prefill_steering_vectors,
decode_steering_vectors) with both bare list[float] and scaled
{"vector": [...], "scale": float} formats, and that to_sampling_params()
passes them through correctly.
"""

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BARE_VECTORS = {
    "pre_attn": {15: [0.1, 0.2, 0.3]},
}

_SCALED_VECTORS = {
    "post_mlp_pre_ln": {
        10: {"vector": [0.4, 0.5, 0.6], "scale": 2.0},
    },
}

_PREFILL_VECTORS = {
    "pre_attn": {15: [0.7, 0.8, 0.9]},
}

_DECODE_VECTORS = {
    "post_attn": {20: [1.0, 1.1, 1.2]},
}

_CHAT_BASE = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "test-model",
}

_COMPLETION_BASE = {
    "prompt": "Hello",
    "model": "test-model",
}


def _make_chat(**extra):
    return ChatCompletionRequest.model_validate({**_CHAT_BASE, **extra})


def _make_completion(**extra):
    return CompletionRequest.model_validate({**_COMPLETION_BASE, **extra})


# ---------------------------------------------------------------------------
# ChatCompletionRequest tests
# ---------------------------------------------------------------------------


class TestChatCompletionSteering:
    """ChatCompletionRequest steering field acceptance."""

    def test_no_steering_fields(self):
        req = _make_chat()
        assert req.steering_vectors is None
        assert req.prefill_steering_vectors is None
        assert req.decode_steering_vectors is None

    def test_bare_steering_vectors(self):
        req = _make_chat(steering_vectors=_BARE_VECTORS)
        assert req.steering_vectors == _BARE_VECTORS

    def test_scaled_steering_vectors(self):
        req = _make_chat(steering_vectors=_SCALED_VECTORS)
        assert req.steering_vectors == _SCALED_VECTORS

    def test_prefill_steering_vectors(self):
        req = _make_chat(prefill_steering_vectors=_PREFILL_VECTORS)
        assert req.prefill_steering_vectors == _PREFILL_VECTORS

    def test_decode_steering_vectors(self):
        req = _make_chat(decode_steering_vectors=_DECODE_VECTORS)
        assert req.decode_steering_vectors == _DECODE_VECTORS

    def test_all_three_tiers(self):
        req = _make_chat(
            steering_vectors=_BARE_VECTORS,
            prefill_steering_vectors=_PREFILL_VECTORS,
            decode_steering_vectors=_DECODE_VECTORS,
        )
        assert req.steering_vectors == _BARE_VECTORS
        assert req.prefill_steering_vectors == _PREFILL_VECTORS
        assert req.decode_steering_vectors == _DECODE_VECTORS

    def test_to_sampling_params_passes_all_fields(self):
        req = _make_chat(
            steering_vectors=_BARE_VECTORS,
            prefill_steering_vectors=_PREFILL_VECTORS,
            decode_steering_vectors=_DECODE_VECTORS,
        )
        sp = req.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )
        assert sp.steering_vectors == _BARE_VECTORS
        assert sp.prefill_steering_vectors == _PREFILL_VECTORS
        assert sp.decode_steering_vectors == _DECODE_VECTORS

    def test_to_sampling_params_none_when_absent(self):
        req = _make_chat()
        sp = req.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None

    def test_scaled_format_passes_through(self):
        req = _make_chat(steering_vectors=_SCALED_VECTORS)
        sp = req.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )
        entry = sp.steering_vectors["post_mlp_pre_ln"][10]
        assert isinstance(entry, dict)
        assert entry["vector"] == [0.4, 0.5, 0.6]
        assert entry["scale"] == 2.0


# ---------------------------------------------------------------------------
# CompletionRequest tests
# ---------------------------------------------------------------------------


class TestCompletionSteering:
    """CompletionRequest steering field acceptance."""

    def test_no_steering_fields(self):
        req = _make_completion()
        assert req.steering_vectors is None
        assert req.prefill_steering_vectors is None
        assert req.decode_steering_vectors is None

    def test_bare_steering_vectors(self):
        req = _make_completion(steering_vectors=_BARE_VECTORS)
        assert req.steering_vectors == _BARE_VECTORS

    def test_scaled_steering_vectors(self):
        req = _make_completion(steering_vectors=_SCALED_VECTORS)
        assert req.steering_vectors == _SCALED_VECTORS

    def test_prefill_steering_vectors(self):
        req = _make_completion(prefill_steering_vectors=_PREFILL_VECTORS)
        assert req.prefill_steering_vectors == _PREFILL_VECTORS

    def test_decode_steering_vectors(self):
        req = _make_completion(decode_steering_vectors=_DECODE_VECTORS)
        assert req.decode_steering_vectors == _DECODE_VECTORS

    def test_all_three_tiers(self):
        req = _make_completion(
            steering_vectors=_BARE_VECTORS,
            prefill_steering_vectors=_PREFILL_VECTORS,
            decode_steering_vectors=_DECODE_VECTORS,
        )
        assert req.steering_vectors == _BARE_VECTORS
        assert req.prefill_steering_vectors == _PREFILL_VECTORS
        assert req.decode_steering_vectors == _DECODE_VECTORS

    def test_to_sampling_params_passes_all_fields(self):
        req = _make_completion(
            steering_vectors=_BARE_VECTORS,
            prefill_steering_vectors=_PREFILL_VECTORS,
            decode_steering_vectors=_DECODE_VECTORS,
        )
        sp = req.to_sampling_params(max_tokens=100)
        assert sp.steering_vectors == _BARE_VECTORS
        assert sp.prefill_steering_vectors == _PREFILL_VECTORS
        assert sp.decode_steering_vectors == _DECODE_VECTORS

    def test_to_sampling_params_none_when_absent(self):
        req = _make_completion()
        sp = req.to_sampling_params(max_tokens=100)
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None

    def test_scaled_format_passes_through(self):
        req = _make_completion(steering_vectors=_SCALED_VECTORS)
        sp = req.to_sampling_params(max_tokens=100)
        entry = sp.steering_vectors["post_mlp_pre_ln"][10]
        assert isinstance(entry, dict)
        assert entry["vector"] == [0.4, 0.5, 0.6]
        assert entry["scale"] == 2.0
