# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoTokenizer

import vllm.utils.gpu_sync_debug as gsd
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.v1.spec_decode.vocab_mapping import VocabMapping, _detect_space_prefix


@pytest.mark.parametrize(
    "model_name,expected_prefix",
    [
        # BPE tokenizer (GPT-2 family) uses Ġ (U+0120)
        ("HuggingFaceTB/SmolLM2-135M-Instruct", ("Ġ",)),
        # SentencePiece tokenizer (LLaMA family) uses ▁ (U+2581)
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ("▁",)),
        # BPE tokenizer (Qwen family) uses Ġ (U+0120)
        ("Qwen/Qwen2.5-0.5B-Instruct", ("Ġ",)),
    ],
)
def test_detect_space_prefix_real_tokenizers(model_name, expected_prefix):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    result = _detect_space_prefix(tokenizer)
    assert result == expected_prefix, (
        f"{model_name}: expected {expected_prefix!r}, got {result!r}"
    )


def test_detect_space_prefix_fallback_on_failure():
    """When tokenizer lacks encode(), fall back to both known prefixes."""

    class BrokenTokenizer:
        def encode(self, text, **kwargs):
            raise RuntimeError("broken")

    result = _detect_space_prefix(BrokenTokenizer())
    assert result == ("Ġ", "▁")


def test_detect_space_prefix_empty_encode():
    """When encode returns empty list, fall back."""

    class EmptyTokenizer:
        def encode(self, text, **kwargs):
            return []

    result = _detect_space_prefix(EmptyTokenizer())
    assert result == ("Ġ", "▁")


class _FakeTokenizer:
    """Minimal tokenizer exposing only what ``VocabMapping`` needs."""

    def __init__(self, vocab, unk_token_id=None, eos_token_id=None):
        self._vocab = dict(vocab)
        self.unk_token_id = unk_token_id
        self.eos_token_id = eos_token_id

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, **kwargs):
        raise RuntimeError("no encode")

    def convert_ids_to_tokens(self, ids):
        raise RuntimeError("no convert")


@pytest.fixture(params=["cpu", "cuda"])
def vocab_mapping(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    # Intersection: {a, b}. x/y/z are target-only, u/v are draft-only
    target = _FakeTokenizer(
        vocab={"a": 0, "b": 1, "x": 2, "y": 3, "z": 4}, unk_token_id=0
    )
    draft = _FakeTokenizer(vocab={"a": 0, "b": 1, "u": 2, "v": 3}, unk_token_id=2)
    return VocabMapping(target, draft, 5, 4, device)


@pytest.mark.parametrize(
    "target_ids,expected",
    [
        ([0, 1, 2, 3, 4], [0, 1, 2, 2, 2]),  # mixed
        ([0, 1], [0, 1]),  # all present
        ([2, 3, 4], [2, 2, 2]),  # all missing
    ],
)
def test_map_target_to_draft_ids(monkeypatch, vocab_mapping, target_ids, expected):
    device = vocab_mapping.device
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error" if device == "cuda" else None)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    ids = torch.tensor(target_ids, dtype=torch.int32, device=device)
    out = with_gpu_sync_check(lambda: vocab_mapping.map_target_to_draft_ids(ids))()

    assert out.tolist() == expected
    assert out.dtype == torch.int32, f"expected torch.int32, got {out.dtype}"
    assert out.device.type == device, f"expected {device} device, got {out.device}"
    assert (out != -1).all()


@pytest.mark.parametrize(
    "draft_ids,expected",
    [
        ([0, 1, 2, 3], [0, 1, 0, 0]),  # mixed
        ([0, 1], [0, 1]),  # all present
        ([2, 3], [0, 0]),  # all missing
    ],
)
def test_map_draft_to_target_ids(monkeypatch, vocab_mapping, draft_ids, expected):
    device = vocab_mapping.device
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error" if device == "cuda" else None)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    ids = torch.tensor(draft_ids, dtype=torch.int32, device=device)
    out = with_gpu_sync_check(lambda: vocab_mapping.map_draft_to_target_ids(ids))()

    assert out.tolist() == expected
    assert out.dtype == torch.int32, f"expected torch.int32, got {out.dtype}"
    assert out.device.type == device, f"expected {device} device, got {out.device}"
    assert (out != -1).all()
