# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import BatchEncoding

from vllm.exceptions import VLLMValidationError
from vllm.tokenizers.encoding_k3 import EncodeSegment
from vllm.tokenizers.kimi_k3 import get_kimi_k3_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

pytestmark = pytest.mark.skip_global_cleanup


class FakeKimiTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.segments: list[tuple[str, bool]] = []
        self.format_calls = 0

    def _encode_text_piece(
        self,
        text: str,
        allow_special_tokens: bool = True,
    ) -> list[int]:
        self.segments.append((text, allow_special_tokens))
        return [len(self.segments)]

    def _encode_chat_segments(self, segments: list[EncodeSegment]) -> list[int]:
        return [
            token_id
            for segment in segments
            for token_id in self._encode_text_piece(
                segment.text,
                allow_special_tokens=segment.allow_special,
            )
        ]

    def _format_chat_token_output(
        self,
        encoded_inputs: list[list[int]],
        *,
        is_batched: bool,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_dict: bool = False,
    ) -> Any:
        self.format_calls += 1
        if truncation and max_length is not None:
            encoded_inputs = [ids[:max_length] for ids in encoded_inputs]
        if not (is_batched or padding or return_tensors is not None or return_dict):
            return encoded_inputs[0]
        features = [
            {"input_ids": ids, "attention_mask": [1] * len(ids)}
            for ids in encoded_inputs
        ]
        batch = self.pad(
            features,
            padding=padding,
            max_length=max_length if padding else None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        if return_dict:
            return batch
        if is_batched:
            return batch["input_ids"]
        if return_tensors is None:
            return batch["input_ids"][0]
        return batch["input_ids"]

    def pad(
        self,
        features: list[dict[str, list[int]]],
        *,
        padding: bool | str = False,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        **_: Any,
    ) -> BatchEncoding:
        assert return_attention_mask
        if return_tensors is not None:
            raise NotImplementedError
        target = max(len(feature["input_ids"]) for feature in features)
        if padding == "max_length" and max_length is not None:
            target = max_length
        if not padding:
            return BatchEncoding(
                {
                    "input_ids": [feature["input_ids"] for feature in features],
                    "attention_mask": [
                        feature["attention_mask"] for feature in features
                    ],
                }
            )
        return BatchEncoding(
            {
                "input_ids": [
                    feature["input_ids"]
                    + [self.pad_token_id] * (target - len(feature["input_ids"]))
                    for feature in features
                ],
                "attention_mask": [
                    feature["attention_mask"]
                    + [0] * (target - len(feature["attention_mask"]))
                    for feature in features
                ],
            }
        )


def _tokenizer():
    return get_kimi_k3_tokenizer(FakeKimiTokenizer())


def test_kimi_k3_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "KimiK3Tokenizer"


def test_apply_chat_template_renders_string():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        thinking=False,
    )

    assert prompt == (
        '<|open|>message role="user"<|sep|>hello'
        "<|close|>message<|sep|><|end_of_msg|>"
        '<|open|>message role="assistant"<|sep|>'
        "<|open|>response<|sep|>"
    )


def test_reasoning_effort_none_disables_thinking():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        reasoning_effort="none",
    )

    assert "<|open|>think<|sep|>" not in prompt
    assert prompt.endswith("<|open|>response<|sep|>")


@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max"])
def test_reasoning_effort_uses_supported_thinking_effort(reasoning_effort: str):
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        reasoning_effort=reasoning_effort,
    )

    assert f"thinking_effort={reasoning_effort}" in prompt


@pytest.mark.parametrize("thinking_effort", ["none", "minimal", "medium", "xhigh"])
def test_rejects_unsupported_thinking_effort(thinking_effort: str):
    tokenizer = _tokenizer()

    with pytest.raises(VLLMValidationError) as exc_info:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            thinking_effort=thinking_effort,
        )

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == thinking_effort


def test_native_thinking_effort_takes_precedence():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        thinking_effort="low",
        reasoning_effort="medium",
    )

    assert "thinking_effort=low" in prompt


def test_apply_chat_template_encodes_control_and_untrusted_segments():
    tokenizer = _tokenizer()

    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|open|>user text"}],
        tokenize=True,
    )

    assert token_ids
    assert tokenizer.format_calls == 1
    assert ("<|open|>", True) in tokenizer.segments
    assert ("<|open|>user text", False) in tokenizer.segments


def test_apply_chat_template_supports_batched_conversations():
    tokenizer = _tokenizer()

    prompts = tokenizer.apply_chat_template(
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        tokenize=False,
        thinking=False,
    )

    assert isinstance(prompts, list)
    assert len(prompts) == 2
    assert "first" in prompts[0]
    assert "second" in prompts[1]


def test_apply_chat_template_formats_batched_token_ids():
    tokenizer = _tokenizer()

    batch = tokenizer.apply_chat_template(
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        tokenize=True,
        padding=True,
    )

    assert isinstance(batch, list)
    assert len(batch) == 2
    assert len(batch[0]) == len(batch[1])


def test_apply_chat_template_rejects_batched_image_prompts():
    tokenizer = _tokenizer()

    with pytest.raises(ValueError, match="only supported for one chat"):
        tokenizer.apply_chat_template(
            [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            image_prompts=["image"],
        )


def test_apply_chat_template_returns_batch_encoding():
    tokenizer = _tokenizer()

    batch = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        return_dict=True,
    )

    assert isinstance(batch, BatchEncoding)
    assert len(batch["input_ids"]) == 1
