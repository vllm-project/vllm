# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import jinja2
import numpy as np
import pytest
import safetensors.torch
import torch
from transformers.utils import chat_template_utils as hf_chat_utils

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.model_executor.models import kimi_audio as kimi_audio_model
from vllm.model_executor.models.kimi_audio import KimiAudioForConditionalGeneration
from vllm.model_executor.models.kimi_audio_prompt import (
    KimiAudioPromptBuilder,
    KimiAudioTokenContent,
)
from vllm.tokenizers.kimi_audio import KimiAudioTokenizer
from vllm.transformers_utils.processors import kimi_audio_speech as speech_utils
from vllm.transformers_utils.processors.kimi_audio import KimiAudioProcessor
from vllm.transformers_utils.processors.kimi_audio_whisper_vq import (
    WhisperVQConfig,
    WhisperVQEncoder,
)

KIMI_AUDIO_MODEL = "moonshotai/Kimi-Audio-7B-Instruct"


class _FakePromptTokenizer:
    special_tokens = {
        "<|im_msg_end|>": 1,
        "<|im_media_begin|>": 2,
        "<|im_media_end|>": 3,
        "<|im_kimia_text_blank|>": 4,
        "<|im_kimia_text_eos|>": 5,
        "<|im_kimia_user_msg_start|>": 6,
        "<|im_kimia_assistant_msg_start|>": 7,
        "<|im_kimia_speech_ct_id|>": 8,
        "<|im_kimia_speech_ctd_id|>": 9,
    }

    vocab = {
        "hi": [10],
        "hello back": [20, 21],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.vocab.get(text, []))

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_tokens[token]


def test_kimi_audio_tokenizer_encodes_alignment_special_tokens():
    model_info = HF_EXAMPLE_MODELS.get_hf_info("MoonshotKimiaForCausalLM")
    model_info.check_available_online(on_fail="skip")

    tokenizer = KimiAudioTokenizer.from_pretrained(KIMI_AUDIO_MODEL)

    token_ids = tokenizer.encode(
        "<|im_media_begin|><|im_kimia_text_blank|>"
        "<|im_media_end|><|im_kimia_speech_ct_id|><|im_kimia_text_eos|>",
        add_special_tokens=False,
    )

    assert token_ids == [151661, 151666, 151663, 151675, 151667]
    assert tokenizer.convert_tokens_to_ids("<|im_kimia_speech_ctd_id|>") == 151676


def test_kimi_audio_prompt_builder_text_output_audio_message():
    multi_audio_placeholder = KimiAudioPromptBuilder.build_audio_placeholder(
        audio_count=2,
    )
    assert multi_audio_placeholder.count("<|im_kimia_speech_ct_id|>") == 2

    prompt = KimiAudioPromptBuilder.build_transcription_prompt(
        "Please transcribe the following audio:",
        audio_count=1,
    )

    assert prompt == (
        "<|im_kimia_user_msg_start|>Please transcribe the following audio:\n"
        "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
        "<|im_kimia_speech_ct_id|><|im_msg_end|>"
        "<|im_kimia_assistant_msg_start|>"
    )


def test_kimi_audio_prompt_builder_supports_text_history():
    prompt = KimiAudioPromptBuilder.build_prompt_from_messages(
        [
            {"role": "user", "message_type": "text", "content": "hi"},
            {
                "role": "assistant",
                "message_type": "text",
                "content": "hello back",
            },
            {"role": "user", "message_type": "audio", "content": "transcribe"},
        ]
    )

    assert prompt == (
        "<|im_kimia_user_msg_start|>hi<|im_msg_end|>"
        "<|im_kimia_assistant_msg_start|>hello back<|im_kimia_text_eos|><|im_msg_end|>"
        "<|im_kimia_user_msg_start|>transcribe\n"
        "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
        "<|im_kimia_speech_ct_id|><|im_msg_end|>"
        "<|im_kimia_assistant_msg_start|>"
    )


@pytest.mark.parametrize(
    "input_style",
    ["messages", "conversation", "conversations"],
)
def test_kimi_audio_chat_template_renders_service_messages(input_style):
    template_path = (
        Path(__file__).resolve().parents[4]
        / "vllm"
        / "transformers_utils"
        / "chat_templates"
        / "template_kimi_audio.jinja"
    )
    template = template_path.read_text()
    content = KimiAudioPromptBuilder.build_user_audio_content(
        "Please answer the question in the audio.",
    )
    assistant = "你能不能从一数到十？"
    conversation = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": assistant},
    ]

    if input_style == "conversations":
        rendered, _ = hf_chat_utils.render_jinja_template(
            [conversation],
            chat_template=template,
        )
        prompt = rendered[0]
    else:
        prompt = jinja2.Environment().from_string(template).render(
            **{input_style: conversation}
        )

    assert prompt == (
        "<|im_kimia_user_msg_start|>Please answer the question in the audio.\n"
        "<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>"
        "<|im_kimia_speech_ct_id|><|im_msg_end|>"
        "<|im_kimia_assistant_msg_start|>你能不能从一数到十？"
        "<|im_kimia_text_eos|><|im_msg_end|>"
    )


def test_kimi_audio_tokenizer_wraps_single_conversation_for_template_render(
    monkeypatch,
):
    captured = {}
    tokenizer = object.__new__(KimiAudioTokenizer)
    tokenizer.get_chat_template = lambda chat_template, tools=None: chat_template
    tokenizer.encode = lambda text, add_special_tokens=False: [1, 2, 3]

    def fake_render_jinja_template(conversations, **kwargs):
        captured["conversations"] = conversations
        captured["kwargs"] = kwargs
        return ["rendered prompt"], []

    monkeypatch.setattr(
        "vllm.tokenizers.kimi_audio.hf_chat_utils.render_jinja_template",
        fake_render_jinja_template,
    )

    prompt = KimiAudioTokenizer.apply_chat_template(
        tokenizer,
        conversation=[{"role": "user", "content": "hello"}],
        chat_template="unused template",
        tokenize=False,
    )

    assert prompt == "rendered prompt"
    assert captured["conversations"] == [[{"role": "user", "content": "hello"}]]
    assert "conversation" not in captured["kwargs"]


def test_kimi_audio_prompt_builder_builds_token_level_audio_text_streams():
    tokenizer = _FakePromptTokenizer()

    packed = KimiAudioPromptBuilder.build_token_content(
        tokenizer=tokenizer,
        messages=[
            {"role": "user", "message_type": "text", "content": "hi"},
            {
                "role": "assistant",
                "message_type": "text",
                "content": "hello back",
            },
            {"role": "user", "message_type": "audio", "content": [152064, 152065]},
        ],
    )

    assert isinstance(packed, KimiAudioTokenContent)
    assert packed.audio_token_ids == [
        6,
        4,
        1,
        7,
        4,
        4,
        4,
        1,
        6,
        2,
        152064,
        152065,
        3,
        8,
        1,
        7,
    ]
    assert packed.text_token_ids == [
        4,
        10,
        4,
        4,
        20,
        21,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    assert packed.is_continuous_mask == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_kimi_audio_declares_model_side_multimodal_embed_build():
    assert KimiAudioForConditionalGeneration.supports_multimodal_raw_input_only
    assert KimiAudioForConditionalGeneration.builds_multimodal_inputs_embeds_in_forward


def test_kimi_audio_runtime_padding_preserves_prefix_embeddings():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.embed_input_ids = lambda input_ids: torch.full((6, 4), -1.0)

    kimi_inputs_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    padded, used_runtime_padding = (
        KimiAudioForConditionalGeneration._pad_runtime_kimi_inputs_embeds(
            model,
            input_ids=torch.arange(6, dtype=torch.long),
            kimi_inputs_embeds=kimi_inputs_embeds,
        )
    )

    assert used_runtime_padding
    assert padded.shape == (6, 4)
    assert torch.equal(padded[:3], kimi_inputs_embeds)
    assert torch.equal(padded[3:], torch.full((3, 4), -1.0))


def test_kimi_audio_build_inputs_embeds_supports_runtime_flat_tokens():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model._normalize_multimodal_embeddings = lambda embeddings, batch_size: []
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=lambda token_ids: token_ids.unsqueeze(-1).to(torch.float32)
        )
    )

    outputs = KimiAudioForConditionalGeneration._build_kimi_audio_inputs_embeds(
        model,
        audio_token_ids=torch.tensor([1, 2], dtype=torch.long),
        text_input_ids=torch.tensor([10, 20], dtype=torch.long),
        is_continuous_mask=torch.tensor([False, False]),
        multimodal_embeddings=None,
    )

    assert outputs.shape == (2, 1)
    assert torch.equal(outputs, torch.tensor([[11.0], [22.0]]))


def test_kimi_audio_build_inputs_embeds_supports_flat_batch_multimodal_embeddings():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    outputs = KimiAudioForConditionalGeneration._build_kimi_audio_inputs_embeds(
        model,
        audio_token_ids=torch.tensor([10, 11, 20], dtype=torch.long),
        text_input_ids=torch.tensor([1, 1, 2], dtype=torch.long),
        is_continuous_mask=torch.tensor([True, False, True]),
        multimodal_embeddings=[
            torch.tensor([[100.0, 100.0]]),
            torch.tensor([[200.0, 200.0]]),
        ],
    )

    sqrt2 = 2**0.5
    expected = torch.tensor(
        [
            [(10.0 + 100.0) * sqrt2 + 1.0, (10.0 + 100.0) * sqrt2 + 1.0],
            [12.0, 12.0],
            [(20.0 + 200.0) * sqrt2 + 2.0, (20.0 + 200.0) * sqrt2 + 2.0],
        ]
    )
    assert torch.allclose(outputs, expected)


def test_kimi_audio_runtime_slice_supports_flat_batch_request_lists():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([9, 9, 9, 9, 9], dtype=torch.long),
        positions=torch.tensor([1, 2, 3, 0, 2], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11, 12, 13], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ],
        text_input_ids=[
            torch.tensor([1, 1, 1, 1], dtype=torch.long),
            torch.tensor([2, 2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, False, True, True]),
            torch.tensor([True, False, True]),
        ],
    )

    assert torch.equal(audio_token_ids, torch.tensor([11, 12, 13, 20, 22]))
    assert torch.equal(text_input_ids, torch.tensor([1, 1, 1, 2, 2]))
    assert torch.equal(
        is_continuous_mask,
        torch.tensor([False, True, True, True, True]),
    )


def test_kimi_audio_runtime_slice_flattens_batch_request_lists_for_global_positions():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([9, 9, 9, 9, 9], dtype=torch.long),
        positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ],
        text_input_ids=[
            torch.tensor([1, 1, 1], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, True, True]),
            torch.tensor([False, True]),
        ],
    )

    assert torch.equal(audio_token_ids, torch.tensor([10, 11, 12, 20, 21]))
    assert torch.equal(text_input_ids, torch.tensor([1, 1, 1, 2, 2]))
    assert torch.equal(
        is_continuous_mask,
        torch.tensor([False, True, True, False, True]),
    )


def test_kimi_audio_runtime_slice_skips_raw_prompt_path_for_batch_decode_tokens():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([1, 2], dtype=torch.long),
        positions=torch.tensor([3, 2], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ],
        text_input_ids=[
            torch.tensor([1, 1, 1], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, True, True]),
            torch.tensor([False, True]),
        ],
    )

    assert audio_token_ids is None
    assert text_input_ids is None
    assert is_continuous_mask is None


def test_kimi_audio_runtime_slice_supports_mixed_decode_and_prefill_batch():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([99, 9, 9, 9, 9], dtype=torch.long),
        positions=torch.tensor([3, 0, 1, 0, 2], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ],
        text_input_ids=[
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([2, 2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, True]),
            torch.tensor([False, True, True]),
        ],
    )

    assert torch.equal(
        audio_token_ids,
        torch.tensor(
            [KimiAudioProcessor.KIMIA_TEXT_BLANK, 10, 11, 20, 22],
            dtype=torch.long,
        ),
    )
    assert torch.equal(text_input_ids, torch.tensor([99, 1, 1, 2, 2]))
    assert torch.equal(
        is_continuous_mask,
        torch.tensor([False, False, True, False, True]),
    )


def test_kimi_audio_runtime_slice_supports_single_prompt_request_after_decode():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([99, 9], dtype=torch.int32),
        positions=torch.tensor([4, 2], dtype=torch.long),
        audio_token_ids=torch.tensor([[20, 21, 22]], dtype=torch.long),
        text_input_ids=torch.tensor([[2, 2, 2]], dtype=torch.long),
        is_continuous_mask=torch.tensor([[False, True, True]]),
    )

    assert torch.equal(
        audio_token_ids,
        torch.tensor([KimiAudioProcessor.KIMIA_TEXT_BLANK, 22], dtype=torch.long),
    )
    assert torch.equal(text_input_ids, torch.tensor([99, 2], dtype=torch.long))
    assert torch.equal(is_continuous_mask, torch.tensor([False, True]))


def test_kimi_audio_runtime_slice_supports_single_prompt_request_before_decode():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([9, 9, 9, 99], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 4], dtype=torch.long),
        audio_token_ids=torch.tensor([[20, 21, 22]], dtype=torch.long),
        text_input_ids=torch.tensor([[2, 2, 2]], dtype=torch.long),
        is_continuous_mask=torch.tensor([[False, True, True]]),
    )

    assert torch.equal(
        audio_token_ids,
        torch.tensor(
            [20, 21, 22, KimiAudioProcessor.KIMIA_TEXT_BLANK],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        text_input_ids,
        torch.tensor([2, 2, 2, 99], dtype=torch.long),
    )
    assert torch.equal(
        is_continuous_mask,
        torch.tensor([False, True, True, False]),
    )


def test_kimi_audio_runtime_slice_supports_interleaved_decode_between_prompts():
    model = object.__new__(KimiAudioForConditionalGeneration)

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([9, 9, 99, 8, 8], dtype=torch.int32),
        positions=torch.tensor([0, 1, 4, 0, 1], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ],
        text_input_ids=[
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, True]),
            torch.tensor([False, True]),
        ],
    )

    assert torch.equal(
        audio_token_ids,
        torch.tensor(
            [10, 11, KimiAudioProcessor.KIMIA_TEXT_BLANK, 20, 21],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        text_input_ids,
        torch.tensor([1, 1, 99, 2, 2], dtype=torch.long),
    )
    assert torch.equal(
        is_continuous_mask,
        torch.tensor([False, True, False, False, True]),
    )


def test_kimi_audio_runtime_slice_uses_runtime_request_boundaries():
    model = object.__new__(KimiAudioForConditionalGeneration)
    full_audio_ids = torch.cat(
        [
            torch.full((176,), 7, dtype=torch.long),
            torch.tensor([20, 21, 22, 23], dtype=torch.long),
        ]
    )
    full_text_ids = torch.cat(
        [
            torch.full((176,), 3, dtype=torch.long),
            torch.tensor([2, 2, 2, 2], dtype=torch.long),
        ]
    )
    full_continuous_mask = torch.cat(
        [
            torch.zeros(176, dtype=torch.bool),
            torch.tensor([False, True, True, False]),
        ]
    )

    (
        audio_token_ids,
        text_input_ids,
        is_continuous_mask,
    ) = KimiAudioForConditionalGeneration._slice_runtime_kimi_prompt_inputs(
        model,
        input_ids=torch.tensor([91, 92, 93, 9, 9, 9, 9], dtype=torch.int32),
        positions=torch.tensor([176, 192, 208, 176, 177, 178, 179], dtype=torch.long),
        audio_token_ids=full_audio_ids.unsqueeze(0),
        text_input_ids=full_text_ids.unsqueeze(0),
        is_continuous_mask=full_continuous_mask.unsqueeze(0),
        runtime_request_num_scheduled_tokens=torch.tensor(
            [1, 1, 1, 4], dtype=torch.int32
        ),
        runtime_request_has_raw_mm_inputs=torch.tensor(
            [False, False, False, True], dtype=torch.bool
        ),
    )

    assert torch.equal(
        audio_token_ids,
        torch.tensor(
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK,
                KimiAudioProcessor.KIMIA_TEXT_BLANK,
                KimiAudioProcessor.KIMIA_TEXT_BLANK,
                20,
                21,
                22,
                23,
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        text_input_ids,
        torch.tensor([91, 92, 93, 2, 2, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        is_continuous_mask,
        torch.tensor([False, False, False, False, True, True, False]),
    )


def test_kimi_audio_generation_prompt_uses_prompt_builder(monkeypatch):
    captured = {}

    class FakeTokenizer:
        def encode(self, prompt: str):
            captured["prompt"] = prompt
            return [11, 22, 33]

    monkeypatch.setattr(
        kimi_audio_model,
        "cached_get_tokenizer",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    audio = np.zeros(8, dtype=np.float32)
    model_config = SimpleNamespace(
        tokenizer="unused",
        tokenizer_mode="kimi_audio",
        tokenizer_revision=None,
        trust_remote_code=True,
    )
    stt_config = SimpleNamespace(sample_rate=16000)

    prompt = KimiAudioForConditionalGeneration.get_generation_prompt(
        audio=audio,
        model_config=model_config,
        stt_config=stt_config,
        language=None,
        task_type="transcribe",
        request_prompt="Please transcribe the following audio:",
        to_language=None,
    )

    prompt_text = captured["prompt"]
    assert prompt_text == KimiAudioPromptBuilder.build_transcription_prompt(
        "Please transcribe the following audio:",
        audio_count=1,
    )
    assert prompt["prompt_token_ids"] == [11, 22, 33]
    assert np.array_equal(prompt["multi_modal_data"]["audio"], audio)
    assert prompt["mm_processor_kwargs"] == {
        "messages": [
            {
                "role": "user",
                "message_type": "text",
                "content": "Please transcribe the following audio:",
            },
            {
                "role": "user",
                "message_type": "audio",
            },
        ],
        "output_type": "text",
    }


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("hello world<|im_kimia_text_eos|>ignored", "hello world"),
        ("  hello world  ", "  hello world  "),
        ("", ""),
    ],
)
def test_kimi_audio_post_process_output(raw_text: str, expected: str):
    assert KimiAudioForConditionalGeneration.post_process_output(raw_text) == expected


class _FakeTextTokenizer:
    def __call__(self, text, return_tensors="pt", padding=True):
        batch = len(text)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
        }


class _FakeFeatureExtractor:
    hop_length = 4

    def __call__(
        self,
        audios,
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        assert padding == "max_length"
        batch = len(audios)
        return {
            "input_features": torch.ones((batch, 3, 5), dtype=torch.float32),
            "attention_mask": torch.ones((batch, 12), dtype=torch.long),
        }


class _FakeSpeechTokenizer:
    def encode(self, audios, sampling_rate=16000):
        if len(audios) == 1:
            return [[152064, 152065]]
        return [[152064, 152065], [152066]]


def test_kimi_audio_processor_can_return_discrete_speech_tokens():
    processor = KimiAudioProcessor(
        feature_extractor=_FakeFeatureExtractor(),
        tokenizer=_FakeTextTokenizer(),
        speech_tokenizer=_FakeSpeechTokenizer(),
    )

    outputs = processor(
        text=["hi", "hello"],
        audio=[np.ones(7, dtype=np.float32), np.ones(5, dtype=np.float32)],
        return_speech_token_ids=True,
    )

    assert outputs["whisper_input_features"].shape == (2, 3, 5)
    assert outputs["feature_attention_mask"].shape == (2, 12)
    assert outputs["audio_sample_lengths"].tolist() == [7, 5]
    assert outputs["speech_token_ids"].tolist() == [
        [152064, 152065],
        [152066, -1],
    ]
    assert outputs["speech_attention_mask"].tolist() == [
        [1, 1],
        [1, 0],
    ]


def test_kimi_audio_speech_tokenizer_prefers_local_override(monkeypatch, tmp_path):
    local_path = tmp_path / "glm-4-voice-tokenizer"
    local_path.mkdir()

    monkeypatch.setenv("KIMI_AUDIO_SPEECH_TOKENIZER_PATH", str(local_path))

    resolved = speech_utils._resolve_speech_tokenizer_path(
        "THUDM/glm-4-voice-tokenizer"
    )

    assert resolved == str(local_path)


def test_kimi_audio_speech_tokenizer_honors_device_override(monkeypatch):
    monkeypatch.setenv("KIMI_AUDIO_SPEECH_TOKENIZER_DEVICE", "cpu")

    tokenizer = speech_utils.KimiAudioSpeechTokenizer(
        model=SimpleNamespace(conv1=SimpleNamespace(weight=torch.ones(1))),
        feature_extractor=SimpleNamespace(sampling_rate=16000, hop_length=160),
    )

    assert tokenizer.device == "cpu"


def test_kimi_audio_speech_tokenizer_loads_vendored_whisper_vq(tmp_path):
    model_path = tmp_path / "glm-4-voice-tokenizer"
    model_path.mkdir()

    config = WhisperVQConfig(
        vocab_size=32,
        num_mel_bins=4,
        d_model=8,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        encoder_layers=4,
        max_source_positions=12,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        pooling_kernel_size=2,
        pooling_type="avg",
        pooling_position=2,
        quantize_vocab_size=4,
        quantize_position=4,
        quantize_encoder_only=True,
        quantize_ema_decay=0.99,
        quantize_causal_block_size=4,
        encoder_causal_convolution=True,
    )
    config.save_pretrained(model_path)

    model = WhisperVQEncoder(config)
    safetensors.torch.save_file(model.state_dict(), model_path / "model.safetensors")

    loaded = speech_utils._load_local_quantize_encoder(str(model_path), "cpu")

    assert isinstance(loaded, WhisperVQEncoder)
    assert loaded.config.quantize_encoder_only
    assert loaded.conv1.weight.shape == model.conv1.weight.shape


class _FakeProcessorTokenizer(_FakePromptTokenizer):
    def __call__(self, text, return_tensors="pt", padding=True):
        batch = len(text)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
        }


def test_kimi_audio_prompt_updates_use_speech_token_lengths():
    processor = object.__new__(kimi_audio_model.KimiAudioMultiModalProcessor)
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda: {
            "speech_token_ids": torch.tensor(
                [
                    [152064, 152065, -1],
                    [152066, 152067, 152068],
                ],
                dtype=torch.long,
            ),
            "speech_attention_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.long,
            ),
            "feature_attention_mask": torch.ones((2, 12), dtype=torch.long),
        }
    )

    prompt_updates = kimi_audio_model.KimiAudioMultiModalProcessor._get_prompt_updates(
        processor,
        mm_items=None,
        hf_processor_mm_kwargs={},
        out_mm_kwargs=out_mm_kwargs,
    )

    replacement = prompt_updates[0]
    assert replacement.target == [KimiAudioProcessor.KIMIA_TEXT_BLANK]
    assert replacement.replacement(0) == [KimiAudioProcessor.KIMIA_TEXT_BLANK] * 2
    assert replacement.replacement(1) == [KimiAudioProcessor.KIMIA_TEXT_BLANK] * 3


def test_kimi_audio_prompt_updates_can_use_audio_sample_lengths():
    processor = object.__new__(kimi_audio_model.KimiAudioMultiModalProcessor)
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda: {
            "audio_sample_lengths": torch.tensor([232880], dtype=torch.long),
        }
    )

    prompt_updates = kimi_audio_model.KimiAudioMultiModalProcessor._get_prompt_updates(
        processor,
        mm_items=None,
        hf_processor_mm_kwargs={},
        out_mm_kwargs=out_mm_kwargs,
    )

    replacement = prompt_updates[0]
    assert replacement.replacement(0) == [KimiAudioProcessor.KIMIA_TEXT_BLANK] * 182


def test_kimi_audio_processor_can_return_packed_kimi_token_streams():
    processor = KimiAudioProcessor(
        feature_extractor=_FakeFeatureExtractor(),
        tokenizer=_FakeProcessorTokenizer(),
        speech_tokenizer=_FakeSpeechTokenizer(),
    )

    outputs = processor(
        audio=[np.ones(7, dtype=np.float32)],
        messages=[
            {"role": "user", "message_type": "audio"},
        ],
        return_packed_kimi_tokens=True,
    )

    assert outputs["audio_token_ids"].tolist() == [[6, 2, 152064, 152065, 3, 8, 1, 7]]
    assert outputs["text_token_ids"].tolist() == [[4, 4, 4, 4, 4, 4, 4, 4]]
    assert outputs["is_continuous_mask"].tolist() == [
        [
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        ]
    ]


def test_kimi_audio_processor_batches_packed_kimi_token_streams_per_request():
    processor = KimiAudioProcessor(
        feature_extractor=_FakeFeatureExtractor(),
        tokenizer=_FakeProcessorTokenizer(),
        speech_tokenizer=_FakeSpeechTokenizer(),
    )

    outputs = processor(
        audio=[np.ones(7, dtype=np.float32), np.ones(5, dtype=np.float32)],
        messages=[
            {"role": "user", "message_type": "audio"},
        ],
        return_packed_kimi_tokens=True,
    )

    assert outputs["audio_token_ids"].tolist() == [
        [6, 2, 152064, 152065, 3, 8, 1, 7],
        [6, 2, 152066, 3, 8, 1, 7, 0],
    ]
    assert outputs["text_token_ids"].tolist() == [
        [4, 4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4, 0],
    ]
    assert outputs["is_continuous_mask"].tolist() == [
        [False, False, True, True, False, False, False, False],
        [False, False, True, False, False, False, False, False],
    ]


class _FakeEmbedTokens:
    def __call__(self, input_ids):
        return input_ids.unsqueeze(-1).repeat(1, 1, 2).to(torch.float32)


class _FakeLanguageModelCore:
    def __init__(self):
        self.embed_tokens = _FakeEmbedTokens()

    def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        return (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )


class _FakeLanguageModel:
    def __init__(self):
        self.model = _FakeLanguageModelCore()


class _FakeTupleLanguageModelCore(_FakeLanguageModelCore):
    def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        hidden_states = super().__call__(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states, [hidden_states + 10.0]


class _FakeTupleLanguageModel:
    def __init__(self):
        self.model = _FakeTupleLanguageModelCore()
        self.lm_head = object()


def test_kimi_audio_model_builds_packed_dual_stream_embeddings():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    audio_token_ids = torch.tensor([[6, 2, 152064, 152065, 3, 8, 1, 7]])
    text_token_ids = torch.tensor([[4, 4, 4, 4, 4, 4, 4, 4]])
    is_continuous_mask = torch.tensor(
        [[False, False, True, True, False, False, False, False]]
    )
    multimodal_embeddings = [
        torch.tensor(
            [
                [100.0, 100.0],
                [200.0, 200.0],
            ]
        )
    ]

    embeds = KimiAudioForConditionalGeneration._build_kimi_audio_inputs_embeds(
        model,
        audio_token_ids=audio_token_ids,
        text_input_ids=text_token_ids,
        is_continuous_mask=is_continuous_mask,
        multimodal_embeddings=multimodal_embeddings,
    )

    sqrt2 = 2**0.5
    expected = torch.tensor(
        [
            [10.0, 10.0],
            [6.0, 6.0],
            [(152064.0 + 100.0) * sqrt2 + 4.0, (152064.0 + 100.0) * sqrt2 + 4.0],
            [(152065.0 + 200.0) * sqrt2 + 4.0, (152065.0 + 200.0) * sqrt2 + 4.0],
            [7.0, 7.0],
            [12.0, 12.0],
            [5.0, 5.0],
            [11.0, 11.0],
        ]
    ).unsqueeze(0)
    assert torch.allclose(embeds, expected)


def test_kimi_audio_forward_accepts_flat_batch_request_lists():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([0, 0, 0, 0, 0], dtype=torch.long),
        positions=torch.tensor([1, 2, 3, 0, 2], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11, 12, 13], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ],
        text_token_ids=[
            torch.tensor([1, 1, 1, 1], dtype=torch.long),
            torch.tensor([2, 2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, False, False, False]),
            torch.tensor([False, False, False]),
        ],
    )

    expected = torch.tensor(
        [
            [12.0, 12.0],
            [13.0, 13.0],
            [14.0, 14.0],
            [22.0, 22.0],
            [24.0, 24.0],
        ]
    )
    assert torch.equal(outputs, expected)


def test_kimi_audio_forward_supports_mixed_decode_and_prefill_batch():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([99, 0, 0, 0, 0], dtype=torch.long),
        positions=torch.tensor([3, 0, 1, 0, 2], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ],
        text_token_ids=[
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([2, 2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, True]),
            torch.tensor([False, True, True]),
        ],
    )

    expected = torch.tensor(
        [
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
            ],
            [11.0, 11.0],
            [12.0, 12.0],
            [22.0, 22.0],
            [24.0, 24.0],
        ]
    )
    assert torch.equal(outputs, expected)


def test_kimi_audio_forward_supports_single_prompt_request_after_decode():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([99, 0], dtype=torch.int32),
        positions=torch.tensor([4, 2], dtype=torch.long),
        audio_token_ids=torch.tensor([[20, 21, 22]], dtype=torch.long),
        text_token_ids=torch.tensor([[2, 2, 2]], dtype=torch.long),
        is_continuous_mask=torch.tensor([[False, True, True]]),
    )

    expected = torch.tensor(
        [
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
            ],
            [24.0, 24.0],
        ]
    )
    assert torch.equal(outputs, expected)


def test_kimi_audio_forward_supports_single_prompt_request_before_decode():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([0, 0, 0, 99], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 4], dtype=torch.long),
        audio_token_ids=torch.tensor([[20, 21, 22]], dtype=torch.long),
        text_token_ids=torch.tensor([[2, 2, 2]], dtype=torch.long),
        is_continuous_mask=torch.tensor([[False, True, True]]),
    )

    expected = torch.tensor(
        [
            [22.0, 22.0],
            [23.0, 23.0],
            [24.0, 24.0],
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
            ],
        ]
    )
    assert torch.equal(outputs, expected)


def test_kimi_audio_forward_supports_interleaved_decode_between_prompts():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([0, 0, 99, 0, 0], dtype=torch.int32),
        positions=torch.tensor([0, 1, 4, 0, 1], dtype=torch.long),
        audio_token_ids=[
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ],
        text_token_ids=[
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([2, 2], dtype=torch.long),
        ],
        is_continuous_mask=[
            torch.tensor([False, True]),
            torch.tensor([False, True]),
        ],
    )

    expected = torch.tensor(
        [
            [11.0, 11.0],
            [12.0, 12.0],
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 99.0,
            ],
            [22.0, 22.0],
            [23.0, 23.0],
        ]
    )
    assert torch.equal(outputs, expected)


def test_kimi_audio_forward_uses_runtime_request_boundaries():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()
    full_audio_ids = torch.cat(
        [
            torch.full((176,), 7, dtype=torch.long),
            torch.tensor([20, 21, 22, 23], dtype=torch.long),
        ]
    )
    full_text_ids = torch.cat(
        [
            torch.full((176,), 3, dtype=torch.long),
            torch.tensor([2, 2, 2, 2], dtype=torch.long),
        ]
    )
    full_continuous_mask = torch.cat(
        [
            torch.zeros(176, dtype=torch.bool),
            torch.tensor([False, True, True, False]),
        ]
    )

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([91, 92, 93, 0, 0, 0, 0], dtype=torch.int32),
        positions=torch.tensor([176, 192, 208, 176, 177, 178, 179], dtype=torch.long),
        audio_token_ids=full_audio_ids.unsqueeze(0),
        text_token_ids=full_text_ids.unsqueeze(0),
        is_continuous_mask=full_continuous_mask.unsqueeze(0),
        runtime_request_num_scheduled_tokens=torch.tensor(
            [1, 1, 1, 4], dtype=torch.int32
        ),
        runtime_request_has_raw_mm_inputs=torch.tensor(
            [False, False, False, True], dtype=torch.bool
        ),
    )

    expected = torch.tensor(
        [
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 91.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 91.0,
            ],
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 92.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 92.0,
            ],
            [
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 93.0,
                KimiAudioProcessor.KIMIA_TEXT_BLANK + 93.0,
            ],
            [22.0, 22.0],
            [23.0, 23.0],
            [24.0, 24.0],
            [25.0, 25.0],
        ]
    )
    assert torch.equal(outputs, expected)


def test_kimi_audio_model_matches_official_bfloat16_fusion_formula():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeLanguageModel()
    original_embed_tokens = model.language_model.model.embed_tokens
    model.language_model.model.embed_tokens = lambda ids: original_embed_tokens(ids).to(
        torch.bfloat16
    )

    audio_token_ids = torch.tensor([[152064, 6]], dtype=torch.long)
    text_token_ids = torch.tensor([[4, 4]], dtype=torch.long)
    is_continuous_mask = torch.tensor([[True, False]])
    multimodal_embeddings = [
        torch.tensor([[100.0, 100.0]], dtype=torch.bfloat16),
    ]

    embeds = KimiAudioForConditionalGeneration._build_kimi_audio_inputs_embeds(
        model,
        audio_token_ids=audio_token_ids,
        text_input_ids=text_token_ids,
        is_continuous_mask=is_continuous_mask,
        multimodal_embeddings=multimodal_embeddings,
    )

    audio_emb = model.language_model.model.embed_tokens(audio_token_ids).to(
        torch.bfloat16
    )
    text_emb = model.language_model.model.embed_tokens(text_token_ids).to(
        torch.bfloat16
    )
    whisper_emb = torch.zeros_like(audio_emb)
    whisper_emb[0, 0] = torch.tensor([100.0, 100.0], dtype=torch.bfloat16)
    continuous_mask = is_continuous_mask[:, :, None].to(torch.bool)
    sqrt_two = torch.sqrt(
        torch.tensor(2.0, dtype=audio_emb.dtype, device=audio_emb.device)
    )
    expected = (
        audio_emb * (~continuous_mask)
        + ((audio_emb + whisper_emb) * sqrt_two) * continuous_mask
    ) + text_emb

    assert torch.equal(embeds, expected)


def test_kimi_audio_model_normalizes_chunked_multimodal_embeddings():
    model = object.__new__(KimiAudioForConditionalGeneration)

    normalized = KimiAudioForConditionalGeneration._normalize_multimodal_embeddings(
        model,
        multimodal_embeddings=[
            torch.tensor(
                [
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                ]
            ),
            torch.tensor(
                [
                    [[5.0, 5.0], [6.0, 6.0]],
                    [[7.0, 7.0], [8.0, 8.0]],
                ]
            ),
        ],
        batch_size=4,
    )

    assert len(normalized) == 4
    assert torch.equal(normalized[0], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert torch.equal(normalized[3], torch.tensor([[7.0, 7.0], [8.0, 8.0]]))


def test_kimi_audio_projector_matches_official_activation_and_norm():
    projector = kimi_audio_model.KimiAudioMultiModalProjector(norm_eps=1e-6)

    assert isinstance(projector.vq_adaptor_activation, torch.nn.SiLU)
    assert projector.vq_adaptor_layers_4.eps == pytest.approx(1e-6)


def test_kimi_audio_model_process_audio_input_accepts_feature_lists():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.audio_tower = lambda features: torch.ones(
        (len(features), 8, 2), dtype=torch.float32
    )
    model.multi_modal_projector = lambda features: features + 3.0

    audio_embeds = KimiAudioForConditionalGeneration._process_audio_input(
        model,
        {"whisper_input_features": [torch.ones((3, 5)), torch.ones((3, 5))]},
    )

    assert len(audio_embeds) == 2
    assert audio_embeds[0].shape == (2, 8)
    assert torch.allclose(audio_embeds[0], torch.full((2, 8), 4.0))


def test_kimi_audio_model_process_audio_input_accepts_tuple_batches():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.audio_tower = lambda features: torch.ones(
        (len(features), 6, 2), dtype=torch.float32
    )
    model.multi_modal_projector = lambda features: features + 2.0

    audio_embeds = KimiAudioForConditionalGeneration._process_audio_input(
        model,
        {
            "whisper_input_features": (
                torch.ones((3, 5)),
                torch.ones((1, 3, 7)),
            )
        },
    )

    assert len(audio_embeds) == 2
    assert audio_embeds[0].shape == (2, 8)
    assert torch.allclose(
        audio_embeds[1],
        torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0],
            ]
        ),
    )


def test_kimi_audio_model_process_audio_input_trims_batched_padding_with_mask():
    captured_lengths = []
    model = object.__new__(KimiAudioForConditionalGeneration)

    def fake_audio_tower(features):
        captured_lengths.append(features[0].shape[-1])
        return torch.ones((len(features), 8, 2), dtype=torch.float32)

    model.audio_tower = fake_audio_tower
    model.multi_modal_projector = lambda features: features

    _ = KimiAudioForConditionalGeneration._process_audio_input(
        model,
        {
            "whisper_input_features": torch.ones((2, 3, 8), dtype=torch.float32),
            "feature_attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=torch.long,
            ),
        },
    )

    assert captured_lengths == [5, 7]


def test_kimi_audio_model_process_audio_input_uses_sample_length_post_encoder_trim():
    captured = {}
    model = object.__new__(KimiAudioForConditionalGeneration)

    def fake_audio_tower(features, input_lengths=None):
        captured["feature_length"] = features[0].shape[-1]
        captured["input_lengths"] = input_lengths
        output_length = 1500 if input_lengths is None else input_lengths[0]
        return torch.ones((len(features), output_length, 2), dtype=torch.float32)

    model.audio_tower = fake_audio_tower
    model.multi_modal_projector = lambda features: features

    audio_embeds = KimiAudioForConditionalGeneration._process_audio_input(
        model,
        {
            "whisper_input_features": torch.ones((1, 3, 3000), dtype=torch.float32),
            "audio_sample_lengths": torch.tensor([232880], dtype=torch.long),
            "feature_attention_mask": torch.tensor(
                [[1] * 1456 + [0] * (3000 - 1456)],
                dtype=torch.long,
            ),
        },
    )

    assert captured["feature_length"] == 3000
    assert captured["input_lengths"] is None
    assert audio_embeds[0].shape == (182, 8)


def test_kimi_audio_model_keeps_main_hidden_states_for_text_output(monkeypatch):
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = _FakeTupleLanguageModel()
    model.use_mimo_text_path = True
    model.mimo_model = lambda positions, hidden_states: hidden_states + 5.0

    monkeypatch.setattr(
        kimi_audio_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    outputs = KimiAudioForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        positions=torch.tensor([0, 1], dtype=torch.long),
    )

    blank = float(KimiAudioProcessor.KIMIA_TEXT_BLANK)
    expected = torch.tensor([[[1.0 + blank, 1.0 + blank], [2.0 + blank, 2.0 + blank]]])
    assert torch.allclose(outputs, expected)


def test_kimi_audio_model_compute_logits_uses_lm_head_for_text_output(monkeypatch):
    captured: dict[str, object] = {}
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.use_mimo_text_path = True
    model.mimo_output = object()
    model.language_model = SimpleNamespace(lm_head=object())
    model.logits_processor = (
        lambda lm_head, hidden_states, sampling_metadata: captured.setdefault(
            "lm_head",
            lm_head,
        )
        or hidden_states
    )

    monkeypatch.setattr(
        kimi_audio_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    _ = KimiAudioForConditionalGeneration.compute_logits(
        model,
        hidden_states=torch.ones((1, 2, 3)),
        sampling_metadata=None,
    )

    assert captured["lm_head"] is model.language_model.lm_head


def test_kimi_audio_embed_input_ids_uses_all_multimodal_embeddings():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=lambda input_ids: input_ids.unsqueeze(-1)
            .repeat(1, 2)
            .to(torch.float32)
        )
    )

    input_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    is_multimodal = torch.tensor([False, True, True, True, False])
    multimodal_embeddings = (
        torch.tensor([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]),
    )

    embeds = KimiAudioForConditionalGeneration.embed_input_ids(
        model,
        input_ids=input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    sqrt2 = 2**0.5
    blank = float(KimiAudioProcessor.KIMIA_TEXT_BLANK)
    expected = torch.tensor(
        [
            [10.0 + blank, 10.0 + blank],
            [(11.0 + blank + 100.0) * sqrt2, (11.0 + blank + 100.0) * sqrt2],
            [(12.0 + blank + 200.0) * sqrt2, (12.0 + blank + 200.0) * sqrt2],
            [(13.0 + blank + 300.0) * sqrt2, (13.0 + blank + 300.0) * sqrt2],
            [14.0 + blank, 14.0 + blank],
        ]
    )

    assert torch.allclose(embeds, expected)


def test_kimi_audio_embed_input_ids_does_not_spill_embeddings_across_segments():
    model = object.__new__(KimiAudioForConditionalGeneration)
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=lambda input_ids: input_ids.unsqueeze(-1)
            .repeat(1, 2)
            .to(torch.float32)
        )
    )

    input_ids = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    is_multimodal = torch.tensor([False, True, True, False, True, False])
    multimodal_embeddings = (
        torch.tensor([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0]]),
        torch.tensor([[200.0, 200.0]]),
    )

    embeds = KimiAudioForConditionalGeneration.embed_input_ids(
        model,
        input_ids=input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    sqrt2 = 2**0.5
    blank = float(KimiAudioProcessor.KIMIA_TEXT_BLANK)
    expected = torch.tensor(
        [
            [10.0 + blank, 10.0 + blank],
            [(11.0 + blank + 100.0) * sqrt2, (11.0 + blank + 100.0) * sqrt2],
            [(12.0 + blank + 101.0) * sqrt2, (12.0 + blank + 101.0) * sqrt2],
            [13.0 + blank, 13.0 + blank],
            [(14.0 + blank + 200.0) * sqrt2, (14.0 + blank + 200.0) * sqrt2],
            [15.0 + blank, 15.0 + blank],
        ]
    )

    assert torch.allclose(embeds, expected)


def test_kimi_audio_multimodal_processor_requests_packed_kimi_tokens():
    captured = {}
    processor = object.__new__(kimi_audio_model.KimiAudioMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: object(),
        ctx=SimpleNamespace(
            call_hf_processor=lambda hf_processor, inputs, kwargs: captured.update(
                {
                    "inputs": inputs,
                    "kwargs": kwargs,
                }
            )
            or kwargs
        ),
    )

    _ = kimi_audio_model.KimiAudioMultiModalProcessor._call_hf_processor(
        processor,
        prompt="unused",
        mm_data={"audios": [np.ones(4, dtype=np.float32)]},
        mm_kwargs={"messages": [{"role": "user", "message_type": "audio"}]},
        tok_kwargs={},
    )

    assert "audio" in captured["inputs"]
    assert captured["kwargs"]["return_packed_kimi_tokens"] is True


def test_kimi_audio_whisper_attention_matches_official_bias_layout():
    config = kimi_audio_model.HFWhisperConfig(
        d_model=8,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        activation_function="gelu",
        num_mel_bins=3,
        max_source_positions=10,
        encoder_layers=1,
    )

    attention = kimi_audio_model.KimiAudioWhisperAttention(config)

    assert attention.q_proj.bias is not None
    assert attention.k_proj.bias is None
    assert attention.v_proj.bias is not None


def test_kimi_audio_process_audio_input_slices_after_encoder():
    model = object.__new__(KimiAudioForConditionalGeneration)

    captured = {}

    def fake_iter_single_audio_features(input_features, feature_attention_mask):
        captured["feature_attention_mask"] = feature_attention_mask
        return [input_features[0]]

    def fake_audio_tower(features, input_lengths=None):
        captured["input_lengths"] = input_lengths
        return torch.arange(16, dtype=torch.float32).view(1, 8, 2)

    model._iter_single_audio_features = fake_iter_single_audio_features
    model.audio_tower = fake_audio_tower
    model._project_audio_features = lambda audio_features: audio_features

    outputs = KimiAudioForConditionalGeneration._process_audio_input(
        model,
        {
            "whisper_input_features": torch.ones((1, 3, 5), dtype=torch.float32),
            "feature_attention_mask": torch.ones((1, 12), dtype=torch.long),
            "audio_sample_lengths": torch.tensor([7], dtype=torch.long),
        },
    )

    assert captured["feature_attention_mask"] is None
    assert captured["input_lengths"] is None
    assert len(outputs) == 1
    assert outputs[0].shape == (4, 2)
    assert torch.equal(outputs[0], torch.arange(8, dtype=torch.float32).view(4, 2))
