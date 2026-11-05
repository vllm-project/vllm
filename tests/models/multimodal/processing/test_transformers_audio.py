# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import transformers
from packaging.version import Version

from vllm.config import ModelConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.model_executor.model_loader import get_model_cls
from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.transformers.multimodal import (
    MultiModalProcessor as TransformersMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processor import cached_processor_from_config

AUDIO_MODEL_SETTINGS = {
    "ibm-granite/granite-speech-3.3-2b": {
        "prompt": (
            "<|start_of_role|>system<|end_of_role|>"
            "You are a helpful AI assistant<|end_of_text|>\n"
            "<|start_of_role|>user<|end_of_role|>"
            "<|audio|>can you transcribe the speech into a written format?"
            "<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        ),
    },
    "nvidia/audio-flamingo-3-hf": {
        "prompt": (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<sound>Transcribe the input speech.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "mistralai/Voxtral-Mini-3B-2507": {
        "prompt": ("[INST][AUDIO]What can you tell me about this audio?[/INST]"),
    },
    "microsoft/VibeVoice-ASR-HF": {
        "prompt": (
            "<|im_start|>system\n"
            "You are a helpful assistant that transcribes audio input "
            "into text output in JSON format.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|object_ref_start|><|box_start|><|object_ref_end|>\n"
            "This is a 1.0 seconds audio, please transcribe it with "
            "these keys: Start time, End time, Speaker ID, Content"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "zai-org/GLM-ASR-Nano-2512": {
        "prompt": (
            "<|user|>\n"
            "<|begin_of_audio|><|pad|><|end_of_audio|><|user|>\n"
            "Please transcribe this audio into text"
            "<|assistant|>\n"
        ),
    },
}


@pytest.mark.parametrize(
    "model_id",
    [
        "ibm-granite/granite-speech-3.3-2b",
        "nvidia/audio-flamingo-3-hf",
        pytest.param(
            "mistralai/Voxtral-Mini-3B-2507",
            marks=pytest.mark.xfail(
                reason="Voxtral's mistral_common processor does not compose with "
                "the Transformers modelling backend. Loading it currently fails "
                "outright, because MistralCommonBackend.from_pretrained rejects "
                "the kwargs vLLM passes, and it implements no "
                "`replace_audio_token`, so it would report no replacement "
                "offsets. Both fixes belong in mistral_common or transformers.",
                strict=False,
            ),
        ),
        "microsoft/VibeVoice-ASR-HF",
        "zai-org/GLM-ASR-Nano-2512",
    ],
)
def test_audio_multimodal_processor(model_id):
    settings = AUDIO_MODEL_SETTINGS[model_id]

    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )

    audio = np.zeros(16000, dtype=np.float32)
    mm_data = {"audio": (audio, 16000)}

    result = mm_processor(
        prompt=settings["prompt"],
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    assert "prompt_token_ids" in result
    assert len(result["prompt_token_ids"]) > 0

    mm_placeholders = result.get("mm_placeholders", {})
    assert "audio" in mm_placeholders, f"No audio placeholders found for {model_id}"
    assert len(mm_placeholders["audio"]) == 1

    placeholder = mm_placeholders["audio"][0]
    assert placeholder.length > 0
    assert placeholder.offset >= 0

    audio_items = result.get("mm_kwargs", {}).get("audio", [])
    assert len(audio_items) == 1, f"Expected 1 audio item, got {len(audio_items)}"
    assert audio_items[0] is not None
    item_keys = list(audio_items[0].keys())
    has_features = "input_features" in item_keys or "input_values" in item_keys
    assert has_features, (
        f"No audio features (input_features/input_values) in {item_keys} for {model_id}"
    )


@pytest.mark.parametrize("separator", [" and ", ""])
def test_audio_multiple_inputs(separator):
    """Multiple audios per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers modelling backend."""
    model_id = "ibm-granite/granite-speech-3.3-2b"
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )

    audio_token = mm_processor.info.get_hf_processor().audio_token
    # One token per audio; the processor expands each to its placeholder run.
    prompt = (
        "<|start_of_role|>user<|end_of_role|>"
        f"{audio_token}{separator}{audio_token} transcribe<|end_of_text|>\n"
    )
    audios = [np.zeros(16000, dtype=np.float32), np.zeros(24000, dtype=np.float32)]

    result = mm_processor(
        prompt=prompt,
        mm_items=mm_processor.info.parse_mm_data({"audio": audios}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["mm_placeholders"]["audio"]) == 2
    assert len(result["mm_kwargs"]["audio"]) == 2


def test_audio_fields_not_claimed_by_image():
    """Audio fields survive when the image branch is also active."""
    model_id = "ibm-granite/granite-speech-3.3-2b"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    assert isinstance(mm_processor, TransformersMultiModalProcessor)

    audio_keys = ["input_features", "input_features_mask"]
    owned = mm_processor._partition_keys_by_modality(audio_keys, ["audio", "image"])

    assert owned["audio"] == audio_keys
    assert owned["image"] == []


def test_unclaimed_fields_warn_rather_than_raise():
    """Keys no sub-processor declares are dropped with a warning, not an error."""
    model_id = "ibm-granite/granite-speech-3.3-2b"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    assert isinstance(mm_processor, TransformersMultiModalProcessor)

    owned = mm_processor._partition_keys_by_modality(
        ["input_features", "surprise_field"], ["audio", "image"]
    )

    assert owned["audio"] == ["input_features"]
    assert owned["image"] == []


@pytest.mark.parametrize(
    ("model_id", "long_form"),
    [
        ("zai-org/GLM-ASR-Nano-2512", False),
        ("ibm-granite/granite-speech-3.3-2b", False),
        ("Qwen/Qwen3-ASR-1.7B-hf", False),
        ("microsoft/VibeVoice-ASR-HF", True),
    ],
)
def test_only_long_form_audio_fills_the_context(model_id, long_form):
    """Audio models keep their usual budget unless they take whole recordings, as
    sizing them all by the context stops small GPUs from starting."""
    seq_len = 4096
    mm_processor = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(model=model_id, model_impl="transformers")
    )
    info = mm_processor.info

    max_tokens_per_item = info.get_mm_max_tokens_per_item(seq_len, {"audio": 1})
    assert max_tokens_per_item is not None
    max_tokens = max_tokens_per_item["audio"]
    result = mm_processor(
        prompt=mm_processor.dummy_inputs.get_dummy_text({"audio": 1}),
        mm_items=info.parse_mm_data(
            mm_processor.dummy_inputs.get_dummy_mm_data(seq_len, {"audio": 1}, {})
        ),
        hf_processor_mm_kwargs={},
    )
    num_tokens = result["mm_placeholders"]["audio"][0].length

    if long_form:
        assert max_tokens == seq_len
        assert num_tokens == pytest.approx(seq_len, rel=0.01)
    else:
        assert max_tokens != seq_len
        assert num_tokens <= max_tokens


@pytest.mark.parametrize(
    ("limit", "expected_tokens"),
    [({"count": 1}, 27000), ({"count": 1, "length": 24000 * 60}, 450)],
)
def test_long_form_audio_limit(limit, expected_tokens):
    """Long-form audio stops at an hour unless the audio limit says otherwise."""
    info = MULTIMODAL_REGISTRY.create_processor(
        ModelConfig(
            model="microsoft/VibeVoice-ASR-HF",
            model_impl="transformers",
            limit_mm_per_prompt={"audio": limit},
        )
    ).info

    max_tokens = info.get_mm_max_tokens_per_item(10**6, {"audio": 1})
    assert max_tokens == {"audio": expected_tokens}


@pytest.mark.parametrize(
    ("model_id", "transcribes"),
    [
        ("microsoft/VibeVoice-ASR-HF", True),
        ("zai-org/GLM-ASR-Nano-2512", True),
        ("Qwen/Qwen3-ASR-1.7B-hf", True),
        ("ibm-granite/granite-speech-3.3-2b", False),
    ],
)
def test_transcription_needs_a_transcription_request(model_id, transcribes):
    model_config = ModelConfig(model=model_id, model_impl="transformers")

    assert supports_transcription(get_model_cls(model_config)) == transcribes


def test_qwen3_asr_config_follows_the_checkpoint_format():
    """VLLM's own config would drop the `audio_config` of a Transformers checkpoint."""
    hf_config = ModelConfig(model="Qwen/Qwen3-ASR-1.7B-hf").hf_config
    native_config = ModelConfig(model="Qwen/Qwen3-ASR-1.7B").hf_config

    assert hf_config.audio_config is not None
    assert not hasattr(native_config, "audio_config")


VIBEVOICE_OUTPUT = (
    'assistant\n[{"Start":0.0,"End":1.5,"Speaker":0,"Content":"Hello there."},'
    '{"Start":1.62,"End":3,"Speaker":1,"Content":"Hi!"},'
    '{"Start":3,"End":4,"Content":" "}]\n'
)


@pytest.fixture
def vibevoice():
    if Version(transformers.__version__) < Version("5.23.0.dev0"):
        pytest.skip("Requires `tokenize` on `apply_transcription_request`")
    model_config = ModelConfig(
        model="microsoft/VibeVoice-ASR-HF", model_impl="transformers"
    )
    return get_model_cls(model_config), model_config


def test_transcription_config_keeps_uncapped_audio_whole(vibevoice):
    model_cls, model_config = vibevoice

    stt_config = model_cls.get_speech_to_text_config(model_config, "transcribe")

    assert stt_config.sample_rate == 24000
    assert stt_config.max_audio_clip_s is None


@pytest.mark.parametrize("context", ["", "Speakers: Ada"])
def test_transcription_prompt_matches_hf_processor(vibevoice, context):
    """VLLM never sees the text the HF processor fills in, so the prompt it builds
    has to tokenize to what the processor would give."""
    model_cls, model_config = vibevoice
    audio = np.zeros(36000, dtype=np.float32)
    stt_config = model_cls.get_speech_to_text_config(model_config, "transcribe")

    prompt = model_cls.get_generation_prompt(
        SpeechToTextParams(
            audio=audio,
            stt_config=stt_config,
            model_config=model_config,
            request_prompt=context,
        )
    )

    processor = cached_processor_from_config(model_config)
    hf_inputs = processor.apply_transcription_request(audio, prompt=context or None)
    ours = processor.tokenizer.encode(prompt["prompt"], add_special_tokens=False)
    audio_token_id = processor.audio_token_id
    hf_ids = hf_inputs["input_ids"][0].tolist()
    assert ours == [
        token
        for i, token in enumerate(hf_ids)
        if token != audio_token_id or hf_ids[i - 1] != audio_token_id
    ]


def test_transcription_rejects_translation(vibevoice):
    model_cls, model_config = vibevoice

    with pytest.raises(ValueError, match="translation"):
        model_cls.get_generation_prompt(
            SpeechToTextParams(
                audio=np.zeros(24000, dtype=np.float32),
                stt_config=model_cls.get_speech_to_text_config(
                    model_config, "translate"
                ),
                model_config=model_config,
                task_type="translate",
            )
        )


def test_transcription_output_parsing(vibevoice):
    model_cls, model_config = vibevoice

    text = model_cls.post_process_output(VIBEVOICE_OUTPUT, model_config)
    segments = model_cls.parse_diarized_transcript(VIBEVOICE_OUTPUT, model_config)

    assert text == "Hello there. Hi!"
    assert [(s.start, s.end, s.speaker, s.text) for s in segments] == [
        (0.0, 1.5, "0", "Hello there."),
        (1.62, 3.0, "1", "Hi!"),
    ]


def test_transcription_output_parsing_fails_closed(vibevoice):
    model_cls, model_config = vibevoice

    assert model_cls.post_process_output("Hello.\n", model_config) == "Hello."
    assert model_cls.parse_diarized_transcript("Hello.", model_config) == []
