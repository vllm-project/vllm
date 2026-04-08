# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.envs import disable_envs_cache
from vllm.multimodal import MULTIMODAL_REGISTRY

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
                reason="MistralCommonBackend tokenizer does not produce audio "
                "placeholder token (ID 24) from text; requires "
                "apply_chat_template path",
                strict=False,
            ),
        ),
        "microsoft/VibeVoice-ASR-HF",
        "zai-org/GLM-ASR-Nano-2512",
    ],
)
def test_audio_multimodal_processor(model_id):
    settings = AUDIO_MODEL_SETTINGS[model_id]

    model_config = ModelConfig(
        model=model_id,
        model_impl="transformers",
    )

    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

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
    item_keys = list(audio_items[0].keys())
    has_features = "input_features" in item_keys or "input_values" in item_keys
    assert has_features, (
        f"No audio features (input_features/input_values) in {item_keys} for {model_id}"
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "ibm-granite/granite-speech-3.3-2b",
        "nvidia/audio-flamingo-3-hf",
        pytest.param(
            "mistralai/Voxtral-Mini-3B-2507",
            marks=pytest.mark.xfail(
                reason="MistralCommonBackend tokenizer does not produce audio "
                "placeholder token (ID 24) from text; requires "
                "apply_chat_template path",
                strict=False,
            ),
        ),
        "microsoft/VibeVoice-ASR-HF",
        "zai-org/GLM-ASR-Nano-2512",
    ],
)
def test_audio_model_loading(monkeypatch, vllm_runner, model_id):
    """Single-process workaround for V1 fork safety deadlock issue
    (vllm-project/vllm/issues/17676) in models with nested CausalLMs.
    For other models (i.e., Voxtral, VibeVoice, GLM-ASR-Nano) this workaround
    is not strictly required in practice, but running them together under pytest
    can cause (possibly flaky) hangs, so they are grouped under the same config.
    Using VLLM_WORKER_MULTIPROC_METHOD=spawn avoids the deadlock and allows worker
    processes to terminate cleanly, and release GPU memory between test runs until the
    issue is fixed."""
    # TODO: Remove monkeypatch once
    # https://github.com/vllm-project/vllm/issues/17676 is fixed.
    disable_envs_cache()
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    settings = AUDIO_MODEL_SETTINGS[model_id]

    with vllm_runner(
        model_id,
        model_impl="transformers",
        max_model_len=2048,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        audio = np.zeros(16000 * 2, dtype=np.float32)
        outputs = vllm_model.generate(
            prompts=[settings["prompt"]],
            sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
            audios=[(audio, 16000)],
        )
        assert len(outputs) == 1
        assert len(outputs[0][1]) > 0
