# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any

import pytest

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.envs import disable_envs_cache

AUDIO_ASSET = AudioAsset("mary_had_lamb")

AUDIO_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
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
        "llm_kwargs": {
            "gpu_memory_utilization": 0.85,
        },
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
        "llm_kwargs": {
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.85,
        },
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


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../fixtures/transformers_audio_expected_results.json",
)


def assert_output_matches(output, expected_text, expected_token_ids):
    generated = output.outputs[0]
    assert generated.text.strip() == expected_text
    actual_token_ids = list(generated.token_ids)
    assert (
        actual_token_ids == expected_token_ids
        or actual_token_ids == expected_token_ids[:-1]
        or actual_token_ids[:-1] == expected_token_ids
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "ibm-granite/granite-speech-3.3-2b",
        "nvidia/audio-flamingo-3-hf",
        "microsoft/VibeVoice-ASR-HF",
        "zai-org/GLM-ASR-Nano-2512",
    ],
)
def test_transformers_audio_generation(monkeypatch, model_id):
    """Single-process workaround for V1 fork safety deadlock issue
    (vllm-project/vllm/issues/17676). Running multiple audio models together
    under pytest can cause (possibly flaky) hangs, so they are grouped under
    the same config. Using VLLM_WORKER_MULTIPROC_METHOD=spawn avoids the
    deadlock and allows worker processes to terminate cleanly, and release
    GPU memory between test runs until the issue is fixed."""
    # TODO: Remove monkeypatch once
    # https://github.com/vllm-project/vllm/issues/17676 is fixed.
    disable_envs_cache()
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if not os.path.exists(FIXTURE_PATH):
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")

    with open(FIXTURE_PATH) as f:
        all_expected = json.load(f)

    if model_id not in all_expected:
        pytest.skip(f"No fixture data for {model_id}")

    expected = all_expected[model_id]
    settings = AUDIO_MODEL_SETTINGS[model_id]

    try:
        llm = LLM(
            model=model_id,
            model_impl="transformers",
            max_model_len=2048,
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1},
            **settings.get("llm_kwargs", {}),
        )
    except Exception as e:
        pytest.skip(f"Failed to load model {model_id}: {e}")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = llm.generate(
        prompts=[
            {
                "prompt": settings["prompt"],
                "multi_modal_data": {"audio": AUDIO_ASSET.audio_and_sample_rate},
            }
        ],
        sampling_params=sampling_params,
    )
    assert len(outputs) == 1
    assert_output_matches(
        outputs[0],
        expected["transcriptions"][0],
        expected["token_ids"][0],
    )
