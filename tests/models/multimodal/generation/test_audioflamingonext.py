# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm import LLM, SamplingParams

MODEL_NAME = "nvidia/audio-flamingo-next-hf"
SINGLE_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is surprising about the relationship between "
                "the barking and the music?",
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://huggingface.co/datasets/nvidia/AudioSkills/"
                    "resolve/main/assets/"
                    "dogs_barking_in_sync_with_the_music.wav",
                },
            },
        ],
    }
]
BATCHED_CONVERSATIONS = [
    SINGLE_CONVERSATION,
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Why is the philosopher's name mentioned in the "
                    "lyrics? (A) To express a sense of nostalgia "
                    "(B) To indicate that language cannot express clearly, "
                    "satirizing the inversion of black and white in the world "
                    "(C) To add depth and complexity to the lyrics "
                    "(D) To showcase the wisdom and influence of the "
                    "philosopher",
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://huggingface.co/datasets/nvidia/"
                        "AudioSkills/resolve/main/assets/"
                        "Ch6Ae9DT6Ko_00-04-03_00-04-31.wav",
                    },
                },
            ],
        }
    ],
]


def get_fixture_path(filename):
    return os.path.join(
        os.path.dirname(__file__), "../../fixtures/audioflamingonext", filename
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


@pytest.fixture(scope="module")
def llm():
    model_info = HF_EXAMPLE_MODELS.get_hf_info(
        "AudioFlamingoNextForConditionalGeneration"
    )
    model_info.check_transformers_version(on_fail="skip")

    try:
        return LLM(
            model=MODEL_NAME,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1},
        )
    except Exception as e:
        pytest.skip(f"Failed to load model {MODEL_NAME}: {e}")


def test_single_generation(llm):
    fixture_path = get_fixture_path("expected_results_single.json")
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        expected = json.load(f)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = llm.chat(
        messages=SINGLE_CONVERSATION,
        sampling_params=sampling_params,
    )
    assert_output_matches(
        outputs[0],
        expected["transcriptions"][0],
        expected["token_ids"][0],
    )


def test_batched_generation(llm):
    fixture_path = get_fixture_path("expected_results_batched.json")
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        expected = json.load(f)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = llm.chat(
        messages=BATCHED_CONVERSATIONS,
        sampling_params=sampling_params,
    )

    for i, output in enumerate(outputs):
        assert_output_matches(
            output,
            expected["transcriptions"][i],
            expected["token_ids"][i],
        )


def test_single_and_batched_generation_match(llm):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    single_output = llm.chat(
        messages=SINGLE_CONVERSATION,
        sampling_params=sampling_params,
    )[0]
    batched_output = llm.chat(
        messages=BATCHED_CONVERSATIONS,
        sampling_params=sampling_params,
    )[0]

    assert single_output.outputs[0].text == batched_output.outputs[0].text
    assert list(single_output.outputs[0].token_ids) == list(
        batched_output.outputs[0].token_ids
    )
