# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm import LLM, SamplingParams

MODEL_NAME = "nvidia/music-flamingo-2601-hf"
SINGLE_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this track in full detail - tell me the "
                "genre, tempo, and key, then dive into the instruments, "
                "production style, and overall mood it creates.",
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": "https://huggingface.co/datasets/nvidia/AudioSkills/"
                    "resolve/main/assets/song_1.mp3",
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
                    "text": "Generate a structured lyric sheet from the input music.",
                },
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "https://huggingface.co/datasets/nvidia/"
                        "AudioSkills/resolve/main/assets/song_2.mp3",
                    },
                },
            ],
        }
    ],
]


def get_fixture_path(filename):
    return os.path.join(
        os.path.dirname(__file__), "../../fixtures/musicflamingo", filename
    )


def assert_output_matches(output, expected_text, expected_token_ids):
    generated = output.outputs[0]
    assert generated.text == expected_text
    actual_token_ids = list(generated.token_ids)
    assert (
        actual_token_ids == expected_token_ids
        or actual_token_ids == expected_token_ids[:-1]
        or actual_token_ids[:-1] == expected_token_ids
    )


@pytest.fixture(scope="module")
def llm():
    model_info = HF_EXAMPLE_MODELS.get_hf_info("MusicFlamingoForConditionalGeneration")
    model_info.check_transformers_version(on_fail="skip")

    try:
        return LLM(
            model=MODEL_NAME,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=8192,
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

    outputs = llm.chat(
        messages=SINGLE_CONVERSATION,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=50),
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

    outputs = llm.chat(
        messages=BATCHED_CONVERSATIONS,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=50),
    )

    for i, output in enumerate(outputs):
        assert_output_matches(
            output,
            expected["transcriptions"][i],
            expected["token_ids"][i],
        )


def test_single_and_batched_generation_match(llm):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

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
