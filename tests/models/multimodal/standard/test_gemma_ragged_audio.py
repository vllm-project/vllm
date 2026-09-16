# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.assets.audio import AudioAsset

MODELS = {
    "Gemma3nForConditionalGeneration": "google/gemma-3n-E2B-it",
    "Gemma4ForConditionalGeneration": "google/gemma-4-E2B-it",
}

# Seconds. Spread wide enough that no two clips can produce the same number of
# mel frames: equal-length clips stack normally and never reach the ragged path.
CLIP_DURATIONS = (1.0, 3.5, 7.0, 12.0)


@pytest.fixture(scope="module")
def ragged_clips():
    """One asset sliced to several durations, so only length varies."""
    audio, sample_rate = AudioAsset("mary_had_lamb").audio_and_sample_rate

    clips = []
    for duration in CLIP_DURATIONS:
        num_samples = int(duration * sample_rate)
        if num_samples > len(audio):
            pytest.skip(f"asset is shorter than {duration}s")
        clips.append((audio[:num_samples], sample_rate))

    assert len({len(clip) for clip, _ in clips}) == len(clips)
    return clips


@pytest.mark.core_model
@pytest.mark.parametrize("arch", sorted(MODELS))
@pytest.mark.parametrize("max_tokens", [8])
def test_variable_length_audio_batching(
    vllm_runner,
    ragged_clips,
    arch: str,
    max_tokens: int,
) -> None:
    """Test batching of requests with different audio durations.

    Audio features are unpadded per item so a multimodal cache entry does not
    depend on the batch it was first processed in, and
    `MultiModalFieldConfig.batched` stacks items only when their shapes agree.
    A batch of differently sized clips therefore reaches the model as a list,
    which used to take EngineCore down with

        AttributeError: 'list' object has no attribute 'squeeze'

    failing every in-flight request with EngineDeadError.
    """
    from transformers import AutoProcessor

    model_name = MODELS[arch]
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # The two architectures spell the audio placeholder differently; ask the
    # processor rather than hardcoding either.
    processor = AutoProcessor.from_pretrained(model_name)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": None},
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    with vllm_runner(
        model_name,
        dtype="bfloat16",
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    ) as vllm_model:
        # One call so the clips are scheduled into the same forward pass;
        # issued separately they never reach the ragged path.
        outputs = vllm_model.generate_greedy(
            [prompt] * len(ragged_clips),
            max_tokens,
            audios=[[clip] for clip in ragged_clips],
        )

    assert len(outputs) == len(ragged_clips)
    for output in outputs:
        assert len(output[1]) > 0, "Expected non-empty output"
