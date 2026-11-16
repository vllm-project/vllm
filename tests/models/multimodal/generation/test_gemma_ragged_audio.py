# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest
import torch

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest

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


@pytest.mark.core_model
def test_single_frame_audio_keeps_engine_usable(vllm_runner) -> None:
    """Single-frame clips and subsequent ordinary audio produce completions."""
    from transformers import AutoProcessor

    model_name = MODELS["Gemma4ForConditionalGeneration"]
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
    clips = [
        (np.zeros(samples, dtype=np.float32), 16000) for samples in (161, 320, 321)
    ]
    clips.append(AudioAsset("mary_had_lamb").audio_and_sample_rate)

    with vllm_runner(
        model_name,
        dtype="bfloat16",
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    ) as vllm_model:
        # Mixed-length batching could pad a singleton frame and hide the bug.
        for clip in clips:
            outputs = vllm_model.llm.generate(
                vllm_model.get_inputs([prompt], audios=[[clip]]),
                sampling_params=SamplingParams(temperature=0, max_tokens=8),
            )
            assert len(outputs) == 1
            assert outputs[0].finished
            assert outputs[0].outputs[0].token_ids, "Expected generated tokens"


@pytest.mark.core_model
def test_ragged_audio_with_different_adapters(
    vllm_runner, ragged_clips, tmp_path, monkeypatch
) -> None:
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoProcessor

    # Queue both requests before the first encoder step.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_name = MODELS["Gemma4ForConditionalGeneration"]
    config = AutoConfig.from_pretrained(model_name)
    audio_size = config.audio_config.hidden_size
    projected_size = config.audio_config.output_proj_dims
    modules = {
        "audio_tower.output_proj": (audio_size, projected_size),
        "embed_audio.embedding_projection": (
            projected_size,
            config.text_config.hidden_size,
        ),
        "audio_tower.layers.0.self_attn.relative_k_proj": (audio_size, audio_size),
    }
    adapter_paths = []
    for index in range(2):
        path = tmp_path / f"audio-adapter-{index}"
        path.mkdir()
        generator = torch.Generator(device="cpu").manual_seed(42 + index)
        weights = {}
        for name, (inputs, outputs) in modules.items():
            for suffix, shape, scale in (
                ("A", (8, inputs), 8),
                ("B", (outputs, 8), 4),
            ):
                weights[f"base_model.model.{name}.lora_{suffix}.weight"] = (
                    torch.randn(shape, generator=generator, device="cpu") / scale
                ).to(torch.bfloat16)
        save_file(weights, str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 8,
                    "lora_alpha": 8,
                    "target_modules": [name.rsplit(".", 1)[-1] for name in modules],
                }
            )
        )
        adapter_paths.append(str(path))

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
        enable_thinking=False,
    )
    next_id = iter(range(1, 9))
    with vllm_runner(
        model_name,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=2,
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_loras=2,
        max_lora_rank=8,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
    ) as runner:

        def generate(indices):
            # Fresh IDs also force fresh encoder results for each comparison.
            requests = [
                LoRARequest(str(i := next(next_id)), i, adapter_paths[index])
                for index in indices
            ]
            return runner.generate_w_logprobs(
                [prompt] * len(indices),
                SamplingParams(
                    temperature=0, max_tokens=4, ignore_eos=True, logprobs=5
                ),
                audios=[[ragged_clips[index]] for index in indices],
                lora_request=requests,
            )

        singles = [generate([index])[0] for index in range(2)]
        for order in ([0, 1], [1, 0]):
            for actual, index in zip(generate(order), order):
                expected = singles[index]
                assert len(actual[0]) == len(expected[0]) == 4
                check_logprobs_close(
                    outputs_0_lst=[actual],
                    outputs_1_lst=[expected],
                    name_0="batched audio",
                    name_1="single audio",
                )
