# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest

from vllm.config import DeviceConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.models.granite_speech import get_encoder_batch_size
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget, get_mm_max_toks_per_item

from ...utils import build_model_context

MODEL = "ibm-granite/granite-speech-3.3-2b"


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(MODEL, limit_mm_per_prompt={"audio": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _process_audio(processor, seconds: float):
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    prompt = processor.dummy_inputs.get_dummy_text({"audio": 1})
    return processor(
        prompt, mm_items=processor.info.parse_mm_data({"audio": [(audio, 16000)]})
    )


def _num_features(processed) -> int:
    return processed["mm_kwargs"]["audio"][0]["input_features"].data.shape[0]


def test_audio_budget_covers_ten_minute_clips(processor):
    budget = get_mm_max_toks_per_item(processor, {"audio": 1}, None)["audio"]
    dummy = processor.get_dummy_mm_inputs({"audio": 1})
    ten_minutes = _process_audio(processor, 600)

    assert dummy["mm_placeholders"]["audio"][0].get_num_embeds() == budget
    assert ten_minutes["mm_placeholders"]["audio"][0].get_num_embeds() <= budget


@pytest.mark.parametrize("max_num_batched_tokens", [8192, 16384])
def test_encoder_calls_fit_the_profiled_call(max_num_batched_tokens):
    ctx = build_model_context(
        MODEL,
        model_config_kwargs={"max_model_len": 32768},
        limit_mm_per_prompt={"audio": 1},
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=256,
        max_model_len=32768,
        is_encoder_decoder=False,
    )
    budget = MultiModalBudget(
        VllmConfig(
            model_config=ctx.model_config,
            scheduler_config=scheduler_config,
            device_config=DeviceConfig(device="cpu"),
        ),
        MULTIMODAL_REGISTRY,
        enable_cache=False,
    )
    processor = budget.processor
    config = processor.info.get_hf_config()
    context_size = config.encoder_config.context_size
    max_tokens = scheduler_config.max_num_encoder_input_tokens

    dummy_features = _num_features(processor.get_dummy_mm_inputs({"audio": 1}))
    num_profiled = len(budget.get_dummy_encoder_profile_inputs())
    profiled_blocks = min(
        num_profiled, get_encoder_batch_size(config, dummy_features, max_tokens)
    ) * math.ceil(dummy_features / context_size)

    longest = budget.encoder_cache_size // 10
    for seconds in [2, 5, 60, 600, longest]:
        processed = _process_audio(processor, seconds)
        assert (
            processed["mm_placeholders"]["audio"][0].get_num_embeds()
            <= budget.encoder_cache_size
        )
        num_features = _num_features(processed)
        batch_size = get_encoder_batch_size(config, num_features, max_tokens)

        assert batch_size >= 1
        assert batch_size * math.ceil(num_features / context_size) <= profiled_blocks
