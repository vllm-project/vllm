# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

from ...utils import build_model_context


@pytest.mark.parametrize("max_model_len", [4096, 16384, 65536])
def test_audio_budget_matches_profiled_audio(max_model_len: int):
    """The audio budget must match the profiled dummy audio, which is capped
    at `MAX_AUDIO_LEN_S`."""
    ctx = build_model_context(
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"audio": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"audio": 1})
    num_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["audio"])

    assert budget["audio"] == num_embeds
