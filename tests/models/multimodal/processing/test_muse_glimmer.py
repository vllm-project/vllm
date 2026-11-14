# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "meta-models/Muse-Glimmer-30B"


def _budget_and_dummy(
    modality: str, mm_processor_kwargs: dict[str, object] | None
) -> tuple[int, int]:
    ctx = build_model_context(
        MODEL_ID,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 0, "video": 0} | {modality: 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={modality: 1},
    )
    assert tokens is not None
    dummy = processor.get_dummy_mm_inputs({modality: 1})["mm_placeholders"][modality]
    return tokens[modality], dummy[0].get_num_embeds()


@pytest.mark.parametrize(
    ("modality", "scope", "key", "value"),
    [
        ("image", "images_kwargs", "max_image_tokens", 8192),
        ("video", "videos_kwargs", "max_video_frame_tokens", 576),
    ],
)
def test_profiling_reads_scoped_processor_kwargs(
    modality: str, scope: str, key: str, value: int
):
    flat = _budget_and_dummy(modality, {key: value})
    scoped = _budget_and_dummy(modality, {scope: {key: value}})

    assert flat != _budget_and_dummy(modality, None)
    assert scoped == flat
