# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

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


@pytest.mark.parametrize(
    ("modality", "key", "value"),
    [
        ("image", "max_image_tokens", 8192),
        ("video", "max_video_frame_tokens", 288),
    ],
)
def test_dummy_fills_budget_for_non_square_token_limits(
    modality: str, key: str, value: int
):
    budget, dummy = _budget_and_dummy(modality, {key: value})

    assert dummy == budget


@pytest.mark.parametrize(
    ("hint", "image_size"),
    [
        ({"width": 1792, "height": 2520}, (1792, 2520)),
        ({"width": 2520, "height": 1792}, (2520, 1792)),
        ({"width": 1792}, (1792, 3584)),
        ({"height": 1792}, (3584, 1792)),
        ({"width": 2357, "height": 2726}, (2357, 2726)),
    ],
)
def test_dummy_image_follows_size_hint(
    hint: dict[str, int], image_size: tuple[int, int]
):
    ctx = build_model_context(
        MODEL_ID,
        mm_processor_kwargs={"max_image_tokens": 8192},
        limit_mm_per_prompt={
            "image": {"count": 1, **hint},
            "video": 0,
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    processed = processor(
        processor.dummy_inputs.get_dummy_text({"image": 1}),
        mm_items=processor.info.parse_mm_data(
            {"image": [Image.new("RGB", image_size)]}
        ),
        hf_processor_mm_kwargs={},
    )
    dummy = processor.get_dummy_mm_inputs({"image": 1})

    assert (
        dummy["mm_placeholders"]["image"][0].get_num_embeds()
        >= processed["mm_placeholders"]["image"][0].get_num_embeds()
    )


@pytest.mark.parametrize(
    ("loader_frames", "num_frames"),
    [(None, 96), (97, 97), (128, 128), (129, 129), (-1, 200)],
)
def test_video_budget_covers_loader_frames(loader_frames: int | None, num_frames: int):
    ctx = build_model_context(
        MODEL_ID,
        model_config_kwargs={
            "max_model_len": 20480,
            "media_io_kwargs": (
                {"video": {"num_frames": loader_frames}} if loader_frames else {}
            ),
        },
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len, mm_counts={"video": 1}
    )
    assert tokens is not None
    video = np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    processed = processor(
        processor.dummy_inputs.get_dummy_text({"video": 1}),
        mm_items=processor.info.parse_mm_data({"video": [(video, metadata)]}),
        hf_processor_mm_kwargs={},
    )
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    real = processed["mm_placeholders"]["video"][0].get_num_embeds()
    assert real <= tokens["video"]
    assert dummy["mm_placeholders"]["video"][0].get_num_embeds() == tokens["video"]
