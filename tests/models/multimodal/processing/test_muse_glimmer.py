# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "meta-models/Muse-Glimmer-30B"


def _video(num_frames: int, height: int, width: int) -> tuple[np.ndarray, dict]:
    return (
        np.zeros((num_frames, height, width, 3), dtype=np.uint8),
        {
            "fps": 2.0,
            "duration": num_frames / 2.0,
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        },
    )


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
    processed = processor(
        processor.dummy_inputs.get_dummy_text({"video": 1}),
        mm_items=processor.info.parse_mm_data(
            {"video": [_video(num_frames, 336, 336)]}
        ),
        hf_processor_mm_kwargs={},
    )
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    real = processed["mm_placeholders"]["video"][0].get_num_embeds()
    assert real <= tokens["video"]
    assert dummy["mm_placeholders"]["video"][0].get_num_embeds() == tokens["video"]


@pytest.mark.parametrize(
    ("mm_processor_kwargs", "loader_frames", "modality", "shape", "rejected"),
    [
        (None, None, "video", (98, 336, 336), True),
        (None, 128, "video", (130, 336, 336), True),
        (None, None, "video", (200, 112, 112), False),
        ({"do_resize": False}, None, "video", (2, 336, 364), True),
        ({"do_resize": False}, None, "video", (2, 336, 336), False),
        ({"do_resize": False}, None, "image", (1, 1792, 1820), True),
        ({"do_resize": False}, None, "image", (1, 1792, 1792), False),
    ],
)
def test_items_above_profiled_tokens_are_rejected(
    mm_processor_kwargs: dict[str, object] | None,
    loader_frames: int | None,
    modality: str,
    shape: tuple[int, int, int],
    rejected: bool,
):
    ctx = build_model_context(
        MODEL_ID,
        mm_processor_kwargs=mm_processor_kwargs,
        model_config_kwargs={
            "max_model_len": 20480,
            "media_io_kwargs": (
                {"video": {"num_frames": loader_frames}} if loader_frames else {}
            ),
        },
        limit_mm_per_prompt={"image": 0, "video": 0} | {modality: 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    num_frames, height, width = shape
    if modality == "image":
        item: object = Image.new("RGB", (width, height))
    else:
        item = _video(num_frames, height, width)

    def process():
        return processor(
            processor.dummy_inputs.get_dummy_text({modality: 1}),
            mm_items=processor.info.parse_mm_data({modality: [item]}),
            hf_processor_mm_kwargs={},
        )

    if rejected:
        with pytest.raises(ValueError, match="Muse-Glimmer accepts at most"):
            process()
        return
    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len, mm_counts={modality: 1}
    )
    assert tokens is not None
    num_embeds = process()["mm_placeholders"][modality][0].get_num_embeds()
    assert 0 < num_embeds <= tokens[modality]


def test_frame_hint_shrinks_dummy_but_not_limit():
    ctx = build_model_context(
        MODEL_ID,
        model_config_kwargs={"max_model_len": 20480},
        limit_mm_per_prompt={"image": 0, "video": {"count": 1, "num_frames": 32}},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len, mm_counts={"video": 1}
    )
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    processed = processor(
        processor.dummy_inputs.get_dummy_text({"video": 1}),
        mm_items=processor.info.parse_mm_data({"video": [_video(96, 336, 336)]}),
        hf_processor_mm_kwargs={},
    )

    assert tokens is not None
    assert tokens["video"] == 6912
    assert dummy["mm_placeholders"]["video"][0].get_num_embeds() == 2304
    assert processed["mm_placeholders"]["video"][0].get_num_embeds() == 6912


@pytest.mark.parametrize(("num_frames", "dummy_tokens"), [(None, 2048), (6, 1728)])
def test_short_context_admits_more_groups_with_fewer_tokens(
    num_frames: int | None, dummy_tokens: int
):
    ctx = build_model_context(
        MODEL_ID,
        mm_processor_kwargs={"max_video_frame_tokens": 576},
        model_config_kwargs={"max_model_len": 2048},
        limit_mm_per_prompt={
            "image": 0,
            "video": {"count": 1, "num_frames": num_frames} if num_frames else 1,
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=2048, mm_counts={"video": 1}
    )
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    processed = processor(
        processor.dummy_inputs.get_dummy_text({"video": 1}),
        mm_items=processor.info.parse_mm_data({"video": [_video(8, 448, 784)]}),
        hf_processor_mm_kwargs={},
    )

    assert tokens is not None
    assert tokens["video"] == 2048
    assert dummy["mm_placeholders"]["video"][0].get_num_embeds() == dummy_tokens
    assert processed["mm_placeholders"]["video"][0].get_num_embeds() == 1792
    assert len(processed["prompt_token_ids"]) <= 2048
