# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from ...utils import build_model_context

MODEL_ID = "google/embeddinggemma-2"


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [{"max_soft_tokens": 280}, {"videos_kwargs": {"max_soft_tokens": 280}}],
)
def test_video_budget_reads_configured_max_soft_tokens(
    mm_processor_kwargs: dict[str, object],
):
    ctx = build_model_context(
        MODEL_ID,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"video": 1},
    )

    assert tokens is not None
    assert tokens["video"] == 32 * (280 + 2)


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {"max_frames": 1},
        {"videos_kwargs": {"max_frames": 1}},
        {"max_frames": None, "overflow_strategy": None},
        {"videos_kwargs": {"max_frames": None, "overflow_strategy": None}},
    ],
)
def test_video_budget_covers_loader_frames(mm_processor_kwargs: dict[str, object]):
    assert _video_budget(mm_processor_kwargs) == _video_budget(None)


def test_video_budget_covers_processor_frames():
    loader_cap = {"max_frames": 8}

    assert _video_budget(None, loader_cap) == _video_budget(None)
    assert _video_budget({"max_frames": 8}, loader_cap) < _video_budget(None)


def _video_budget(
    mm_processor_kwargs: dict[str, object] | None,
    video_media_io_kwargs: dict[str, object] | None = None,
) -> int:
    ctx = build_model_context(
        MODEL_ID,
        model_config_kwargs=(
            {"media_io_kwargs": {"video": video_media_io_kwargs}}
            if video_media_io_kwargs
            else None
        ),
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"video": 1},
    )
    assert tokens is not None
    return tokens["video"]


@pytest.mark.parametrize("configured", [True, False])
def test_video_placeholder_reads_scoped_add_timestamps(configured: bool):
    num_frames = 4
    frames = np.zeros((num_frames, 96, 96, 3), dtype=np.uint8)
    metadata = {
        "fps": 1.0,
        "duration": float(num_frames),
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": True,
    }

    def placeholder_len(mm_processor_kwargs: dict[str, object]) -> int:
        ctx = build_model_context(
            MODEL_ID,
            mm_processor_kwargs=mm_processor_kwargs if configured else None,
            limit_mm_per_prompt={"image": 0, "audio": 0, "video": 1},
        )
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        inputs = ProcessorInputs(
            prompt=processor.info.get_tokenizer().encode(
                "<|video|>", add_special_tokens=False
            ),
            mm_data_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
            hf_processor_mm_kwargs={} if configured else mm_processor_kwargs,
        )
        outputs = processor.apply(inputs, TimingContext(enabled=False))
        return outputs["mm_placeholders"]["video"][0].length

    flat = placeholder_len({"add_timestamps": True})
    scoped = placeholder_len({"videos_kwargs": {"add_timestamps": True}})

    assert flat > placeholder_len({})
    assert scoped == flat
