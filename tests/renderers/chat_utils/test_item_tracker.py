# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import MagicMock

import pytest

from vllm.config import ModelConfig
from vllm.renderers.chat_utils import (
    AsyncMultiModalItemTracker,
    MultiModalItemTracker,
)


@pytest.mark.asyncio
async def test_resolve_items_runs_modalities_concurrently_and_preserves_order():
    """Media fetches overlap while modality and item order are preserved."""
    active_fetches = 0
    max_active_fetches = 0

    async def _fetch(name: str, delay: float):
        nonlocal active_fetches, max_active_fetches
        active_fetches += 1
        max_active_fetches = max(max_active_fetches, active_fetches)
        try:
            await asyncio.sleep(delay)
            return name, f"{name}-uuid"
        finally:
            active_fetches -= 1

    tracker = AsyncMultiModalItemTracker(MagicMock(is_multimodal_model=True))
    tracker.__dict__["mm_processor"] = MagicMock()
    tracker._items_by_modality["video"] = [
        lambda: _fetch("video-0", 0.02),
        lambda: _fetch("video-1", 0.01),
    ]
    tracker._items_by_modality["image"] = [
        lambda: _fetch("image-0", 0.02),
        lambda: _fetch("image-1", 0.01),
    ]
    tracker._items_by_modality["audio"] = [
        lambda: _fetch("audio-0", 0.02),
        lambda: _fetch("audio-1", 0.01),
    ]

    mm_data, mm_uuids = await tracker.resolve_items()

    assert mm_data is not None
    assert mm_uuids is not None
    assert max_active_fetches == 6
    assert list(mm_data) == ["image", "audio", "video"]
    assert list(mm_uuids) == ["image", "audio", "video"]
    assert mm_data == {
        "image": ["image-0", "image-1"],
        "audio": ["audio-0", "audio-1"],
        "video": ["video-0", "video-1"],
    }
    assert mm_uuids == {
        "image": ["image-0-uuid", "image-1-uuid"],
        "audio": ["audio-0-uuid", "audio-1-uuid"],
        "video": ["video-0-uuid", "video-1-uuid"],
    }


@pytest.mark.asyncio
async def test_resolve_items_does_not_leak_tasks_on_partial_failure():
    """Regression test: one failing media fetch must not abandon the other
    still-in-flight fetches across modality batches.

    Before the fix, `resolve_items` gathered per-modality fetches with plain
    `asyncio.gather`, so the first exception propagated immediately while
    sibling fetches (real network/thread-pool work in production) kept
    running detached, with nothing left to await or cancel them.
    """

    async def _fetch(should_fail: bool, delay: float):
        if should_fail:
            await asyncio.sleep(0.01)
            raise ValueError("simulated fetch failure")
        await asyncio.sleep(delay)
        return ("decoded", None)

    tracker = AsyncMultiModalItemTracker(MagicMock())
    tracker._items_by_modality["image"] = [
        lambda: _fetch(True, 0),
        lambda: _fetch(False, 0.2),
        lambda: _fetch(False, 0.2),
    ]
    tracker._items_by_modality["audio"] = [lambda: _fetch(False, 0.2)]
    tracker._items_by_modality["video"] = [lambda: _fetch(False, 0.2)]

    tasks_before = asyncio.all_tasks()
    with pytest.raises(ValueError, match="simulated fetch failure"):
        await tracker.resolve_items()

    leaked_tasks = asyncio.all_tasks() - tasks_before
    assert not leaked_tasks, (
        f"resolve_items left {len(leaked_tasks)} task(s) running after "
        f"raising: {leaked_tasks}"
    )


MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


@pytest.fixture(scope="function")
def mistral_model_config():
    return ModelConfig(
        MISTRAL_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={
            "image": 2,
        },
    )


@pytest.fixture(scope="function")
def phi3v_model_config():
    return ModelConfig(
        PHI3V_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={
            "image": 2,
        },
    )


@pytest.mark.parametrize(
    "tracker_cls",
    [MultiModalItemTracker, AsyncMultiModalItemTracker],
)
def test_tracker_media_io_kwargs_mistral_disables_image_orientation(
    mistral_model_config,
    tracker_cls,
):
    """The Mistral chat template re-decodes images from their source bytes
    without EXIF normalization; vLLM's decode must match it or the image
    token grids computed from the two decodes disagree."""
    tracker = tracker_cls(mistral_model_config)

    assert tracker.uses_mistral_chat_template
    assert tracker.media_io_kwargs == {"image": {"normalize_orientation": False}}


@pytest.mark.parametrize(
    "tracker_cls",
    [MultiModalItemTracker, AsyncMultiModalItemTracker],
)
def test_tracker_media_io_kwargs_non_mistral_keeps_default(
    phi3v_model_config,
    tracker_cls,
):
    tracker = tracker_cls(phi3v_model_config)

    assert not tracker.uses_mistral_chat_template
    assert (
        (tracker.media_io_kwargs or {})
        .get("image", {})
        .get("normalize_orientation", True)
    )
