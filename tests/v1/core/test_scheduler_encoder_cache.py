# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.sched.scheduler import Scheduler

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 768
IMAGE_EMBEDS = 500
TEXT_GAP = 10
IMAGE2_OFFSET = IMAGE_EMBEDS + TEXT_GAP
TOTAL_TOKENS = IMAGE2_OFFSET + IMAGE_EMBEDS
ENCODER_CACHE_SIZE = 600


def _make_mamba_split_scheduler(
    *,
    block_size: int = BLOCK_SIZE,
    encoder_cache_size: int = ENCODER_CACHE_SIZE,
):
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=8192,
        block_size=block_size,
        enable_chunked_prefill=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.cache_config.mamba_cache_mode = "align"
    manager = scheduler.encoder_cache_manager
    manager.cache_size = encoder_cache_size
    manager.num_free_slots = encoder_cache_size
    manager.num_freeable_slots = encoder_cache_size
    return scheduler


def _make_two_image_request() -> list:
    mm_positions = [
        [
            PlaceholderRange(offset=0, length=IMAGE_EMBEDS),
            PlaceholderRange(offset=IMAGE2_OFFSET, length=IMAGE_EMBEDS),
        ]
    ]
    return create_requests(
        num_requests=1,
        num_tokens=TOTAL_TOKENS,
        mm_positions=mm_positions,
        block_size=BLOCK_SIZE,
    )


def test_mamba_block_aligned_split_preserves_sub_block_tokens():
    """Sub-block chunks must not collapse to zero."""
    request = _make_two_image_request()[0]
    request.num_computed_tokens = 0

    mock = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        use_eagle=False,
    )
    num_new_tokens = Scheduler._mamba_block_aligned_split(
        self=mock,
        request=request,
        num_new_tokens=IMAGE2_OFFSET,
    )
    assert num_new_tokens == IMAGE2_OFFSET

    long_request = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE * 3,
        block_size=BLOCK_SIZE,
    )[0]
    long_request.num_computed_tokens = 0
    num_new_tokens = Scheduler._mamba_block_aligned_split(
        self=mock,
        request=long_request,
        num_new_tokens=BLOCK_SIZE * 2,
    )
    assert num_new_tokens == BLOCK_SIZE * 2


def test_two_images_make_progress_with_encoder_cache_contention():
    """Regression for issue #47738: two large images must not deadlock."""
    scheduler = _make_mamba_split_scheduler()
    (request,) = _make_two_image_request()
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    scheduled_tokens = output.num_scheduled_tokens[request.request_id]
    assert 0 < scheduled_tokens < BLOCK_SIZE
    assert scheduled_tokens == IMAGE2_OFFSET
    assert output.scheduled_encoder_inputs[request.request_id] == [0]
    assert request.num_computed_tokens == IMAGE2_OFFSET

    scheduler._free_encoder_inputs(request)

    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert output.scheduled_encoder_inputs[request.request_id] == [1]
