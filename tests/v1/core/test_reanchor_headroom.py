# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup guard for --enable-unbounded-streaming: re-anchoring drops whole
scheduler blocks past the sliding window, so max_model_len - margin -
sliding_window must be at least one block."""

import pytest
import torch

from vllm.v1.kv_cache_interface import SlidingWindowSpec

from .utils import create_scheduler

BLOCK_SIZE = 128
SLIDING_WINDOW = 256
MAX_MODEL_LEN = 1024


def _scheduler(margin: int):
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )
    scheduler.scheduler_config.enable_unbounded_streaming = True
    scheduler.scheduler_config.reanchor_margin_tokens = margin
    return scheduler


# 1024 - 640 - 256 = 128: exactly one block.
@pytest.mark.parametrize("margin", [512, 640])
def test_reanchor_headroom_ok(margin):
    _scheduler(margin)._check_reanchor_headroom()


@pytest.mark.parametrize("margin", [641, 700])
def test_reanchor_headroom_too_small(margin):
    with pytest.raises(ValueError, match="re-anchoring drops whole blocks"):
        _scheduler(margin)._check_reanchor_headroom()
