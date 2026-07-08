# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest


def test_mamba_align_chunk_split_uses_dcp_effective_block_size():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=1000,
        num_tokens=1000,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        block_size=16,
        dcp_world_size=3,
        use_eagle=False,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=128,
    )

    assert adjusted == 96
    assert adjusted % (scheduler.cache_config.block_size * scheduler.dcp_world_size) == 0


def test_mamba_align_split_keeps_small_chunks_when_dcp_alignment_exceeds_budget():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=835,
        num_tokens=835,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        block_size=208,
        dcp_world_size=3,
        use_eagle=False,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=256,
    )

    assert adjusted == 256
