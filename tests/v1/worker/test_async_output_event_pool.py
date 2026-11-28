# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`AsyncOutput` with a `CopyEventPool`: no per-step event creation, and the D2H
copy still waits for the main-stream work that produced its sources.

An event pair goes back to the pool only once `get_output()` has synchronized
on it, so outputs in flight never share events.
"""

import dataclasses

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput, CopyEventPool
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def test_pool_reuses_released_pairs(monkeypatch):
    created: list[dict] = []

    class FakeEvent:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(torch, "Event", FakeEvent)
    pool = CopyEventPool()
    first, second = pool.acquire(), pool.acquire()
    assert first is not second
    assert created == [{}, {"blocking": True}] * 2
    pool.release(first)
    assert pool.acquire() is first
    assert len(created) == 4


def _sampler_output(sampled, num_sampled):
    fields = {f.name for f in dataclasses.fields(SamplerOutput)}
    kw: dict = {name: None for name in fields}
    kw["sampled_token_ids"] = sampled
    if "num_sampled" in fields:
        kw["num_sampled"] = num_sampled
    if "num_rejected" in fields:
        kw["num_rejected"] = torch.zeros_like(num_sampled)
    return SamplerOutput(**kw)


def _output(step, x, main_stream, copy_stream, pool):
    # A long kernel chain on the main stream, then the sampled ids ordered after
    # it: a copy that did not wait for the main stream would read them early.
    y = x
    for _ in range(8):
        y = y @ x * 1e-3
    sampled = torch.full((2, 1), step, dtype=torch.int64, device="cuda")
    sampled.add_((y[0, 0] * 0).to(torch.int64))
    num_sampled = torch.ones(2, dtype=torch.int32, device="cuda")
    return AsyncOutput(
        ModelRunnerOutput(req_ids=["a", "b"]),
        _sampler_output(sampled, num_sampled),
        num_sampled,
        main_stream=main_stream,
        copy_stream=copy_stream,
        check_ep_fault=False,
        pending_aux_output=None,
        event_pool=pool,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA streams")
def test_pooled_outputs_copy_after_main_stream_and_reuse_events():
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    pool = CopyEventPool()
    x = torch.randn(4096, 4096, device="cuda")

    outs = [_output(step, x, main_stream, copy_stream, pool) for step in range(3)]
    # Outputs in flight hold distinct pairs.
    assert len({id(out.copy_event) for out in outs}) == 3
    for step, out in enumerate(outs):
        assert out.get_output().sampled_token_ids == [[step], [step]]

    # Every pair is back in the pool: the next output takes one of them.
    nxt = _output(3, x, main_stream, copy_stream, pool)
    assert any(nxt.copy_event is out.copy_event for out in outs)
    assert nxt.get_output().sampled_token_ids == [[3], [3]]
    torch.accelerator.synchronize()
