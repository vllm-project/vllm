# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`AsyncOutput.get_output` must not return before the drafter's work when it
carries a `drafter_done` event.

A KV consumer's scheduler returns a request's blocks to the pool once it has the
step's output, and a KV connector may write those blocks right away, off the
compute stream. The output copy is started before the drafter runs, so without
the extra event the output can be ready while the drafter still writes KV.
"""

import dataclasses

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA streams"
)


def _sampler_output(sampled, num_sampled):
    fields = {f.name for f in dataclasses.fields(SamplerOutput)}
    kw: dict = {name: None for name in fields}
    kw["sampled_token_ids"] = sampled
    if "num_sampled" in fields:
        kw["num_sampled"] = num_sampled
    if "num_rejected" in fields:
        kw["num_rejected"] = torch.zeros_like(num_sampled)
    return SamplerOutput(**kw)


def _async_output():
    sampled = torch.tensor([[5], [7]], dtype=torch.int64, device="cuda")
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    return AsyncOutput(
        ModelRunnerOutput(req_ids=["a", "b"]),
        _sampler_output(sampled, num_sampled),
        num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        check_ep_fault=False,
        pending_aux_output=None,
    )


def _drafter_stand_in() -> None:
    """Long work on the main stream, enqueued after the output copy."""
    x = torch.randn(4096, 4096, device="cuda")
    y = x
    for _ in range(64):
        y = y @ x * 1e-3


def test_get_output_waits_for_drafter_done():
    out = _async_output()  # the output copy is enqueued here
    _drafter_stand_in()
    out.drafter_done = torch.Event(blocking=True)
    out.drafter_done.record(torch.cuda.current_stream())

    result = out.get_output()

    # Checked before anything else synchronizes: the drafter has finished.
    assert out.drafter_done.query()
    assert result.sampled_token_ids == [[5], [7]]


def test_get_output_without_drafter_done_is_unchanged():
    out = _async_output()
    assert out.drafter_done is None
    assert out.get_output().sampled_token_ids == [[5], [7]]
