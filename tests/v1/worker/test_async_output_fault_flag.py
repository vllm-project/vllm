# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`AsyncOutput` must keep the EP fault flag alive until it has been copied.

The flag is computed on the main stream and copied on the copy stream after
`__init__` returns. If the GPU tensor were dropped at that point, the caching
allocator would hand its block back to the main stream, where the next kernel
could overwrite it before the copy ran: a spurious "fault detected" with an
all-zero rank mask.
"""

import dataclasses
import weakref

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA streams"
)


class FakeManager:
    """query_fault(): a long kernel chain on the current stream, then a flag
    that reads False. Keeps no reference to the flag."""

    def __init__(self):
        self.x = torch.randn(4096, 4096, device="cuda")

    def query_fault(self):
        y = self.x
        for _ in range(16):
            y = y @ self.x * 1e-3
        flag = torch.zeros((), dtype=torch.bool, device="cuda")
        flag.add_(y[0, 0].isnan())  # ordered after the chain, stays False
        return flag

    def query_active_mask(self):
        return torch.zeros(2, dtype=torch.int32, device="cuda")


def _sampler_output(sampled, num_sampled):
    fields = {f.name for f in dataclasses.fields(SamplerOutput)}
    kw: dict = {name: None for name in fields}
    kw["sampled_token_ids"] = sampled
    if "num_sampled" in fields:
        kw["num_sampled"] = num_sampled
    if "num_rejected" in fields:
        kw["num_rejected"] = torch.zeros_like(num_sampled)
    return SamplerOutput(**kw)


def test_fault_flag_is_retained_until_copied(monkeypatch):
    mgr = FakeManager()
    monkeypatch.setattr(async_utils, "get_ep_all2all_manager", lambda: mgr)
    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    returned: list[weakref.ref] = []
    orig = mgr.query_fault

    def spy():
        flag = orig()
        returned.append(weakref.ref(flag))
        return flag

    monkeypatch.setattr(mgr, "query_fault", spy)

    sampled = torch.tensor([[5], [7]], dtype=torch.int64, device="cuda")
    num_sampled = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    out = AsyncOutput(
        ModelRunnerOutput(req_ids=["a", "b"]),
        _sampler_output(sampled, num_sampled),
        num_sampled,
        main_stream=main_stream,
        copy_stream=copy_stream,
        check_ep_fault=True,
        pending_aux_output=None,
    )
    # The GPU flag is still referenced while the copy stream may read it.
    assert returned[0]() is not None
    # Anything the main stream allocates now must not alias the flag: fill
    # same-sized blocks with True while the copy is still queued.
    for _ in range(64):
        torch.full((), True, dtype=torch.bool, device="cuda")
    out.get_output()  # no RuntimeError("Fault detected ...")
    torch.accelerator.synchronize()
