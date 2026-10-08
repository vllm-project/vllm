# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise quant-stream tensor lifetime against the next scatter allocation."""

import pytest
import torch

from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferEngine,
)
from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo

NUMEL = 16 << 20


def _hold(stream: torch.cuda.Stream, seconds: float) -> None:
    """Keep a CUDA stream busy for approximately ``seconds``."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    probe = 10_000_000
    start.record()
    torch.cuda._sleep(probe)
    end.record()
    end.synchronize()
    cycles = int(probe * seconds * 1000 / start.elapsed_time(end))
    with torch.cuda.stream(stream):
        torch.cuda._sleep(cycles)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_layers_kernel_copy_is_not_overwritten_by_the_next_layer():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    engine = object.__new__(ShardedRDTWeightTransferEngine)
    engine.device = device
    engine._proc_stream = torch.cuda.Stream(device=device)
    engine._quant_stream = torch.cuda.Stream(device=device)
    kernel = torch.nn.Parameter(
        torch.zeros(NUMEL, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    torch.cuda.synchronize()

    # The layer as materialize_layer leaves it and the scatter fills it.
    layer = torch.nn.Module()
    with torch.cuda.stream(engine._proc_stream):
        loaded = torch.full((NUMEL,), 1.0, dtype=torch.bfloat16, device=device)
        layer.weight = torch.nn.Parameter(loaded, requires_grad=False)
        ready = torch.cuda.Event()
        ready.record(engine._proc_stream)
    loaded_ptr = loaded.data_ptr()
    del loaded
    LAYERWISE_INFO[layer] = LayerReloadingInfo(
        restore_metadata=({}, {}),
        restore_device=device,
        kernel_tensors=({"weight": kernel}, {}),
    )
    engine._proc_stream.synchronize()

    _hold(engine._quant_stream, 0.5)
    engine._run_quant([layer], ready)

    # The next layer, materialized and scattered on the process stream.
    with torch.cuda.stream(engine._proc_stream):
        following = torch.empty(NUMEL, dtype=torch.bfloat16, device=device)
        following.fill_(2.0)
    reused = following.data_ptr() == loaded_ptr
    engine._quant_stream.synchronize()
    engine._proc_stream.synchronize()

    assert layer.weight is kernel
    assert torch.equal(kernel, torch.full_like(kernel, 1.0)), (
        "the kernel tensor holds the next layer's bytes: its source was reused "
        f"before the quant stream's copy read it (memory reused: {reused})"
    )
