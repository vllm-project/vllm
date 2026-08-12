# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: custom AR graph capture under expandable_segments.

Reproduces the failure mode from https://github.com/vllm-project/vllm/issues/42609
where pre-capture VMM activations are not legacy-IPC exportable.

Requires 2 GPUs with P2P. Run with:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc_per_node=2 \\
    tests/distributed/test_custom_all_reduce_expandable_segments.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

# Must be set before CUDA allocator first use in this process.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_SKIP_P2P_CHECK", "1")

from vllm.distributed.device_communicators.custom_all_reduce import (  # noqa: E402
    CustomAllreduce,
    _tensor_is_legacy_ipc_capable,
)


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "this regression test expects nproc_per_node=2"

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Allocate under expandable_segments *before* capture (the residual hole).
    n = 4096
    x = torch.full((n,), float(rank + 1), device=device, dtype=torch.float32)
    # Best-effort: on expandable_segments this is often not legacy-IPC-capable.
    # The fix must still succeed either way.
    print(
        f"rank{rank}: pre-capture legacy_ipc={_tensor_is_legacy_ipc_capable(x)}",
        flush=True,
    )

    ca = CustomAllreduce(group=dist.group.WORLD, device=device, max_size=1 << 20)
    assert not ca.disabled, f"rank{rank}: custom AR disabled at init"

    # Eager path first.
    out_e = ca.custom_all_reduce(x)
    assert out_e is not None
    torch.cuda.synchronize()
    expected = float(sum(range(1, world_size + 1)))
    err_e = (out_e - expected).abs().max().item()
    assert err_e < 1e-5, f"rank{rank}: eager err={err_e}"

    # Graph capture with pre-allocated input (must not crash at IPC meta).
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    static_in = x.clone()
    static_out = torch.empty_like(static_in)
    with torch.cuda.stream(s):
        with ca.capture():
            g.capture_begin()
            y = ca.custom_all_reduce(static_in)
            assert y is not None
            static_out.copy_(y)
            g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    assert not ca.disabled, f"rank{rank}: custom AR disabled after capture"
    print(f"rank{rank}: graph capture+register OK", flush=True)

    static_in.fill_(float(rank + 1))
    g.replay()
    torch.cuda.synchronize()
    err_g = (static_out - expected).abs().max().item()
    assert err_g < 1e-5, f"rank{rank}: graph replay err={err_g}"
    print(f"rank{rank}: graph replay OK err={err_g}", flush=True)

    ca.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
