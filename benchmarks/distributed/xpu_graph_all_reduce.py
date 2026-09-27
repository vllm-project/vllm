#!/usr/bin/env python3
"""Minimal XPU graph replay check for an xccl all-reduce."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")
    dist.init_process_group("xccl", device_id=device)

    value = torch.empty(4, dtype=torch.float32, device=device)
    capture_stream = torch.xpu.Stream()
    capture_stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(capture_stream):
        for _ in range(2):
            torch.xpu.synchronize()
            dist.barrier()
            value.fill_(local_rank + 1)
            dist.all_reduce(value)

        value.fill_(local_rank + 1)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=capture_stream):
            dist.all_reduce(value)

    expected = sum(range(1, world_size + 1))
    for replay in range(3):
        offset = replay + 1
        value.fill_(local_rank + 1 + offset)
        graph.replay()
        torch.xpu.synchronize()
        actual = value.cpu()
        replay_expected = expected + offset * world_size
        if not torch.all(actual == replay_expected):
            raise AssertionError(
                f"rank={local_rank} replay={replay}: "
                f"expected={replay_expected}, actual={actual.tolist()}"
            )
        dist.barrier()

    if local_rank == 0:
        print("XPU graph all-reduce replay passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
