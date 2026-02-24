#!/usr/bin/env python3
"""
Minimal 2-rank P2P send/recv using vLLM PyNcclCommunicator with CUDA graph capture and replay.

Run (2 GPUs on one machine):
  cd /path/to/vllm && torchrun --nproc_per_node=2 examples/pynccl_p2p_cudagraph.py

Or with 2 nodes (1 GPU each), set MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE accordingly.
"""

import os
import sys

# Allow importing vllm from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This script expects exactly 2 ranks."

    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # PyNcclCommunicator requires a non-NCCL group (Gloo used for bootstrap)
    print("jcz haha 0.1")
    comm = PyNcclCommunicator(group=_get_default_group(), device=device)
    print("jcz haha 0.2")
    if comm.disabled:
        print(f"Rank {rank}: PyNcclCommunicator disabled, exit.")
        return

    # Fixed buffer shape for capture/replay (must be same on both ranks): 64*2048, bfloat16
    shape = (64, 2048)
    dtype = torch.bfloat16

    if rank == 0:
        send_buf = torch.zeros(shape, dtype=dtype, device=device)
        recv_buf = None
    else:
        send_buf = None
        recv_buf = torch.empty(shape, dtype=dtype, device=device)

    # Dedicated stream for capture and replay
    capture_stream = torch.cuda.Stream(device=device)
    print("jcz haha 1")
    # Warmup with random data
    torch.manual_seed(42)
    if rank == 0:
        send_buf.copy_(torch.randn(*shape, dtype=dtype, device=device))
        comm.send(send_buf, dst=1, stream=torch.cuda.current_stream(device))
    else:
        comm.recv(recv_buf, src=0, stream=torch.cuda.current_stream(device))
    print("jcz haha 2")
    torch.cuda.synchronize(device)
    print("jcz haha 3")

    dist.barrier()
    if rank == 1:
        torch.manual_seed(42)
        expected = torch.randn(*shape, dtype=dtype, device=device)
        assert torch.allclose(recv_buf, expected), "Warmup recv mismatch"
    print(f"Rank {rank}: warmup OK")

    # Capture CUDA graph (same buffer addresses used on every replay)
    graph = torch.cuda.CUDAGraph()
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=capture_stream):
        if rank == 0:
            comm.send(send_buf, dst=1, stream=capture_stream)
        else:
            comm.recv(recv_buf, src=0, stream=capture_stream)
    print(f"Rank {rank}: captured graph")

    torch.cuda.synchronize(device)
    dist.barrier()
    print("Both ranks: capture done")

    # Replay several times with random data (same seed per step for verification)
    num_replays = 4
    for step in range(num_replays):
        if rank == 0:
            torch.manual_seed(100 + step)
            send_buf.copy_(torch.randn(*shape, dtype=dtype, device=device))
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            graph.replay()
        else:
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            graph.replay()

        torch.cuda.synchronize(device)
        dist.barrier()

        if rank == 1:
            torch.manual_seed(100 + step)
            expected = torch.randn(*shape, dtype=dtype, device=device)
            ok = torch.allclose(recv_buf, expected)
            print(f"Rank 1: replay step {step} recv_buf[0,0]={recv_buf[0,0].item():.4f} (expected {expected[0,0].item():.4f}) OK={ok}")
            assert ok, f"Replay step {step} recv mismatch"

    print(f"Rank {rank}: all {num_replays} replays OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
