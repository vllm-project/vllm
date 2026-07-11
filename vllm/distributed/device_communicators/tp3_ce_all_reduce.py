# SPDX-License-Identifier: Apache-2.0
"""Experimental TP3 int8 reduction through pinned shared host memory."""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.triton_utils import tl, triton


_WORKSPACE: dict[str, object] | None = None


@triton.jit
def _quantize_i8_block(
    x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = tl.arange(0, block)
    cols = block_id * block + offs
    mask = cols < n_cols
    values = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 127.0, 1.0e-8)
    quantized = tl.floor(values / scale + 0.5)
    quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0) + 128.0
    tl.store(q_ptr + row * n_cols + cols, quantized.to(tl.uint8), mask=mask)
    n_blocks = tl.cdiv(n_cols, block)
    tl.store(s_ptr + row * n_blocks + block_id, scale)


@triton.jit
def _dequant_sum_i8_block(
    q0_ptr, q1_ptr, q2_ptr, s0_ptr, s1_ptr, s2_ptr, out_ptr,
    n_cols: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = tl.arange(0, block)
    cols = block_id * block + offs
    mask = cols < n_cols
    q0 = tl.load(q0_ptr + row * n_cols + cols, mask=mask, other=128).to(tl.float32)
    q1 = tl.load(q1_ptr + row * n_cols + cols, mask=mask, other=128).to(tl.float32)
    q2 = tl.load(q2_ptr + row * n_cols + cols, mask=mask, other=128).to(tl.float32)
    n_blocks = tl.cdiv(n_cols, block)
    scale_idx = row * n_blocks + block_id
    values = ((q0 - 128.0) * tl.load(s0_ptr + scale_idx)
              + (q1 - 128.0) * tl.load(s1_ptr + scale_idx)
              + (q2 - 128.0) * tl.load(s2_ptr + scale_idx))
    tl.store(out_ptr + row * n_cols + cols, values, mask=mask)


def _wait_flags(mapping: mmap.mmap, offset: int, world_size: int, generation: int) -> None:
    deadline = time.monotonic() + float(os.environ.get("VLLM_TP3_CE_TIMEOUT_S", "120"))
    while True:
        if all(struct.unpack_from("i", mapping, offset + rank * 4)[0] == generation
               for rank in range(world_size)):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"TP3 CE barrier timed out at generation {generation}")
        time.sleep(0)


def _workspace(group: ProcessGroup, device: torch.device, cols: int) -> dict[str, object]:
    global _WORKSPACE
    if _WORKSPACE is not None:
        return _WORKSPACE
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    max_rows = int(os.environ.get("VLLM_TP3_CE_MAX_ROWS", "12288"))
    quant_block = int(os.environ.get("VLLM_TP3_CE_QUANT_BLOCK", "8192"))
    num_blocks = (cols + quant_block - 1) // quant_block
    max_payload = max_rows * (cols + num_blocks * 4)
    header_bytes = 4096
    total_bytes = header_bytes + world_size * max_payload
    path = Path(os.environ.get("VLLM_TP3_CE_SHM_PATH", "/dev/shm/vllm-tp3-ce"))
    if rank == 0:
        with path.open("w+b") as handle:
            handle.truncate(total_bytes)
    dist.barrier(group=group)
    handle = path.open("r+b", buffering=0)
    mapping = mmap.mmap(handle.fileno(), total_bytes)
    host_storage = torch.frombuffer(mapping, dtype=torch.uint8)
    status = torch.cuda.cudart().cudaHostRegister(host_storage.data_ptr(), host_storage.numel(), 1)
    if status.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {status}")
    _WORKSPACE = {
        "handle": handle,
        "mapping": mapping,
        "host_storage": host_storage,
        "host": host_storage[header_bytes:].view(world_size, max_payload),
        "max_rows": max_rows,
        "quant_block": quant_block,
        "num_blocks": num_blocks,
        "generation": 0,
        "send": torch.empty(max_payload, device=device, dtype=torch.uint8),
        "remote": torch.empty((world_size - 1, max_payload), device=device, dtype=torch.uint8),
        "stream": torch.cuda.Stream(device=device),
    }
    return _WORKSPACE


def tp3_ce_all_reduce(input_: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world_size != 3 or input_.dim() != 2 or input_.shape[1] != 5120:
        raise ValueError(f"unsupported TP3 CE shape={tuple(input_.shape)}")
    rows, cols = input_.shape
    workspace = _workspace(group, input_.device, cols)
    if rows > int(workspace["max_rows"]):
        raise ValueError(f"rows={rows} exceed CE workspace")
    quant_block = int(workspace["quant_block"])
    num_blocks = int(workspace["num_blocks"])
    q_bytes = rows * cols
    scale_bytes = rows * num_blocks * 4
    total = q_bytes + scale_bytes
    send = workspace["send"]
    remote = workspace["remote"]
    host = workspace["host"]
    mapping = workspace["mapping"]
    stream = workspace["stream"]
    assert isinstance(send, torch.Tensor)
    assert isinstance(remote, torch.Tensor)
    assert isinstance(host, torch.Tensor)
    assert isinstance(mapping, mmap.mmap)
    assert isinstance(stream, torch.cuda.Stream)

    q_local = send[:q_bytes].view(rows, cols)
    scale_local = send[q_bytes:total].view(torch.float32).view(rows, num_blocks)
    _quantize_i8_block[(rows, num_blocks)](
        input_, q_local, scale_local, cols, quant_block
    )
    workspace["generation"] = int(workspace["generation"]) + 1
    generation = int(workspace["generation"])
    current = torch.cuda.current_stream(input_.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        host[rank, :total].copy_(send[:total], non_blocking=True)
    stream.synchronize()
    struct.pack_into("i", mapping, rank * 4, generation)
    _wait_flags(mapping, 0, world_size, generation)
    peers = [peer for peer in range(world_size) if peer != rank]
    with torch.cuda.stream(stream):
        for slot, peer in enumerate(peers):
            remote[slot, :total].copy_(host[peer, :total], non_blocking=True)
    stream.synchronize()
    struct.pack_into("i", mapping, 64 + rank * 4, generation)
    _wait_flags(mapping, 64, world_size, generation)
    current.wait_stream(stream)
    buffers = [send[:total] if peer == rank else remote[peers.index(peer), :total]
               for peer in range(world_size)]
    output = torch.empty_like(input_)
    _dequant_sum_i8_block[(rows, num_blocks)](
        buffers[0][:q_bytes], buffers[1][:q_bytes], buffers[2][:q_bytes],
        buffers[0][q_bytes:].view(torch.float32),
        buffers[1][q_bytes:].view(torch.float32),
        buffers[2][q_bytes:].view(torch.float32), output, cols, quant_block,
    )
    return output
