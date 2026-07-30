# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _peer_access_available(world_size: int) -> bool:
    if not current_platform.is_cuda() or torch.accelerator.device_count() < world_size:
        return False
    return all(
        source == destination or torch.cuda.can_device_access_peer(source, destination)
        for source in range(world_size)
        for destination in range(world_size)
    )


def _shared_ep_mxfp8_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from vllm.model_executor.layers.fused_moe.shared_ep import SharedEPMemory
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        max_tokens = 4
        num_tokens = rank + 1
        hidden_size = 512
        top_k = 2
        memory = SharedEPMemory.create(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            quant_dtype="mxfp8",
            group=dist.group.WORLD,
            device=torch.device("cuda", rank),
        )
        generator = torch.Generator(device=f"cuda:{rank}")
        generator.manual_seed(1000 + rank)
        hidden_states = torch.randn(
            max_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
            generator=generator,
        )
        route_ids = torch.arange(
            max_tokens * top_k,
            dtype=torch.int32,
            device=f"cuda:{rank}",
        ).view(max_tokens, top_k)
        route_ids.add_(rank * max_tokens * top_k)
        route_weights = torch.full(
            (max_tokens, top_k),
            rank + 0.25,
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )
        memory.publish_input(
            hidden_states[:num_tokens],
            route_ids[:num_tokens],
            route_weights[:num_tokens],
        )
        gathered_q, gathered_s, gathered_ids, gathered_weights = (
            memory.gather_mxfp8_inputs()
        )
        for owner in range(world_size):
            owner_tokens = owner + 1
            expected_generator = torch.Generator(device=f"cuda:{rank}")
            expected_generator.manual_seed(1000 + owner)
            expected_hidden = torch.randn(
                max_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=f"cuda:{rank}",
                generator=expected_generator,
            )
            expected_q, expected_s = mxfp8_e4m3_quantize(
                expected_hidden[:owner_tokens],
                is_sf_swizzled_layout=False,
                alignment=256,
            )
            owner_start = owner * max_tokens
            owner_rows = slice(owner_start, owner_start + owner_tokens)
            assert torch.equal(gathered_q[owner_rows], expected_q)
            assert torch.equal(gathered_s[owner_rows], expected_s)
            expected_ids = torch.arange(
                max_tokens * top_k,
                dtype=torch.int32,
                device=f"cuda:{rank}",
            ).view(max_tokens, top_k)
            expected_ids.add_(owner * max_tokens * top_k)
            assert torch.equal(gathered_ids[owner_rows], expected_ids[:owner_tokens])
            assert torch.all(gathered_weights[owner_rows] == owner + 0.25)
            padded_rows = slice(owner_start + owner_tokens, owner_start + max_tokens)
            assert torch.all(gathered_ids[padded_rows] == -1)
            assert torch.all(gathered_weights[padded_rows] == 0)

        global_rows = world_size * max_tokens
        global_row_ids = torch.arange(
            global_rows,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        ).view(global_rows, 1)
        partial_output = (
            (global_row_ids + (rank + 1) * 100)
            .expand(global_rows, hidden_size)
            .contiguous()
        )
        memory.publish_partial_output(partial_output)
        reduced_output = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        memory.reduce_output(reduced_output, num_tokens)
        local_rows = torch.arange(
            rank * max_tokens,
            rank * max_tokens + num_tokens,
            dtype=torch.float32,
            device=f"cuda:{rank}",
        ).view(num_tokens, 1)
        expected_output = torch.zeros_like(local_rows)
        for source_rank in range(world_size):
            expected_output += (local_rows + (source_rank + 1) * 100).to(torch.bfloat16)
        expected_output = expected_output.to(torch.bfloat16)
        assert torch.equal(
            reduced_output,
            expected_output.expand(num_tokens, hidden_size),
        )

        dist.barrier()
        memory.close()
    finally:
        dist.destroy_process_group()


def test_shared_ep_mxfp8_peer_gather_scatter() -> None:
    world_size = 4
    if not _peer_access_available(
        world_size
    ) or not current_platform.is_device_capability_family(100):
        pytest.skip("MXFP8 SharedEP requires four peer-accessible SM100 GPUs")
    mp.spawn(
        _shared_ep_mxfp8_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


def _shared_ep_direct_output_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from vllm.model_executor.layers.fused_moe.shared_ep import SharedEPMemory

        max_tokens = 4
        num_tokens = rank + 1
        hidden_size = 512
        top_k = 8
        memory = SharedEPMemory.create(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            top_k=top_k,
            quant_dtype="nvfp4",
            group=dist.group.WORLD,
            device=torch.device("cuda", rank),
        )
        peer_view = memory._direct_output_peer_view
        assert peer_view is not None
        assert peer_view.global_view is not None

        # Each expert rank writes a disjoint subset of canonical
        # (owner, token, top-k slot) rows, exactly as the W2 epilogue does.
        for owner in range(world_size):
            for token in range(max_tokens):
                for slot in range(rank, top_k, world_size):
                    physical_row = (
                        owner * peer_view.rows_per_rank + token * top_k + slot
                    )
                    peer_view.global_view[physical_row].fill_(rank + 1)
        memory.publish_output()

        output = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=f"cuda:{rank}",
        )
        memory.reduce_direct_output(output, num_tokens)
        expected = sum(slot % world_size + 1 for slot in range(top_k))
        assert torch.all(output == expected)

        dist.barrier()
        memory.close()
    finally:
        dist.destroy_process_group()


def test_shared_ep_nvfp4_direct_output_slots() -> None:
    world_size = 4
    if not _peer_access_available(
        world_size
    ) or not current_platform.is_device_capability_family(100):
        pytest.skip("NVFP4 SharedEP requires four peer-accessible SM100 GPUs")
    mp.spawn(
        _shared_ep_direct_output_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )
