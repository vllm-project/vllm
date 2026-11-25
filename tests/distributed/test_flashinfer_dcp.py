# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dcp_comm_backend='flashinfer_a2a_fused': token mask, and the fused combine on
PyTorch's NCCL symmetric-memory backend next to vLLM's direct DCP A2A."""

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

from vllm.utils.flashinfer import has_flashinfer_dcp_a2a_lse_reduce
from vllm.utils.network_utils import get_file_store_init_method
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.attention.ops.flashinfer_dcp import dcp_valid_token_mask

mp.set_start_method("spawn", force=True)


def test_dcp_valid_token_mask():
    seq_lens = torch.tensor([3, 0, 2])
    query_start_loc = torch.tensor([0, 1, 3, 4])  # request 1 has two query rows
    mask = dcp_valid_token_mask(6, seq_lens, query_start_loc, torch.device("cpu"))
    assert mask.tolist() == [True, False, False, True, False, False]
    assert dcp_valid_token_mask(4, None, None, torch.device("cpu")) is None
    empty = dcp_valid_token_mask(
        3, torch.empty(0, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), "cpu"
    )
    assert not empty.any()


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    rank = int(env["LOCAL_RANK"])
    world_size = int(env["WORLD_SIZE"])
    torch.accelerator.set_device_index(rank)
    device = torch.device(f"cuda:{rank}")
    from vllm.distributed.device_communicators.symm_mem import (
        select_nccl_symm_mem_backend,
    )
    from vllm.v1.attention.ops.flashinfer_dcp import FlashInferDCPWorkspace

    select_nccl_symm_mem_backend()
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        init_method=env["DISTRIBUTED_INIT_METHOD"],
        rank=rank,
        world_size=world_size,
    )
    try:
        from vllm.v1.attention.ops.dcp import (
            DirectDCPA2AWorkspace,
            _lse_weighted_combine,
        )

        group = dist.group.WORLD
        # At 4 GPUs heads == world_size, where FlashInfer needs layout= to
        # tell destination-major from head-major.
        heads, head_dim, max_tokens = 4, 512, 16
        dtype = torch.bfloat16
        # vLLM's direct A2A workspace on the NCCL backend, allocated first as
        # in a real worker, then FlashInfer's.
        direct = DirectDCPA2AWorkspace(
            group, device, max_tokens, heads, head_dim, dtype
        )
        fi = FlashInferDCPWorkspace(group, device, max_tokens, heads, head_dim, dtype)

        rows = 6
        gen = torch.Generator(device=device).manual_seed(1234 + rank)
        out = torch.randn(
            rows,
            world_size * heads,
            head_dim,
            dtype=dtype,
            device=device,
            generator=gen,
        )
        lse = torch.randn(
            rows, world_size * heads, dtype=torch.float32, device=device, generator=gen
        )
        # Request 2 has no local KV on even ranks; row 5 is graph padding.
        seq_lens = torch.tensor(
            [5, 7, 0 if rank % 2 == 0 else 3, 9, 4], dtype=torch.int32, device=device
        )
        query_start_loc = torch.arange(6, dtype=torch.int32, device=device)
        # Undefined values in rows that must not be read.
        if rank % 2 == 0:
            out[2] = float("nan")
            lse[2] = 123.0

        for is_base_e in (True, False):
            actual_fi = fi.lse_reduce(out, lse, is_base_e, seq_lens, query_start_loc)
            actual_direct = direct.lse_reduce(
                out, lse, is_base_e, seq_lens, query_start_loc
            )
            all_out = [torch.empty_like(out) for _ in range(world_size)]
            all_lse = [torch.empty_like(lse) for _ in range(world_size)]
            dist.all_gather(all_out, out)
            dist.all_gather(all_lse, lse)
            sl = slice(rank * heads, (rank + 1) * heads)
            outputs = torch.stack([t[:, sl] for t in all_out]).float()
            lses = torch.stack([t[:, sl] for t in all_lse]).clone()
            lses[0::2, 2] = float("-inf")
            outputs[0::2, 2] = 0.0
            expected, _ = _lse_weighted_combine(
                outputs, lses, return_lse=True, is_lse_base_on_e=is_base_e
            )
            torch.testing.assert_close(
                actual_fi[:5].float(), expected[:5], rtol=3e-2, atol=3e-2
            )
            torch.testing.assert_close(
                actual_direct[:5].float(), expected[:5], rtol=3e-2, atol=3e-2
            )

        # As in vLLM: eager warmup, then capture and replay on another stream.
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            fi.lse_reduce(out, lse, False, seq_lens, query_start_loc)
            with torch.cuda.graph(graph, stream=side):
                captured = fi.lse_reduce(out, lse, False, seq_lens, query_start_loc)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            captured[:5].float(), expected[:5], rtol=3e-2, atol=3e-2
        )

        # Metadata updated in place between replays must reach the captured
        # mask: request 1 loses its local KV on odd ranks, whose rows go stale.
        if rank % 2 == 1:
            seq_lens[1] = 0
            out[1] = float("nan")
        graph.replay()
        torch.accelerator.synchronize()
        lses[1::2, 1] = float("-inf")
        outputs[1::2, 1] = 0.0
        expected, _ = _lse_weighted_combine(
            outputs, lses, return_lse=True, is_lse_base_on_e=False
        )
        torch.testing.assert_close(
            captured[:5].float(), expected[:5], rtol=3e-2, atol=3e-2
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
@pytest.mark.skipif(
    not has_flashinfer_dcp_a2a_lse_reduce(),
    reason="Needs FlashInfer >= 0.7.6.",
)
@pytest.mark.parametrize("world_size", [4, 8])
def test_flashinfer_combine_on_nccl_symm_mem(world_size: int):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs.")
    init_method = get_file_store_init_method()
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "DISTRIBUTED_INIT_METHOD": init_method,
        }
        p = mp.Process(target=_worker, args=(env,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join(timeout=600)
    for p in processes:
        if p.is_alive():
            p.kill()
            p.join()
        assert p.exitcode == 0
