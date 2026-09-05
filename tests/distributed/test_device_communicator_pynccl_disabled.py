# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group, set_custom_all_reduce

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)


@ray.remote(num_gpus=1, max_calls=1)
def fallback_collectives_gloo(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_DISABLE_PYNCCL", "1")
        set_custom_all_reduce(False)

        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        tp_group = get_tp_group()
        comm = tp_group.device_communicator
        assert comm is not None
        assert comm.pynccl_comm is None

        # Test tensor_model_parallel_all_reduce over Gloo fallback
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            inp = torch.full((1024,), float(rank + 1), dtype=dtype, device=device)
            out = tensor_model_parallel_all_reduce(inp)
            expected_sum = sum(range(1, tp_size + 1))
            torch.testing.assert_close(
                out,
                torch.full((1024,), float(expected_sum), dtype=dtype, device=device),
            )

        # Test all_gather over Gloo fallback
        ag_inp = torch.full((4,), float(rank + 1), dtype=torch.float32, device=device)
        ag_out = comm.all_gather(ag_inp, dim=0)
        expected_ag = torch.cat(
            [
                torch.full((4,), float(r + 1), dtype=torch.float32, device=device)
                for r in range(tp_size)
            ],
            dim=0,
        )
        torch.testing.assert_close(ag_out, expected_ag)

        # Test all_gatherv with variable sizes and batched list of tensors
        sizes = [2 + r * 2 for r in range(tp_size)]  # e.g. [2, 4] for tp_size=2
        local_size = sizes[rank]
        agv_inp1 = torch.full(
            (local_size, 3), float(rank + 1), dtype=torch.float32, device=device
        )
        agv_inp2 = torch.full(
            (local_size, 3), float((rank + 1) * 10), dtype=torch.float32, device=device
        )

        # Test single-tensor all_gatherv
        agv_out = comm.all_gatherv(agv_inp1, dim=0, sizes=sizes)
        expected_agv1 = torch.cat(
            [
                torch.full(
                    (sizes[r], 3), float(r + 1), dtype=torch.float32, device=device
                )
                for r in range(tp_size)
            ],
            dim=0,
        )
        torch.testing.assert_close(agv_out, expected_agv1)

        # Test list-of-tensors all_gatherv
        agv_outs = comm.all_gatherv([agv_inp1, agv_inp2], dim=0, sizes=sizes)
        assert isinstance(agv_outs, list) and len(agv_outs) == 2
        torch.testing.assert_close(agv_outs[0], expected_agv1)
        expected_agv2 = torch.cat(
            [
                torch.full(
                    (sizes[r], 3),
                    float((r + 1) * 10),
                    dtype=torch.float32,
                    device=device,
                )
                for r in range(tp_size)
            ],
            dim=0,
        )
        torch.testing.assert_close(agv_outs[1], expected_agv2)

        # Test reduce_scatter over Gloo fallback
        rs_inp = torch.arange(4 * tp_size, dtype=torch.float32, device=device) + rank
        rs_out = comm.reduce_scatter(rs_inp, dim=0)
        assert rs_out.shape == (4,)

        # Test reduce_scatterv with non-uniform sizes
        total_rs_size = sum(sizes)
        rsv_inp = torch.full(
            (total_rs_size, 2), float(rank + 1), dtype=torch.float32, device=device
        )
        rsv_out = comm.reduce_scatterv(rsv_inp, dim=0, sizes=sizes)
        assert rsv_out.shape == (local_size, 2)
        expected_rsv_sum = sum(range(1, tp_size + 1))
        torch.testing.assert_close(
            rsv_out,
            torch.full(
                (local_size, 2),
                float(expected_rsv_sum),
                dtype=torch.float32,
                device=device,
            ),
        )

        # Test gather over Gloo fallback (gather to rank 0)
        g_inp = torch.tensor([10.0 + rank], dtype=torch.float32, device=device)
        g_out = comm.gather(g_inp, dst=0, dim=0)
        if rank == 0:
            assert g_out is not None
            expected_g = torch.tensor(
                [10.0 + r for r in range(tp_size)],
                dtype=torch.float32,
                device=device,
            )
            torch.testing.assert_close(g_out, expected_g)
        else:
            assert g_out is None

        # Test broadcast over Gloo fallback
        bcast_tensor = torch.tensor([42.0 + rank], dtype=torch.float32, device=device)
        bcast_out = comm.broadcast(bcast_tensor, src=0)
        torch.testing.assert_close(
            bcast_out,
            torch.tensor([42.0], dtype=torch.float32, device=device),
        )

        # Test point-to-point send/recv over Gloo fallback
        if tp_size == 2:
            if rank == 0:
                send_tensor = torch.tensor([123.45], dtype=torch.float32, device=device)
                comm.send(send_tensor, dst=1)
            elif rank == 1:
                recv_tensor = comm.recv(torch.Size([1]), dtype=torch.float32, src=0)
                torch.testing.assert_close(
                    recv_tensor,
                    torch.tensor([123.45], dtype=torch.float32, device=device),
                )


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_fallback_collectives_pynccl_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pipeline_parallel_size: int,
) -> None:
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    multi_process_parallel(
        monkeypatch, tp_size, pipeline_parallel_size, fallback_collectives_gloo
    )
