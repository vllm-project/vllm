# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compact multicast with a dense collective oracle."""

import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _worker(rank: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    nccl = dist.new_group(backend="nccl")
    from vllm.v1.worker.gpu.pcp_hidden_restore import PCPMulticastHiddenStateRestorer

    for dtype in (torch.bfloat16, torch.float16):
        restorer = PCPMulticastHiddenStateRestorer(
            group=dist.group.WORLD,
            device=torch.device("cuda", rank),
            max_num_tokens=64,
            hidden_size=32,
            dtype=dtype,
        )
        for selected in (1, 7, 64):
            for skew in (False, True):
                n = selected if skew else (selected + 3) // 4
                index = torch.arange(n, device="cuda", dtype=torch.int64) * 2
                rows = torch.arange(selected, device="cuda")
                owners = torch.zeros_like(rows) if skew else rows % 4
                slots = rows if skew else rows // 4
                restore = owners * n + slots
                previous = previous_expected = None
                for epoch in range(3):
                    hidden = torch.randn((2 * n + 1, 32), device="cuda", dtype=dtype)
                    full = torch.empty(
                        (4 * hidden.shape[0], 32), device="cuda", dtype=dtype
                    )
                    dist.all_gather_into_tensor(full, hidden, group=nccl)
                    expected = full[owners * hidden.shape[0] + slots * 2]
                    if rank == 0 and epoch == 1:
                        time.sleep(0.01)
                    result = restorer.restore_selected(
                        hidden,
                        index,
                        restore,
                        num_selected_rows=selected,
                    )
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    if previous is not None:
                        torch.testing.assert_close(
                            previous, previous_expected, rtol=0, atol=0
                        )
                    previous, previous_expected = result, expected.clone()
        restorer.close()
        restorer.close()
        with pytest.raises(RuntimeError, match="closed"):
            restorer.restore_selected(
                hidden, index, restore, num_selected_rows=selected
            )
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.accelerator.device_count() < 4,
    reason="requires four peer-connected CUDA GPUs",
)
def test_selected_rows_match_dense_oracle() -> None:
    mp.spawn(_worker, args=(get_open_port(),), nprocs=4, join=True)
