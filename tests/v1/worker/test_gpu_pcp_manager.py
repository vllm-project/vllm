# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.pcp_manager as pcp_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partition_reuses_gpu_cursor_for_replicated_spec_decode():
    device = torch.device("cuda")
    global_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=4, device=device)
    global_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=4,
        input_buffers=global_buffers,
    )

    # Model an async step after rejection: the CPU scheduler cursor is still
    # optimistic, while the GPU cursor used to build positions/seq_lens has
    # already rolled back to the accepted prefix.
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    global_batch.num_computed_tokens_np[:] = 20
    global_batch.prefill_len_np[:] = 8
    global_batch.num_computed_prefill_tokens_np[:] = 8
    global_batch.positions.copy_(torch.arange(10, 14, device=device))
    global_batch.seq_lens.fill_(14)

    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=device,
        req_states=SimpleNamespace(),
        max_num_reqs=1,
        max_num_tokens=4,
    )
    local_batch = manager.partition_batch(global_batch)

    # q_len > 1 remains one replicated decode row. Its actual device metadata
    # follows the corrected GPU cursor, not the stale CPU upper bound.
    assert local_batch.num_reqs == 1
    assert local_batch.num_scheduled_tokens.tolist() == [4]
    torch.testing.assert_close(
        local_batch.positions,
        torch.arange(10, 14, device=device),
    )
    torch.testing.assert_close(
        local_batch.seq_lens,
        torch.tensor([14], dtype=torch.int32, device=device),
    )
    assert local_batch.num_computed_tokens_np.tolist() == [20]


def test_restore_hidden_states_appends_zero_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    manager._global_batch = SimpleNamespace(
        num_tokens=5,
        num_tokens_after_padding=8,
    )
    restored = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    manager._hidden_restore_idx = torch.arange(5)
    monkeypatch.setattr(
        pcp_module,
        "get_pcp_group",
        lambda: SimpleNamespace(all_gather=lambda *_args, **_kwargs: restored),
    )

    actual = manager.restore_hidden_states(torch.empty(0))

    assert actual.shape == (8, 2)
    torch.testing.assert_close(actual[:5], restored)
    torch.testing.assert_close(actual[5:], torch.zeros(3, 2))
