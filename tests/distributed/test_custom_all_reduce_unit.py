# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.distributed as dist

import vllm.distributed.device_communicators.custom_all_reduce as custom_ar


def test_custom_allreduce_respects_nccl_p2p_disable(monkeypatch):
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setattr(custom_ar, "custom_ar", True)
    monkeypatch.setattr(custom_ar.dist, "get_backend", lambda group: dist.Backend.GLOO)
    monkeypatch.setattr(custom_ar.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(custom_ar.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        custom_ar,
        "in_the_same_node_as",
        lambda group, source_rank: [True, True],
    )

    def fail_can_p2p(rank, world_size):
        pytest.fail("_can_p2p should not run when NCCL_P2P_DISABLE=1")

    monkeypatch.setattr(custom_ar, "_can_p2p", fail_can_p2p)

    allreduce = custom_ar.CustomAllreduce(group=object(), device=0)

    assert allreduce.disabled
