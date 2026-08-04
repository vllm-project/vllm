# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.models.kimi_k3.nvidia.mla import _configure_no_context_parallelism


def test_configure_no_context_parallelism_uses_single_rank_state():
    impl = SimpleNamespace(
        dcp_world_size=-1,
        dcp_rank=-1,
        pcp_world_size=0,
        pcp_rank=-1,
        total_cp_world_size=0,
        total_cp_rank=-1,
        need_to_return_lse_for_decode=True,
    )

    _configure_no_context_parallelism(impl)

    assert impl.dcp_world_size == 1
    assert impl.dcp_rank == 0
    assert impl.pcp_world_size == 1
    assert impl.pcp_rank == 0
    assert impl.total_cp_world_size == 1
    assert impl.total_cp_rank == 0
    assert not impl.need_to_return_lse_for_decode
