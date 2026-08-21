# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import torch

import vllm.v1.attention.ops.pcp as pcp


class _FakeGroupCoordinator:
    device_group = None
    world_size = 2


def test_fused_workspace_cache_is_scoped_by_group(monkeypatch):
    pcp._get_fused_pcp_norm_rope_workspace.cache_clear()
    monkeypatch.setattr(pcp, "direct_cp_multicast_enabled", lambda *args: True)
    workspace_a = object()
    workspace_b = object()
    init_workspace = MagicMock(side_effect=[workspace_a, workspace_b])
    monkeypatch.setattr(
        pcp,
        "DirectPCPFusedNormRopeWorkspace",
        init_workspace,
    )
    group_a = _FakeGroupCoordinator()
    group_b = _FakeGroupCoordinator()
    args = (torch.device("cpu"), 16, "fp8", False, 1)

    try:
        first = pcp._get_fused_pcp_norm_rope_workspace(group_a, *args)
        reused = pcp._get_fused_pcp_norm_rope_workspace(group_a, *args)
        isolated = pcp._get_fused_pcp_norm_rope_workspace(group_b, *args)
    finally:
        pcp._get_fused_pcp_norm_rope_workspace.cache_clear()

    assert first is workspace_a
    assert reused is workspace_a
    assert isolated is workspace_b
    assert init_workspace.call_count == 2
