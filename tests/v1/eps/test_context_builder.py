# ABOUTME: Ensures EPS forward context builder wires request metadata correctly.
# ABOUTME: Validates layer/group mappings used by KV write hooks.

import pytest
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch required for this test
    pytest.skip("torch not available", allow_module_level=True)

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.runtime import build_eps_forward_context
from vllm.v1.eps.state import EpsJLState


def make_state():
    state = EpsJLState.init(
        num_layers=1,
        num_heads=1,
        head_dim=2,
        num_groups=1,
        sketch_dim=2,
        device=torch.device("cpu"),
    )
    return state


def test_build_eps_forward_context_maps_requests():
    cfg = EpsRuntimeConfig(
        enabled=True,
        scope="union",
        method="jl",
        head_scope="all",
        group_blocks=1,
        last_n=0,
        alpha=1.0,
        dim=2,
        top_pages=None,
        strict=False,
    )
    state_a = make_state()
    state_b = make_state()

    ctx = build_eps_forward_context(
        cfg=cfg,
        layer_lookup={"layer0": (0, 0)},
        group_block_sizes=[2],
        request_ids=["req0", "req1"],
        request_states=[[state_a], [state_b]],
        request_block_ids=[([10, 11],), ([20],)],
        token_request_indices=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        cudagraph_capture=False,
    )

    assert ctx.layer_map["layer0"].group_id == 0
    assert ctx.group_runtimes[0].block_size == 2

    req0_runtime = ctx.group_runtimes[0].request_runtimes[0]
    req1_runtime = ctx.group_runtimes[0].request_runtimes[1]

    assert req0_runtime.request_id == "req0"
    assert req0_runtime.block_mapping == {10: 0, 11: 1}
    assert req1_runtime.block_mapping == {20: 0}
    assert req1_runtime.state is state_b
    assert torch.equal(ctx.token_request_indices, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
