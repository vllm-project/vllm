# ABOUTME: Validates JL updates triggered during KV cache writes.
# ABOUTME: Ensures EPS pre-write hooks accumulate Gram summaries correctly.

import pytest
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch required for this test
    pytest.skip("torch not available", allow_module_level=True)

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.context import (
    EpsForwardContext,
    EpsGroupRuntime,
    EpsLayerInfo,
    EpsRequestRuntime,
    eps_context,
)
from vllm.v1.eps.state import EpsJLState
from vllm.v1.eps.writer import apply_eps_prefill_updates


def _identity_phi(state: EpsJLState) -> None:
    """Replace random projection with identity for deterministic tests."""
    num_heads = state.Phi.shape[0]
    head_dim = state.Phi.shape[1]
    sketch_dim = state.Phi.shape[2]
    assert sketch_dim == head_dim
    eye = torch.eye(head_dim, dtype=state.Phi.dtype, device=state.Phi.device)
    for h in range(num_heads):
        state.Phi[h].copy_(eye)


def make_context(*, cfg: EpsRuntimeConfig, state: EpsJLState) -> EpsForwardContext:
    group_runtime = EpsGroupRuntime(
        block_size=2,
        request_runtimes=[
            EpsRequestRuntime(
                request_id="req0",
                state=state,
                block_mapping={0: 0, 1: 1},
            )
        ],
    )
    layer_info = EpsLayerInfo(group_id=0, layer_index=0, layer_name="layer0")
    return EpsForwardContext(
        enabled=True,
        cfg=cfg,
        layer_map={"layer0": layer_info},
        group_runtimes=[group_runtime],
        token_request_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        cudagraph_capture=False,
    )


def test_prefill_updates_accumulate_gram():
    cfg = EpsRuntimeConfig(
        enabled=True,
        scope="union",
        method="jl",
        head_scope="all",
        group_blocks=1,
        last_n=1,
        alpha=1.0,
        dim=2,
        top_pages=None,
        strict=False,
    )
    state = EpsJLState.init(
        num_layers=1,
        num_heads=1,
        head_dim=2,
        num_groups=1,
        sketch_dim=2,
        device=torch.device("cpu"),
    )
    _identity_phi(state)
    ctx = make_context(cfg=cfg, state=state)

    key = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 2.0]],
            [[3.0, 0.0]],
            [[0.0, 4.0]],
        ]
    )
    slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    with eps_context(ctx):
        apply_eps_prefill_updates(
            layer_name="layer0",
            key=key,
            slot_mapping=slot_mapping,
        )

    # Block 0 tokens -> norms 1^2 + 0^2 + 0^2 + 2^2 = 5
    # Block 1 tokens -> norms 3^2 + 0^2 + 0^2 + 4^2 = 25
    assert torch.isclose(state.frob2[0, 0, 0], torch.tensor(5.0))
    assert torch.isclose(state.frob2[0, 0, 1], torch.tensor(25.0))


def test_context_integration_updates_target_request():
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
    state = EpsJLState.init(
        num_layers=1,
        num_heads=1,
        head_dim=2,
        num_groups=1,
        sketch_dim=2,
        device=torch.device("cpu"),
    )
    _identity_phi(state)
    ctx = EpsForwardContext(
        enabled=True,
        cfg=cfg,
        layer_map={"layer0": EpsLayerInfo(group_id=0, layer_index=0, layer_name="layer0")},
        group_runtimes=[
            EpsGroupRuntime(
                block_size=2,
                request_runtimes=[
                    EpsRequestRuntime("req0", state, {0: 0}),
                    EpsRequestRuntime("req1", None, {}),
                ],
            )
        ],
        token_request_indices=torch.tensor([0, 0], dtype=torch.int32),
        cudagraph_capture=False,
    )

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)

    with eps_context(ctx):
        apply_eps_prefill_updates(
            layer_name="layer0",
            key=key,
            slot_mapping=slot_mapping,
        )

    assert torch.isclose(state.frob2[0, 0, 0], torch.tensor(5.0))
