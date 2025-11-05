
# ABOUTME: Integration tests for JL sketch state and summarizer updates.
# ABOUTME: Ensures EPS JL logic matches expectations under simple inputs.

import pytest
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch required for this test
    pytest.skip("torch not available", allow_module_level=True)

from vllm.v1.eps.state import EpsJLState
from vllm.v1.eps.summarizer import jl_update_block, jl_update_once


def test_eps_jl_state_init_shapes():
    state = EpsJLState.init(
        num_layers=2,
        num_heads=3,
        head_dim=4,
        num_groups=5,
        sketch_dim=6,
        device=torch.device("cpu"),
    )
    assert state.Phi.shape == (3, 4, 6)
    assert state.G.shape == (2, 3, 5, 6, 6)
    assert state.frob2.shape == (2, 3, 5)


def test_jl_update_once_and_block():
    state = EpsJLState.init(
        num_layers=1,
        num_heads=1,
        head_dim=3,
        num_groups=1,
        sketch_dim=2,
        device=torch.device("cpu"),
    )
    k = torch.tensor([1.0, 2.0, 3.0])
    jl_update_once(state, layer=0, head=0, group=0, k_vec=k)
    assert torch.isclose(state.frob2[0, 0, 0], torch.tensor(14.0))

    block = torch.stack([k, k * 2])
    jl_update_block(state, layer=0, head=0, group=0, K_block=block)
    assert torch.isclose(state.frob2[0, 0, 0], torch.tensor(14.0 + 14.0 + 56.0))
