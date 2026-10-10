# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GraniteMoeHybrid weight loading under pipeline parallelism.

Every rank iterates the full checkpoint, so the loader has to skip weights
belonging to layers held by other ranks. `_load_shard`, `_load_expert` and
`_load_quant_expert` do; `_load` did not, so the router-to-gate rename and the
unstacked fallback raised `KeyError` on any rank that does not own layer 0.
"""

import pytest
import torch

from vllm.model_executor.models import granitemoehybrid
from vllm.model_executor.models.granitemoehybrid import GraniteMoeHybridModel

LOCAL_LAYER = "layers.20."
REMOTE_LAYER = "layers.0."


class _StubModel:
    """Stands in for a PP shard owning a single layer's parameters."""

    def __init__(self, params: dict[str, torch.Tensor]):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())

    def get_expert_mapping(self):
        return []


@pytest.fixture
def local_only_pp(monkeypatch):
    """Report every parameter outside LOCAL_LAYER as owned by another rank."""
    monkeypatch.setattr(
        granitemoehybrid,
        "is_pp_missing_parameter",
        lambda name, model: not name.startswith(LOCAL_LAYER),
    )


def _load(params: dict[str, torch.Tensor], weights) -> set[str]:
    return GraniteMoeHybridModel.load_weights(_StubModel(params), weights)


def test_router_rename_skips_remote_layers(local_only_pp):
    """The router-to-gate rename must not look up another rank's gate."""
    gate = LOCAL_LAYER + "block_sparse_moe.gate.weight"
    params = {gate: torch.zeros(4, 8)}
    weights = [
        (REMOTE_LAYER + "block_sparse_moe.router.layer.weight", torch.ones(4, 8)),
        (LOCAL_LAYER + "block_sparse_moe.router.layer.weight", torch.full((4, 8), 2.0)),
    ]

    loaded = _load(params, weights)

    assert loaded == {gate}
    assert torch.equal(params[gate], torch.full((4, 8), 2.0))


def test_unstacked_fallback_skips_remote_layers(local_only_pp):
    """The catch-all path covers norms and mamba parameters."""
    norm = LOCAL_LAYER + "input_layernorm.weight"
    params = {norm: torch.zeros(8)}
    weights = [
        (REMOTE_LAYER + "input_layernorm.weight", torch.ones(8)),
        (LOCAL_LAYER + "input_layernorm.weight", torch.full((8,), 3.0)),
    ]

    loaded = _load(params, weights)

    assert loaded == {norm}
    assert torch.equal(params[norm], torch.full((8,), 3.0))
