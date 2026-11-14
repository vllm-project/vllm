# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.nemotron_h import (
    NemotronHModel,
    NemotronHSkippedLayer,
    get_moe_top_k,
    get_skip_layers,
)


def _config(skip_layers=None):
    return SimpleNamespace(hybrid_override_pattern="MEM*EME", skip_layers=skip_layers)


def test_get_skip_layers():
    assert get_skip_layers(_config()) == frozenset()
    assert get_skip_layers(_config([4, 5, 4])) == frozenset({4, 5})
    assert get_skip_layers(_config(["4"])) == frozenset({4})
    with pytest.raises(ValueError, match="must be in"):
        get_skip_layers(_config([7]))
    with pytest.raises(ValueError, match="every layer"):
        get_skip_layers(_config(list(range(7))))


def test_get_moe_top_k():
    config = SimpleNamespace(
        num_experts_per_tok=6, n_routed_experts=128, n_group=8, topk_group=2
    )
    assert get_moe_top_k(config, None) == 6
    assert get_moe_top_k(config, 32) == 32
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        get_moe_top_k(config, 33)
    with pytest.raises(ValueError):
        get_moe_top_k(config, 0)


def test_skipped_layer_is_identity():
    hidden, residual = torch.randn(3, 8), torch.randn(3, 8)
    out_hidden, out_residual = NemotronHSkippedLayer()(hidden, residual)
    assert out_hidden is hidden and out_residual is residual
    assert not list(NemotronHSkippedLayer().parameters())


def test_aux_hidden_states_after_skipped_layer_rejected():
    model = NemotronHModel.__new__(NemotronHModel)
    torch.nn.Module.__init__(model)
    model.skip_layers = frozenset({4, 5})
    # Aux index i is the output of layer i - 1.
    with pytest.raises(ValueError, match=r"skipped layers \[5\]"):
        model._set_aux_hidden_state_layers((2, 6))
    model._set_aux_hidden_state_layers((0, 2, 4))
    assert model.aux_hidden_state_layers == (0, 2, 4)
