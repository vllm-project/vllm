# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for weight_loader defensive getattr fallback.

Regression tests for https://github.com/vllm-project/vllm/issues/34201.

After sleep mode level 2 discards model weights and reload_weights() is called,
fresh Parameter objects may lack the custom `weight_loader` attribute that is
normally set during model initialisation.  Model files previously accessed
`param.weight_loader` directly (bare attribute access) in the stacked-parameter
loading branch, causing an AttributeError on reload.

All model `load_weights` implementations now use:

    weight_loader = getattr(param, "weight_loader", default_weight_loader)

in BOTH the stacked-parameter branch and the else branch, matching the
defensive pattern already established in the else-branch of llama.py.
"""

import torch
import torch.nn as nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _ParamWithLoader(nn.Parameter):
    """Parameter that carries a custom weight_loader (normal init path)."""

    def __new__(cls, data):
        instance = super().__new__(cls, data)
        return instance

    def set_loader(self, loader):
        self.weight_loader = loader
        return self


def _make_param_without_loader(shape=(4, 4)) -> nn.Parameter:
    """Simulate a Parameter created after sleep-mode weight discard."""
    return nn.Parameter(torch.zeros(shape))


def test_getattr_with_loader_present():
    """When weight_loader is present, getattr must return it."""
    call_log = []

    def custom_loader(param, weight, shard_id=None):
        call_log.append(("custom", shard_id))

    param = _ParamWithLoader(torch.zeros(4, 4)).set_loader(custom_loader)
    loader = getattr(param, "weight_loader", default_weight_loader)
    assert loader is custom_loader, "Expected the custom loader to be returned"

    loader(param, torch.ones(4, 4), "q")
    assert call_log == [("custom", "q")]


def test_getattr_falls_back_to_default():
    """When weight_loader is absent (e.g. after sleep mode), fall back to default."""
    param = _make_param_without_loader()
    assert not hasattr(param, "weight_loader"), "Pre-condition: no weight_loader"

    loader = getattr(param, "weight_loader", default_weight_loader)
    assert loader is default_weight_loader, "Must fall back to default_weight_loader"

    # default_weight_loader simply copies the weight data; it must not raise
    weight = torch.ones(4, 4)
    loader(param, weight)
    assert torch.allclose(param.data, weight)


def test_bare_access_raises_without_loader():
    """Confirm that the old bare access pattern would fail (documents the bug)."""
    param = _make_param_without_loader()
    try:
        _ = param.weight_loader  # type: ignore[attr-defined]
        raised = False
    except AttributeError:
        raised = True
    assert raised, "Bare access to missing weight_loader must raise AttributeError"


def test_getattr_in_stacked_params_pattern():
    """End-to-end simulation of the stacked-parameter loading branch."""
    loaded_calls = []

    def shard_loader(param, weight, shard_id):
        loaded_calls.append(shard_id)
        param.data.copy_(weight)

    # Case 1: param has weight_loader (normal init, no sleep mode)
    param_with = _ParamWithLoader(torch.zeros(8, 4)).set_loader(shard_loader)
    weight_loader = getattr(param_with, "weight_loader", default_weight_loader)
    weight_loader(param_with, torch.ones(8, 4), "q")
    assert "q" in loaded_calls

    # Case 2: param lacks weight_loader (after sleep-mode reload)
    param_without = _make_param_without_loader((8, 4))
    weight_loader2 = getattr(param_without, "weight_loader", default_weight_loader)
    assert weight_loader2 is default_weight_loader
    # default_weight_loader ignores the shard_id argument, just copies data
    weight_loader2(param_without, torch.full((8, 4), 2.0))
    assert torch.allclose(param_without.data, torch.full((8, 4), 2.0))
