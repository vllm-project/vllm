# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for installing IPC-cached tensors onto a meta model.

Covers https://github.com/vllm-project/vllm/issues/62105: cached tensors must
be swapped into the meta placeholders in place, so objects that captured the
placeholder during model construction (e.g. MoE routers holding
gate.e_score_correction_bias as a plain attribute) see the cached data
instead of being stranded on the meta tensor.
"""

import torch
from torch import nn

from vllm.model_executor.model_loader.utils import _register_processed_tensors


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(4), requires_grad=False
        )


class _Router:
    """Plain object like GroupedTopKRouter: not an nn.Module, so the captured
    tensor is a plain attribute and is never re-resolved through _parameters."""

    def __init__(self, e_score_correction_bias: torch.Tensor):
        self.e_score_correction_bias = e_score_correction_bias


class _MambaLike(nn.Module):
    """conv_weights is registered as a view of the conv1d.weight parameter,
    mirroring MambaMixer2."""

    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(4, 8, 1, bias=False)
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        self.register_buffer(
            "conv_weights",
            self.conv1d.weight.view(8, 4),
            persistent=False,
        )


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()
        self.router = _Router(self.gate.e_score_correction_bias)
        self.mamba = _MambaLike()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        # Tied weights: same Parameter object under two names.
        self.lm_head.weight = self.embed.weight


def _apply(model: nn.Module) -> None:
    # Intentionally list parameters before the buffer, matching daemon
    # export order, so the buffers-first sort in _register_processed_tensors
    # is actually exercised.
    tensors = {
        "gate.e_score_correction_bias": torch.arange(4.0),
        "mamba.conv1d.weight": torch.full((8, 4, 1), 2.0),
        "embed.weight": torch.full((8, 4), 3.0),
        # Post-quantization tensor absent from the meta model.
        "mamba.extra_scale": torch.ones(8),
        "mamba.conv_weights": torch.ones(8, 4),
    }
    params = {
        "gate.e_score_correction_bias",
        "mamba.conv1d.weight",
        "embed.weight",
        "mamba.extra_scale",
    }
    manifest = {
        "tensors": {name: {"is_param": name in params} for name in tensors}
    }
    aliases = {"lm_head.weight": "embed.weight"}
    _register_processed_tensors(model, tensors, manifest, aliases)


def test_apply_entries_keeps_captured_meta_references_alive():
    with torch.device("meta"):
        model = _ToyModel()
    old_bias = model.gate.e_score_correction_bias
    assert old_bias.is_meta

    _apply(model)

    # The router captured the placeholder object; it must now see the cached
    # data through the *same* object (in-place swap, not re-registration).
    assert model.router.e_score_correction_bias is old_bias
    assert model.gate.e_score_correction_bias is old_bias
    assert not old_bias.is_meta
    torch.testing.assert_close(old_bias, torch.arange(4.0))


def test_apply_entries_releases_buffer_views_before_param_swap():
    # Without applying buffers first, swapping mamba.conv1d.weight would fail
    # because the conv_weights buffer is a live view of the meta parameter.
    with torch.device("meta"):
        model = _ToyModel()

    _apply(model)

    assert not model.mamba.conv1d.weight.is_meta
    torch.testing.assert_close(
        model.mamba.conv1d.weight, torch.full((8, 4, 1), 2.0)
    )
    assert not model.mamba.conv_weights.is_meta
    torch.testing.assert_close(model.mamba.conv_weights, torch.ones(8, 4))


def test_apply_entries_preserves_tied_weight_aliases():
    with torch.device("meta"):
        model = _ToyModel()

    _apply(model)

    assert model.lm_head.weight is model.embed.weight
    assert not model.embed.weight.is_meta
    torch.testing.assert_close(model.embed.weight, torch.full((8, 4), 3.0))


def test_apply_entries_registers_tensors_absent_from_meta_model():
    with torch.device("meta"):
        model = _ToyModel()

    _apply(model)

    scale = model.mamba._parameters.get("extra_scale")
    assert isinstance(scale, nn.Parameter)
    torch.testing.assert_close(scale, torch.ones(8))


def test_apply_entries_leaves_no_meta_tensors():
    with torch.device("meta"):
        model = _ToyModel()

    _apply(model)

    leftovers = [
        name
        for name, t in list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
        if t.is_meta
    ]
    assert leftovers == []
