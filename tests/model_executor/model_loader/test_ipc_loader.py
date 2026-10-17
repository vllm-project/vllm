# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for IpcModelLoader._apply_entries.

Covers https://github.com/vllm-project/vllm/issues/62105: cached tensors must
be swapped into the meta placeholders in place, so objects that captured the
placeholder during model construction (e.g. MoE routers holding
gate.e_score_correction_bias as a plain attribute) see the cached data
instead of being stranded on the meta tensor.
"""

import torch
from torch import nn

from vllm.model_executor.model_loader.weight_cache.ipc_loader import (
    IpcModelLoader,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    WeightCacheState,
)


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


def _make_state() -> WeightCacheState:
    # Daemon export order: parameters first, then buffers.
    entries = {
        "gate.e_score_correction_bias": TensorEntry.from_tensor(
            torch.arange(4.0), "param"
        ),
        "mamba.conv1d.weight": TensorEntry.from_tensor(
            torch.full((8, 4, 1), 2.0), "param"
        ),
        "embed.weight": TensorEntry.from_tensor(torch.full((8, 4), 3.0), "param"),
        # Post-quantization tensor absent from the meta model.
        "mamba.extra_scale": TensorEntry.from_tensor(torch.ones(8), "param"),
        "mamba.conv_weights": TensorEntry.from_tensor(
            torch.ones(8, 4), "buffer"
        ),
    }
    aliases = {"lm_head.weight": "embed.weight"}
    return WeightCacheState(entries=entries, aliases=aliases, attrs={})


def _apply(model: nn.Module, mode: str = "zero_copy") -> None:
    loader = IpcModelLoader.__new__(IpcModelLoader)
    loader.mode = mode
    IpcModelLoader._apply_entries(loader, model, _make_state(), 0)


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
