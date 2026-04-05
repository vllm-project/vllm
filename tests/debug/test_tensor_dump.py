# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.debug.tensor_dump."""

import os

import pytest
import torch
from torch import nn

from vllm.debug.tensor_dump import TensorDumper, register_forward_hook_for_model


class TestTensorDumper:
    """Unit tests for TensorDumper."""

    def test_add_tensor_single(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        t = torch.randn(2, 3)
        dumper.add_tensor("layer.0.output", t)
        assert "layer.0.output" in dumper._current_tensors
        assert torch.equal(dumper._current_tensors["layer.0.output"], t.cpu())

    def test_add_tensor_tuple(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        t1 = torch.randn(2, 3)
        dumper.add_tensor("layer.0", (t1,))
        assert torch.equal(dumper._current_tensors["layer.0"], t1.cpu())

    def test_add_tensor_multi_tuple(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        dumper.add_tensor("layer.0", (t1, t2))
        result = dumper._current_tensors["layer.0"]
        assert isinstance(result, list)
        assert len(result) == 2

    def test_dump_creates_file(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        dumper.add_tensor("x", torch.randn(2, 3))
        dumper.dump_current_tensors()
        pt_files = list(tmp_path.rglob("*.pt"))
        assert len(pt_files) == 1
        assert "Pass00000.pt" in pt_files[0].name

    def test_dump_increments_pass_id(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        dumper.add_tensor("x", torch.randn(2, 3))
        dumper.dump_current_tensors()
        dumper.add_tensor("y", torch.randn(2, 3))
        dumper.dump_current_tensors()
        pt_files = sorted(tmp_path.rglob("*.pt"))
        assert len(pt_files) == 2

    def test_skip_passes(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0, skip_passes=2)
        # First two passes should be skipped.
        dumper.add_tensor("x", torch.randn(2, 3))
        dumper.dump_current_tensors()
        dumper.add_tensor("x", torch.randn(2, 3))
        dumper.dump_current_tensors()
        assert len(list(tmp_path.rglob("*.pt"))) == 0
        # Third pass should produce output.
        dumper.add_tensor("x", torch.randn(2, 3))
        dumper.dump_current_tensors()
        assert len(list(tmp_path.rglob("*.pt"))) == 1

    def test_empty_dump_no_file(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        dumper.dump_current_tensors()
        assert len(list(tmp_path.rglob("*.pt"))) == 0

    def test_process_dir_naming(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 4, 1, 2)
        dir_name = os.path.basename(dumper.get_dump_dir())
        assert dir_name.startswith("TP1_PP2_Rank9_pid")

    def test_saved_tensors_loadable(self, tmp_path):
        dumper = TensorDumper(str(tmp_path), None, 1, 0, 0)
        original = torch.randn(3, 4)
        dumper.add_tensor("test_tensor", original)
        dumper.dump_current_tensors()
        pt_files = list(tmp_path.rglob("*.pt"))
        loaded = torch.load(pt_files[0], weights_only=True)
        assert torch.equal(loaded["test_tensor"], original.cpu())


class InnerModel(nn.Module):
    """Inner model that mimics a real model's structure (e.g. LlamaModel)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SimpleModel(nn.Module):
    """A minimal model with the structure expected by the hook registrar.

    Mirrors the real vLLM pattern where the outer module has a ``model``
    attribute whose ``forward()`` is called via ``self.model(x)``.
    """

    def __init__(self):
        super().__init__()
        self.model = InnerModel()

    def forward(self, x):
        return self.model(x)


class TestRegisterForwardHook:
    """Tests for register_forward_hook_for_model."""

    def test_hooks_registered(self, tmp_path):
        model = SimpleModel()
        dumper = register_forward_hook_for_model(model, str(tmp_path))
        assert dumper is not None
        assert os.path.isdir(dumper.get_dump_dir())

    def test_forward_produces_dump(self, tmp_path):
        model = SimpleModel()
        register_forward_hook_for_model(model, str(tmp_path))
        x = torch.randn(2, 4)
        model(x)
        pt_files = list(tmp_path.rglob("*.pt"))
        assert len(pt_files) >= 1

    def test_dump_layers_filter(self, tmp_path):
        model = SimpleModel()
        # Only dump layer 0.
        register_forward_hook_for_model(model, str(tmp_path), dump_layers=[0])
        x = torch.randn(2, 4)
        model(x)
        pt_files = list(tmp_path.rglob("*.pt"))
        assert len(pt_files) >= 1
        # Load and verify only layer 0 keys exist (not layer 1 or 2).
        data = torch.load(pt_files[0], weights_only=True)
        for key in data:
            if "layers." in key:
                layer_num = key.split("layers.")[1].split(".")[0]
                assert layer_num == "0", f"Unexpected layer in dump: {key}"

    def test_missing_top_level_raises(self, tmp_path):
        # A model without a "model" sub-module should raise.
        model = nn.Linear(4, 4)
        with pytest.raises(AssertionError, match="model should have a module"):
            register_forward_hook_for_model(model, str(tmp_path))
