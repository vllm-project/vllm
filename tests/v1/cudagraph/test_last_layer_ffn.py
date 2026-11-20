# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Last-layer FFN pruning: row selection, LoRA metadata, and graph replay."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.layers.last_layer_ffn import (
    apply_pruned_last_ffn,
    current_last_layer_ffn_indices,
    install_transformer_ffn_hooks,
    restore_pruned_last_ffn,
    using_last_layer_ffn_indices,
)
from vllm.v1.worker.gpu.last_layer_ffn import (
    LastLayerFFNGraphWrapper,
    last_layer_ffn_graph_key,
)


class _Adapter:
    def __init__(self):
        self.token_mapping_meta = "per-token"
        self.prompt_mapping_meta = "per-output"


class _Passthrough(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, hidden):
        return hidden + self.bias


class _LoRALinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.punica_wrapper = _Adapter()

    def forward(self, hidden):
        self.seen = self.punica_wrapper.token_mapping_meta
        return hidden + 1


class _FailingLoRALinear(_LoRALinear):
    def forward(self, hidden):
        self.seen = self.punica_wrapper.token_mapping_meta
        raise RuntimeError("mlp failed")


def _identity_norm(hidden, residual):
    return hidden, residual


def test_context_manager_restores_indices_on_success_and_failure():
    """The runner's context manager must clear rows after both return and raise."""
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    residual = torch.zeros_like(hidden)
    previous = torch.tensor([1])
    with using_last_layer_ffn_indices(previous):
        with using_last_layer_ffn_indices(torch.tensor([0, 3])):
            selected, _ = apply_pruned_last_ffn(
                hidden, residual, _identity_norm, _Passthrough(0)
            )
            assert selected.shape[0] == 2
        assert torch.equal(current_last_layer_ffn_indices(), previous)
        try:
            with using_last_layer_ffn_indices(torch.tensor([5])):
                raise RuntimeError("capture failed")
        except RuntimeError:
            pass
        else:
            raise AssertionError("exception did not propagate")
        # [5] is past the next tensor. Leaving it set would make index_select fail.
        assert torch.equal(current_last_layer_ffn_indices(), previous)
        with using_last_layer_ffn_indices(None):
            small = torch.arange(2, dtype=torch.float32).reshape(1, 2)
            output, restored_residual = apply_pruned_last_ffn(
                small,
                torch.zeros_like(small),
                lambda rows, residual: (rows + 3, residual),
                _Passthrough(4),
            )
            torch.testing.assert_close(output, small + 7)
            torch.testing.assert_close(restored_residual, torch.zeros_like(small))
            torch.testing.assert_close(restore_pruned_last_ffn(small, 1), small)
    assert current_last_layer_ffn_indices() is None


def test_lora_mapping_is_restored_after_pruned_ffn():
    """Reduced FFN rows use the per-output mapping, then the original returns."""
    mlp = _LoRALinear()
    hidden = torch.zeros(4, 2)
    residual = torch.zeros_like(hidden)
    with using_last_layer_ffn_indices(None):
        apply_pruned_last_ffn(hidden, residual, _identity_norm, mlp)
    assert mlp.seen == "per-token"
    assert mlp.punica_wrapper.token_mapping_meta == "per-token"

    with using_last_layer_ffn_indices(torch.tensor([1, 2])):
        apply_pruned_last_ffn(hidden, residual, _identity_norm, mlp)
    assert mlp.seen == "per-output"
    assert mlp.punica_wrapper.token_mapping_meta == "per-token"
    assert current_last_layer_ffn_indices() is None

    failing = _FailingLoRALinear()
    try:
        with using_last_layer_ffn_indices(torch.tensor([1])):
            apply_pruned_last_ffn(hidden, residual, _identity_norm, failing)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mlp exception did not propagate")
    assert failing.seen == "per-output"
    assert failing.punica_wrapper.token_mapping_meta == "per-token"
    assert current_last_layer_ffn_indices() is None


def test_transformer_hook_restores_shape_and_lora_mapping():
    """Qwen2 hooks select only while indices are set, and always restore metadata."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _LoRALinear()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()
            self._decoder_layers = [self.block]

    model = Model()
    assert install_transformer_ffn_hooks(model)
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    with using_last_layer_ffn_indices(None):
        unchanged = model.block.mlp(hidden)
    torch.testing.assert_close(unchanged, hidden + 1)
    assert model.block.mlp.punica_wrapper.token_mapping_meta == "per-token"

    with using_last_layer_ffn_indices(torch.tensor([0, 3])):
        selected = model.block.mlp(hidden)
    assert selected.shape == (4, 2)
    torch.testing.assert_close(selected[[0, 3]], hidden[[0, 3]] + 1)
    torch.testing.assert_close(selected[[1, 2]], torch.zeros(2, 2))
    assert model.block.mlp.seen == "per-output"
    assert model.block.mlp.punica_wrapper.token_mapping_meta == "per-token"
    assert current_last_layer_ffn_indices() is None


def test_first_capture_executes_and_replay_uses_current_rows(dist_init, monkeypatch):
    """New row-count graphs must execute immediately and reuse live indices."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = VllmConfig()

    def select(x):
        return x.index_select(0, current_last_layer_ffn_indices()) + 1

    graph = LastLayerFFNGraphWrapper(select, config, key_fn=last_layer_ffn_graph_key)
    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    indices = torch.tensor([0, 2], device="cuda")
    with set_forward_context(
        None,
        config,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=4),
    ):
        for rows, offset, selected_rows in (
            (1, 0, [0, 2]),
            (2, 16, [1, 3]),
            (1, 32, [3, 0]),
            (2, 48, [2, 1]),
        ):
            x.copy_(torch.arange(16, device="cuda").reshape(4, 4) + offset)
            indices.copy_(torch.tensor(selected_rows, device="cuda"))
            selected = indices[:rows]
            with using_last_layer_ffn_indices(selected):
                actual = graph(x)
                torch.testing.assert_close(actual, select(x), rtol=0, atol=0)
    assert current_last_layer_ffn_indices() is None
    assert len(graph.entries) == 2
