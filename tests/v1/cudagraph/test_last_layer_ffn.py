# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Last-layer FFN pruning: row selection, LoRA metadata, and graph replay."""

from typing import TYPE_CHECKING

import pytest
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

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


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
    """Shared hooks select only while indices are set and restore metadata."""

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
    install_transformer_ffn_hooks(model.block.mlp)
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


@pytest.mark.parametrize("architecture", ["llama", "qwen2", "qwen3"])
def test_causal_ffn_selection_preserves_real_decoder_outputs(architecture):
    """Different HF blocks use the same selection logic, preserving sampled rows."""
    from transformers import AutoConfig, AutoModel

    from vllm.model_executor.models.transformers.causal import _terminal_ffn

    config = AutoConfig.for_model(
        architecture,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    torch.manual_seed(42)
    model = AutoModel.from_config(config).eval()
    block = model.layers[-1]
    target = _terminal_ffn(block)
    assert target is not None
    ffn = block.get_submodule(target)
    ids = torch.tensor([[1, 5, 9, 2, 6, 3]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).last_hidden_state
        install_transformer_ffn_hooks(ffn)
        for rows in ([0, 5], [4, 1], [0, 5]):
            indices = torch.tensor(rows)
            with using_last_layer_ffn_indices(indices):
                actual = model(ids, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual[:, rows], expected[:, rows])
        torch.testing.assert_close(
            model(ids, use_cache=False).last_hidden_state, expected
        )


def test_causal_ffn_rejects_later_token_mixing():
    """Finding a GLU is insufficient when its outputs feed a sequence reduction."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP

    from vllm.model_executor.models.transformers.causal import _terminal_ffn

    class Block(nn.Module):
        def __init__(self, mix):
            super().__init__()
            self.projection = LlamaMLP(
                LlamaConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
            )
            self.mix = mix

        def forward(self, hidden_states):
            output = self.projection(hidden_states)
            if self.mix:
                output = output.mean(dim=-2, keepdim=True)
            return hidden_states + output

    assert _terminal_ffn(Block(False)) == "projection"
    assert _terminal_ffn(Block(True)) is None


class CausalFFNTestWorker:
    model_runner: "GPUModelRunner"

    def causal_ffn_batches(self):
        return self.model_runner.last_layer_ffn_pruner.batches


@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
)
def test_transformers_causal_mixin_prunes_in_generation(
    vllm_runner, monkeypatch, model_name
):
    """An inherited capability must reach the runner, not silently disable pruning."""
    from tests.models.utils import check_logprobs_close
    from vllm.transformers_utils.utils import maybe_model_redirect

    model_name = maybe_model_redirect(model_name)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    prompts = [
        "The capital of France is",
        "Explain why the sky appears blue in a few words:",
    ]
    outputs = []
    for enabled in (False, True):
        monkeypatch.setenv("VLLM_ENABLE_LAST_LAYER_FFN_PRUNING", str(int(enabled)))
        with vllm_runner(
            model_name,
            model_impl="transformers",
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=512,
            max_num_seqs=2,
            enable_prefix_caching=False,
            compilation_config={"mode": 0},
            worker_extension_cls=f"{__name__}.CausalFFNTestWorker",
        ) as runner:
            outputs.append(runner.generate_greedy_logprobs(prompts, 8, 5))
            batches = runner.llm.collective_rpc("causal_ffn_batches")
            assert all((count > 0) == enabled for count in batches)
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="original",
        name_1="pruned",
    )
