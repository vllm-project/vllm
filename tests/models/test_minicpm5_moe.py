# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.models.minicpm5_moe.nvidia.attention import apply_attention_gate
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.minicpm5_moe import MiniCPM5MoEConfig


class TestMiniCPM5MoEConfig(unittest.TestCase):
    def test_checkpoint_defaults_and_loading_without_remote_code(self):
        """The 15B checkpoint has head_dim != hidden_size / num_attention_heads."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = {
                "model_type": "minicpm5_moe",
                "architectures": ["MiniCPM5MoEForCausalLM"],
                "vocab_size": 130560,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 28,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 524288,
                "pad_token_id": 1,
                "bos_token_id": 0,
                "num_experts_per_tok": 16,
                "n_routed_experts": 160,
                "n_shared_experts": 1,
                "moe_intermediate_size": 512,
                "first_k_dense_replace": 1,
                "routed_scaling_factor": 3.66,
                "use_gated_attention": True,
                "rope_theta": 5000000,
                "eos_token_id": [1, 130073],
                "auto_map": {"AutoConfig": "must_not_execute.Config"},
            }
            (path / "config.json").write_text(json.dumps(data))
            (path / "must_not_execute.py").write_text(
                "raise RuntimeError('remote code executed')"
            )
            config = get_config(str(path), trust_remote_code=False)
            self.assertIsInstance(config, MiniCPM5MoEConfig)
            defaults = MiniCPM5MoEConfig()
            for key, value in data.items():
                if key != "auto_map":
                    self.assertEqual(getattr(config, key), value, key)
                if key not in {"architectures", "auto_map"}:
                    self.assertEqual(getattr(defaults, key), value, key)
            self.assertEqual(config.rope_parameters["rope_theta"], 5000000)
            self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_legacy_rope_type_is_normalized(self):
        scaling = {"type": "linear", "factor": 2.0}
        config = MiniCPM5MoEConfig(rope_theta=12345, rope_parameters=scaling)
        self.assertEqual(
            config.rope_parameters,
            {
                "rope_type": "linear",
                "factor": 2.0,
                "rope_theta": 12345,
            },
        )
        self.assertEqual(scaling, {"type": "linear", "factor": 2.0})

    def test_explicit_v5_rope_wins_and_round_trips(self):
        parameters = {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
        config = MiniCPM5MoEConfig(rope_theta=12345, rope_parameters=parameters)
        self.assertEqual(config.rope_parameters, parameters)
        self.assertEqual(config.rope_theta, 45678)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            reloaded = MiniCPM5MoEConfig.from_pretrained(directory)
        self.assertEqual(reloaded.rope_parameters, parameters)
        self.assertEqual(
            parameters, {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
        )

    def test_explicit_none_fallbacks_and_forwarded_kwargs(self):
        config = MiniCPM5MoEConfig(
            hidden_size=96,
            num_attention_heads=3,
            num_key_value_heads=None,
            head_dim=None,
            num_experts_per_tok=None,
            use_cache=False,
            custom_field=7,
        )
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.num_key_value_heads, 3)
        self.assertIsNone(config.num_experts_per_tok)
        self.assertFalse(config.use_cache)
        self.assertEqual(config.custom_field, 7)


def make_fp32_gate(weight, correction_bias):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    with torch.device(weight.device):
        gate = GateLinear(
            weight.shape[1],
            weight.shape[0],
            params_dtype=torch.float32,
            out_dtype=torch.float32,
        )
    default_weight_loader(gate.weight, weight)
    gate.e_score_correction_bias = correction_bias
    return gate


@pytest.mark.usefixtures("dist_init")
class TestMiniCPM5Gate(unittest.TestCase):
    def test_bf16_parameters_produce_fp32_logits_without_early_rounding(self):
        weight = nn.Parameter(torch.ones(1, 2, dtype=torch.bfloat16))
        correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        gate = make_fp32_gate(weight, correction_bias)
        hidden = torch.tensor([[1.0, 1.0 / 256]], dtype=torch.bfloat16)

        logits, bias = gate(hidden)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertIsNone(bias)
        self.assertEqual(logits.item(), 1.0 + 1.0 / 256)
        self.assertNotEqual(logits.item(), logits.to(torch.bfloat16).float().item())
        self.assertEqual(gate.weight.dtype, torch.float32)
        torch.testing.assert_close(gate.weight, weight.float(), rtol=0, atol=0)
        self.assertIs(gate.e_score_correction_bias, correction_bias)

    def test_weight_reload_updates_fp32_routing(self):
        weight = nn.Parameter(torch.ones(1, 2, dtype=torch.bfloat16))
        correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        gate = make_fp32_gate(weight, correction_bias)
        with torch.no_grad():
            gate.weight.copy_(torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))
        logits, _ = gate(torch.tensor([[1.0, 0.5]], dtype=torch.bfloat16))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.item(), 3.5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_gate_preserves_sigmoid_rounding_for_all_bit_patterns(dtype):
    """Fusion must preserve sigmoid rounding, including subnormals and infinities."""
    device = current_platform.device_type
    gate = (
        torch.arange(65536, device=device, dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
        .reshape(16, 32, 128)
    )
    out = torch.ones(16, 4096, device=device, dtype=dtype)
    torch.testing.assert_close(
        apply_attention_gate(out, gate),
        out * gate.reshape(out.shape).sigmoid(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,heads", [(0, 32), (1, 32), (8, 16), (128, 8)])
def test_attention_gate_reads_interleaved_qkv_without_mutating_inputs(
    dtype, tokens, heads
):
    torch.manual_seed(0)
    width = heads * 128
    qkv = torch.randn(
        tokens, 2 * width + 512, device=current_platform.device_type, dtype=dtype
    )
    gate = qkv[:, : 2 * width].reshape(tokens, heads, 256)[:, :, 128:]
    out = torch.randn(tokens, 2 * width, device=qkv.device, dtype=dtype)[:, ::2]
    original_qkv, original_out = qkv.clone(), out.clone()
    expected = out * gate.reshape(out.shape).sigmoid()
    actual = apply_attention_gate(out, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(qkv, original_qkv, rtol=0, atol=0)
    torch.testing.assert_close(out, original_out, rtol=0, atol=0)
    if out.is_cuda and tokens:
        # Replay must read the updated buffers rather than captured values.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_out = apply_attention_gate(out, gate)
        qkv.normal_()
        out.normal_()
        graph.replay()
        torch.testing.assert_close(
            replay_out, out * gate.reshape(out.shape).sigmoid(), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "architecture", ["MiniCPM5MoEForCausalLM", "MiniCPM5MoeForCausalLM"]
)
def test_minicpm5_defaults_to_bcg_without_compilation(
    tmp_path, monkeypatch, architecture
):
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    MiniCPM5MoEConfig(architectures=[architecture]).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=4096,
            dtype="bfloat16",
        )
    )
    assert (
        config.compilation_config.use_breakable_cudagraph == current_platform.is_cuda()
    )
    if current_platform.is_cuda():
        assert config.compilation_config.mode == CompilationMode.NONE
        assert (
            config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,heads", [(1, 32), (8, 16), (128, 8)])
def test_interleaved_rope_matches_original_and_preserves_gate(
    default_vllm_config, dtype, tokens, heads
):
    from vllm import ir
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.models.minicpm5_moe.nvidia.attention import apply_interleaved_rope

    torch.manual_seed(0)
    device = current_platform.device_type
    width = heads * 128
    rope = RotaryEmbedding(128, 128, 8192, 5000000, True, dtype).to(device)
    qkv = torch.randn(tokens, 2 * width + 512, device=device, dtype=dtype)
    query = qkv[:, : 2 * width].reshape(tokens, heads, 256)[:, :, :128]
    key = qkv[:, 2 * width : 2 * width + 256]
    gate = qkv[:, : 2 * width].reshape(tokens, heads, 256)[:, :, 128:]
    original_gate = gate.clone()
    positions = torch.randint(0, 8192, (tokens,), device=device)
    providers = ["vllm_c", "native"] if current_platform.is_cuda() else ["native"]
    with ir.ops.rotary_embedding.set_priority(providers):
        expected = rope(positions, query.flatten(1).clone(), key.clone())
        actual = apply_interleaved_rope(positions, query, key, rope)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(gate, original_gate, rtol=0, atol=0)
    assert actual[0].is_contiguous()


@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_router_loads_checkpoint_into_fp32_parameters(dtype, tokens):
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
        finalize_model_weights,
    )

    torch.manual_seed(0)
    device = current_platform.device_type
    weight = nn.Parameter(
        torch.empty(160, 2048, device=device, dtype=dtype), requires_grad=False
    )
    gate = make_fp32_gate(
        weight, nn.Parameter(torch.zeros(160, device=device), requires_grad=False)
    )
    hidden = torch.randn(tokens, 2048, device=device, dtype=dtype)
    checkpoint = torch.randn(160, 2048, device=device, dtype=dtype)
    default_weight_loader(gate.weight, checkpoint)
    assert gate.weight.dtype == torch.float32
    torch.testing.assert_close(gate.weight, checkpoint.float(), rtol=0, atol=0)
    expected = gate(hidden)[0]
    finalize_model_weights(gate)
    actual = gate(hidden)[0]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(gate.state_dict()) == {"weight", "e_score_correction_bias"}


@pytest.mark.usefixtures("dist_init")
def test_router_reload_updates_captured_fp32_parameter():
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
        finalize_model_weights,
    )

    device = current_platform.device_type
    weight = nn.Parameter(
        torch.ones(2, 4, device=device, dtype=torch.bfloat16), requires_grad=False
    )
    gate = make_fp32_gate(
        weight, nn.Parameter(torch.zeros(2, device=device), requires_grad=False)
    )
    weight = gate.weight
    hidden = torch.tensor([[1, 1 / 256, -2, 3]], device=device, dtype=torch.bfloat16)
    finalize_model_weights(gate)
    before = gate(hidden)[0]
    if hidden.is_cuda:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gate(hidden)[0]
    # The real loader writes through .data; its updates bypass _version.
    default_weight_loader(weight, torch.arange(8, device=device).reshape(2, 4))
    finalize_model_weights(gate)
    if hidden.is_cuda:
        graph.replay()
    else:
        actual = gate(hidden)[0]
    expected = torch.nn.functional.linear(hidden.float(), weight.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, before)


if __name__ == "__main__":
    unittest.main()
