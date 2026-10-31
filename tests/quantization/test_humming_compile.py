# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

pytestmark = pytest.mark.skipif(
    not (has_humming() and current_platform.is_cuda()),
    reason="requires humming and CUDA",
)


def test_humming_input_quant_compiles_fullgraph():
    """Humming's forward helpers look up ``DeviceInfo`` on every call; vLLM
    traces them with ``fullgraph=True`` (vllm-project/humming#147)."""
    import vllm.utils.humming as hm

    weight_schema = hm.HummingWeightSchema(b_dtype="int4", weight_scale_group_size=128)
    input_schema = hm.HummingInputSchema(a_dtype="int8")
    from humming.transform import prepare_layer_config

    config = prepare_layer_config(
        shape_n=256,
        shape_k=256,
        weight_schema=weight_schema,
        input_schema=input_schema,
        torch_dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = torch.randn(16, 256, dtype=torch.bfloat16, device="cuda")

    def quant(t: torch.Tensor):
        return hm.may_process_input(config, t)

    expected = quant(x)
    actual = torch.compile(quant, fullgraph=True)(x)
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            torch.testing.assert_close(a, e)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80), reason="requires BF16 support"
)
@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.parametrize("num_groups", [1, 2, 4, 8])
def test_dsv41_grouped_wo_a_uses_humming(
    tmp_path, dist_init, monkeypatch, num_tokens, num_groups
):
    """WO-A must retain quantized groups through loading and inverse-RoPE."""
    from vllm.config import (
        KernelConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.kernels.linear.mxfp8.humming import (
        HummingMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {
                    "hidden_size": 256,
                    "num_attention_heads": 8 * num_groups,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 1,
                    "vocab_size": 256,
                    "head_dim": 512,
                    "qk_rope_head_dim": 64,
                    "q_lora_rank": 128,
                    "o_lora_rank": 128,
                    "o_groups": num_groups,
                    "sliding_window": 128,
                    "rms_norm_eps": 1e-6,
                    "max_position_embeddings": 64,
                    "rope_theta": 10000,
                    "rope_scaling": {"rope_type": "default"},
                    "compress_ratios": [0],
                },
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=64,
        ),
        kernel_config=KernelConfig(linear_backend="humming", enable_jit_warmup=False),
        quant_config=ModelOptMxFp8Config.from_config({"quant_method": "mxfp8"}),
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        attn = DeepseekV4FlashMLAAttention(config, prefix="model.layers.0.attn")
        torch.manual_seed(0)
        reference_weights = []
        for linear in (attn.wo_a, attn.wo_b):
            linear.weight.data.copy_(
                torch.rand_like(linear.weight, dtype=torch.bfloat16).to(
                    linear.weight.dtype
                )
            )
            linear.weight_scale.data.random_(125, 129)
            if linear is attn.wo_a:
                linear.weight_scale.data.view(num_groups, 128, -1).add_(
                    torch.arange(num_groups, dtype=torch.uint8).view(-1, 1, 1)
                )
            reference_weights.append(
                dequant_mxfp8_to_bf16(linear.weight, linear.weight_scale)
            )
            linear.quant_method.process_weights_after_loading(linear)
        assert isinstance(
            getattr(attn.wo_a.quant_method, "kernel", None), HummingMxfp8LinearKernel
        )
        assert attn.wo_a.weight.element_size() == 4  # Humming's packed int32 layout.

        assert attn.wo_a.is_bmm
        output = torch.rand(num_tokens, attn.padded_heads, 512)
        positions = torch.arange(num_tokens)
        live = output[:, : attn.n_local_heads].clone()
        cos, sin = attn.rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
        rope = live[..., -64:].float().reshape(num_tokens, attn.n_local_heads, 32, 2)
        rotated = torch.stack(
            (
                rope[..., 0] * cos[:, None] + rope[..., 1] * sin[:, None],
                rope[..., 1] * cos[:, None] - rope[..., 0] * sin[:, None],
            ),
            dim=-1,
        )
        live[..., -64:] = rotated.flatten(-2).to(live.dtype)
        grouped_input = live.reshape(num_tokens, num_groups, -1)
        grouped_weight = reference_weights[0].view(num_groups, 128, -1)
        projected = torch.bmm(
            grouped_input.transpose(0, 1), grouped_weight.transpose(1, 2)
        ).transpose(0, 1)
        actual_groups = attn.wo_a(grouped_input)
        torch.testing.assert_close(actual_groups, projected)
        expected = torch.nn.functional.linear(
            projected.flatten(1), reference_weights[1]
        )
        actual = attn._o_proj(output, positions)
        torch.testing.assert_close(actual, expected)
        compiled = torch.compile(attn._o_proj, fullgraph=True)(output, positions)
        torch.testing.assert_close(compiled, actual, rtol=0, atol=0)
