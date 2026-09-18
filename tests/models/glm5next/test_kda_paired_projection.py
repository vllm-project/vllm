# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve real KDA projection outputs under pairing, refit, and LoRA.

The input is merged q|k|v|b|f_a|g_a and the output is two gate projections.
Guard against bypassing adapters or replaying stale weights after a refit.
Real single-rank layers and the existing LoRA wrapper are the cheapest path;
attention state and a full model are unnecessary for these projection contracts.
"""

from contextlib import contextmanager

import pytest
import torch
from transformers import Glm5NextConfig, Glm5NextTextConfig

from vllm.config import LoRAConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.lora.layers import ColumnParallelLinearWithLoRA, LoRAMapping
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only BF16 GEMM"
)

HEAD_DIM = 128
LOCAL_NUM_HEADS = 8
LOCAL_PROJ_SIZE = HEAD_DIM * LOCAL_NUM_HEADS
F_A_OFFSET = 3 * LOCAL_PROJ_SIZE + LOCAL_NUM_HEADS
ROW_WIDTH = F_A_OFFSET + 2 * HEAD_DIM


@pytest.fixture
def layer_factory(tmp_path, dist_init):
    @contextmanager
    def create(dtype, *, enable_lora=False):
        config = Glm5NextTextConfig(
            architectures=["Glm5NextForCausalLM"],
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=8,
            vocab_size=256,
            max_position_embeddings=256,
            first_k_dense_replace=1,
            n_routed_experts=4,
            num_experts_per_tok=1,
            layer_types=["linear_attention"],
            linear_num_heads=LOCAL_NUM_HEADS,
            linear_head_dim=HEAD_DIM,
            linear_conv_kernel_dim=4,
            linear_lower_bound=-5.0,
            linear_attn_config={
                "num_heads": LOCAL_NUM_HEADS,
                "head_dim": HEAD_DIM,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        )
        Glm5NextConfig(
            architectures=["Glm5NextForConditionalGeneration"],
            text_config=config.to_dict(),
        ).save_pretrained(tmp_path)
        lora_config = (
            LoRAConfig(max_lora_rank=8, max_loras=1, lora_dtype=dtype)
            if enable_lora
            else None
        )
        vllm_config = VllmConfig(
            model_config=ModelConfig(
                model=str(tmp_path),
                dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
                max_model_len=256,
                skip_tokenizer_init=True,
                enforce_eager=True,
            ),
            lora_config=lora_config,
            additional_config={"kda_prefill_backend": "triton"},
        )
        with (
            set_current_vllm_config(vllm_config),
            torch.device("cuda"),
            set_default_torch_dtype(dtype),
        ):
            layer = Glm5NextLinearAttention(
                config, vllm_config, prefix="model.layers.0.linear_attn"
            )
            for parameter in layer.parameters():
                torch.nn.init.uniform_(parameter, -0.1, 0.1)
            yield layer, vllm_config

    return create


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [1, 5, 64, 256])
def test_paired_fg_projection_bit_identical(layer_factory, num_tokens: int):
    torch.manual_seed(0)
    with layer_factory(torch.bfloat16) as (layer, _):
        layer.process_weights_after_loading(torch.bfloat16)
        projected = torch.randn(num_tokens, ROW_WIDTH, dtype=torch.bfloat16)
        f_a = projected[:, F_A_OFFSET : F_A_OFFSET + HEAD_DIM]
        g_a = projected[:, F_A_OFFSET + HEAD_DIM : F_A_OFFSET + 2 * HEAD_DIM]
        g1, g2 = layer._paired_fg_projections(projected)
        assert torch.equal(g1, layer.f_b_proj(f_a)[0])
        assert torch.equal(g2, layer.g_b_proj(g_a)[0])


@torch.inference_mode()
def test_paired_fg_graph_replay_uses_refitted_weights(layer_factory):
    with layer_factory(torch.bfloat16) as (layer, _):
        layer.f_b_proj.weight.fill_(1 / HEAD_DIM)
        layer.g_b_proj.weight.fill_(2 / HEAD_DIM)
        layer.process_weights_after_loading(torch.bfloat16)
        projected = torch.ones(5, ROW_WIDTH, dtype=torch.bfloat16)
        f_a = projected[:, F_A_OFFSET : F_A_OFFSET + HEAD_DIM]
        g_a = projected[:, F_A_OFFSET + HEAD_DIM : F_A_OFFSET + 2 * HEAD_DIM]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer._paired_fg_projections(projected)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_f, actual_g = layer._paired_fg_projections(projected)
        # Keep the captured storage alive so allocator reuse cannot hide stale data.
        captured_weight = layer._paired_fg_weight
        layer.f_b_proj.weight.fill_(0.5 / HEAD_DIM)
        layer.g_b_proj.weight.fill_(1 / HEAD_DIM)
        layer.process_weights_after_loading(torch.bfloat16)
        graph.replay()
        torch.accelerator.synchronize()
        assert captured_weight is not None
        assert torch.equal(actual_f, layer.f_b_proj(f_a)[0])
        assert torch.equal(actual_g, layer.g_b_proj(g_a)[0])


@torch.inference_mode()
def test_paired_fg_weight_disabled_for_non_bf16(layer_factory):
    with layer_factory(torch.float32) as (layer, _):
        layer.process_weights_after_loading(torch.float32)
        assert layer._paired_fg_weight is None


@torch.inference_mode()
def test_lora_gate_projections_are_used_after_postload(layer_factory):
    with layer_factory(torch.bfloat16, enable_lora=True) as (layer, config):
        layer.in_proj_qkvbfg_a.weight.fill_(1 / 128)
        layer.f_b_proj.weight.zero_()
        layer.g_b_proj.weight.zero_()
        # The manager installs wrappers after the ordinary weight processing.
        layer.process_weights_after_loading(torch.bfloat16)
        punica = get_punica_wrapper(16, 1, "cuda", lora_config=config.lora_config)
        observed: dict[str, list[torch.Tensor]] = {"f_b_proj": [], "g_b_proj": []}
        handles = []
        for name, value in (("f_b_proj", 1), ("g_b_proj", 2)):
            wrapper = ColumnParallelLinearWithLoRA(layer.get_submodule(name))
            wrapper.create_lora_weights(1, config.lora_config)
            wrapper.set_mapping(punica)
            lora_a = torch.zeros(8, HEAD_DIM, dtype=torch.bfloat16)
            lora_b = torch.zeros(LOCAL_PROJ_SIZE, 8, dtype=torch.bfloat16)
            lora_a[0].fill_(1 / HEAD_DIM)
            lora_b[:, 0].fill_(value)
            wrapper.set_lora(0, lora_a, lora_b)
            layer.set_submodule(name, wrapper)

            def capture(module, inputs, output, name=name):
                observed[name].append(output[0].clone())

            handles.append(wrapper.register_forward_hook(capture))
        punica.update_metadata(
            LoRAMapping([1, 1, 1], [1], is_prefill=True), [1], 1, 256
        )
        try:
            # The real profiling path needs no recurrent state; gates run first.
            with set_forward_context(None, config, num_tokens=3):
                layer(torch.ones(3, 128, dtype=torch.bfloat16), torch.arange(3))
        finally:
            for handle in handles:
                handle.remove()
        for name, value in (("f_b_proj", 1), ("g_b_proj", 2)):
            assert len(observed[name]) == 1
            assert torch.equal(
                observed[name][0],
                torch.full((3, LOCAL_PROJ_SIZE), value, dtype=torch.bfloat16),
            )
