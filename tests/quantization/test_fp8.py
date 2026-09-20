# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""

import logging
from types import SimpleNamespace

import pytest
import regex as re
import torch
from packaging.version import Version

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm import _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.config.kernel import KernelConfig
from vllm.config.model import ModelConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.linear.scaled_mm import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils import flashinfer_utils
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
)
from vllm.model_executor.model_loader.reload.trace import ModelReloadTracer
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type

MODELS = [
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    # The checkpoint below was removed from the HF.
    # TODO: add a small replacement checkpoint.
    pytest.param(
        "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
        marks=pytest.mark.skip(reason="Checkpoint removed from HF."),
    ),
]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="DSV4.1 requires CUDA")
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_deepseek_v41_mxfp8_scale_loading(
    dist_init, default_vllm_config, monkeypatch, tp_rank, scale_dtype
):
    """Expand linear scales before TP slicing, including padded shared experts."""
    from vllm.model_executor.layers import linear as linear_module
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.models.deepseek_v41 import quant_config as quant_module
    from vllm.models.deepseek_v41.nvidia import model as model_module

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    quant_config = quant_module.DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[32, 32]
    )
    quant_config._resolved_expert_dtype = "fp4"
    for module in (model_module, linear_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
    tp_args = dict(quant_config=quant_config, bias=False)
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.wq_b = ColumnParallelLinear(128, 128, **tp_args)
    block.attn.fused_wqa_wkv = MergedColumnParallelLinear(128, [128, 64], **tp_args)
    block.ffn = torch.nn.Module()
    block.ffn.shared_experts = torch.nn.Module()
    block.ffn.shared_experts.gate_up_proj = MergedColumnParallelLinear(
        128, [128, 128], **tp_args
    )
    block.ffn.shared_experts.down_proj = RowParallelLinear(128, 128, **tp_args)
    block.ffn.experts = torch.nn.Module()
    block.ffn.experts.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.empty(3, 4, dtype=torch.uint8), False)
    )
    block.engram = torch.nn.Module()
    block.engram.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(torch.empty(3, 4, dtype=scale_dtype), False),
    )

    def load_expert_scale(param, weight, *args, **kwargs):
        param.data.copy_(weight)
        return True

    block.ffn.experts.weight_scale.weight_loader = load_expert_scale
    model = model_module.DeepseekV4Model.__new__(model_module.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList([block])
    model.config = SimpleNamespace(num_attention_heads=2)
    model.quant_config = quant_config
    model.use_sequence_parallel = False
    model.get_expert_mapping = lambda: [
        ("experts.weight_scale", "experts.0.w1.weight_scale", 0, "w1")
    ]

    checkpoints = [
        ("attn.wq_b", "attn.wq_b", 128, 128, 0, None),
        ("attn.wq_a", "attn.fused_wqa_wkv", 128, 128, 0, slice(0, 64)),
        ("attn.wkv", "attn.fused_wqa_wkv", 64, 128, 0, slice(64, 96)),
        (
            "ffn.shared_experts.w1",
            "ffn.shared_experts.gate_up_proj",
            96,
            128,
            0,
            slice(0, 64),
        ),
        (
            "ffn.shared_experts.w3",
            "ffn.shared_experts.gate_up_proj",
            96,
            128,
            0,
            slice(64, 128),
        ),
        (
            "ffn.shared_experts.down_proj",
            "ffn.shared_experts.down_proj",
            128,
            96,
            1,
            None,
        ),
    ]
    weights = []
    expected = []
    for source, target, n, k, axis, shard in checkpoints:
        weight = torch.randint(-4, 5, (n, k)).to(torch.float8_e4m3fn)
        scale_bytes = torch.randint(124, 131, (n // 32, k // 32), dtype=torch.uint8)
        weights.extend(
            [
                (f"layers.0.{source}.weight", weight),
                (f"layers.0.{source}.scale", scale_bytes.view(scale_dtype)),
            ]
        )
        dequant = (
            weight.float().reshape(n // 32, 32, k // 32, 32)
            * torch.exp2(scale_bytes.float() - 127)[:, None, :, None]
        ).reshape(n, k)
        if "shared_experts" in source:
            padded = torch.zeros(128, 128)
            padded[:n, :k] = dequant
            dequant = padded
        expected.append((target, shard, dequant.chunk(2, dim=axis)[tp_rank]))

    expert_scale = torch.full((3, 4), 125, dtype=torch.uint8)
    weights.append(
        ("layers.0.ffn.experts.0.w1.weight_scale", expert_scale.view(scale_dtype))
    )
    weights.append(("layers.0.engram.weight_scale_inv", expert_scale.view(scale_dtype)))
    mapper = model_module._make_deepseek_v4_weights_mapper("fp4", "weight_scale")
    loaded = model.load_weights(
        (name.removeprefix("model."), weight) for name, weight in mapper.apply(weights)
    )
    assert "layers.0.ffn.experts.weight_scale" in loaded
    assert torch.equal(block.ffn.experts.weight_scale, expert_scale)
    assert torch.equal(block.engram.weight_scale_inv.view(torch.uint8), expert_scale)
    for target, shard, reference in expected:
        linear = block.get_submodule(target)
        assert isinstance(linear.quant_method, ModelOptLinearMethod)
        assert f"layers.0.{target}.weight_scale" in loaded
        weight = linear.weight if shard is None else linear.weight[shard]
        scale = linear.weight_scale if shard is None else linear.weight_scale[shard]
        actual = (
            weight.float().unflatten(-1, (-1, 32))
            * torch.exp2(scale.float() - 127).unsqueeze(-1)
        ).flatten(-2)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="DSV4.1 requires CUDA")
@pytest.mark.parametrize(
    "weight_block_size,expert_dtype,scale_name",
    [
        ([32, 32], "fp4", "weight_scale"),
        ([128, 128], "fp4", "weight_scale_inv"),
        ([32, 32], "fp8", "weight_scale_inv"),
    ],
)
def test_deepseek_v41_vl_mapper_routes_linear_scales(
    weight_block_size, expert_dtype, scale_name
):
    """The VL wrapper must map ``.scale`` keys to the parameter the linear
    quant method registers, as the text model does. A hardcoded
    ``weight_scale_inv`` raised KeyError for native MXFP8 checkpoints."""
    from vllm.models.deepseek_v41.nvidia import model as model_module
    from vllm.models.deepseek_v41.nvidia import vl_model as vl_module

    vllm_config = SimpleNamespace(
        quant_config=SimpleNamespace(weight_block_size=weight_block_size)
    )
    resolved = model_module._linear_scale_param_name(vllm_config, expert_dtype)
    assert resolved == scale_name

    mapper = vl_module._make_deepseek_v4_vl_weights_mapper(expert_dtype, resolved)
    weight = torch.empty(0)
    mapped = [
        name
        for name, _ in mapper.apply(
            [("layers.0.attn.wq_a.scale", weight), ("layers.0.attn.wkv.weight", weight)]
        )
    ]
    assert mapped == [
        f"language_model.model.layers.0.attn.wq_a.{scale_name}",
        "language_model.model.layers.0.attn.wkv.weight",
    ]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="DeepGEMM requires CUDA")
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
@pytest.mark.parametrize(
    "weight_shape,is_bmm",
    [
        ((129, 160), False),
        ((387, 160), True),
        ((3, 129, 160), False),
        ((3, 128, 512), False),
    ],
)
def test_deepgemm_mxfp8_preserves_weight_and_scale_values(
    scale_dtype, weight_shape, is_bmm
):
    """DeepGEMM layout conversion preserves native weights and E8M0 scales."""
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported() or not current_platform.is_device_capability_family(
        100
    ):
        pytest.skip("DeepGEMM MXFP8 requires Blackwell")

    weight = torch.randn(weight_shape, device="cuda").to(torch.float8_e4m3fn)
    scales = (
        torch.randint(
            1,
            255,
            (weight.numel() // 32 + 1,),
            device="cuda",
            dtype=torch.uint8,
        )[1:]
        .view(*weight_shape[:-1], weight_shape[-1] // 32)
        .view(scale_dtype)
    )
    original_weight = weight.view(torch.uint8).clone()
    original_scales = scales.view(torch.uint8).clone()
    processed_weight, packed = fp8_utils.deepgemm_post_process_fp8_weight_block(
        weight,
        scales,
        quant_block_shape=(1, 32),
        use_e8m0=True,
        is_bmm=is_bmm,
        bmm_batch_size=3 if is_bmm else 0,
    )
    assert packed.dtype == torch.int32
    unpacked = torch.stack(
        [(packed >> (8 * i)) & 0xFF for i in range(4)], dim=-1
    ).flatten(-2)
    unpacked = unpacked[..., : scales.shape[-1]].to(torch.uint8)
    torch.testing.assert_close(
        unpacked.reshape_as(original_scales), original_scales, rtol=0, atol=0
    )
    torch.testing.assert_close(
        processed_weight.view(torch.uint8).reshape_as(original_weight),
        original_weight,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="DeepGEMM requires CUDA")
@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("config_source", ["deepseek", "mxfp8"])
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
def test_mxfp8_bmm_loads_and_projects_grouped_weights(
    dist_init, default_vllm_config, prequantized, config_source, num_tokens
):
    """BMM metadata set after construction selects grouped weight processing."""
    from vllm.model_executor.kernels.linear.mxfp8.deep_gemm import (
        DeepGemmMxfp8BmmLinearKernel,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model
    from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported() or not current_platform.is_device_capability_family(
        100
    ):
        pytest.skip("DeepGEMM MXFP8 BMM requires Blackwell")

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[32, 32]
    )
    quant_config._resolved_expert_dtype = "fp4"
    if config_source == "mxfp8":
        quant_config = get_quantization_config("mxfp8").from_config(
            {"quant_method": "mxfp8"}
        )
    with torch.device("cuda"):
        linear = ColumnParallelLinear(
            512,
            256,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
        )
    linear.is_bmm = True
    linear.bmm_batch_size = 2
    weight = torch.randn(256, 512, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(124, 131, (8, 16), device="cuda", dtype=torch.uint8)
    checkpoint_scales = (
        scales.view(torch.float8_e8m0fnu)
        if config_source == "deepseek"
        else scales.repeat_interleave(32, dim=0)
    )
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    torch.nn.Module.__init__(model)
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.wo_a = linear
    model.layers = torch.nn.ModuleList([block])
    model.config = SimpleNamespace(num_attention_heads=2)
    model.quant_config = quant_config
    model.use_sequence_parallel = False
    model.get_expert_mapping = lambda: []
    model.load_weights(
        [
            ("layers.0.attn.wo_a.weight", weight),
            ("layers.0.attn.wo_a.weight_scale", checkpoint_scales),
        ]
    )
    reference_weight = (
        weight.float().reshape(8, 32, 16, 32)
        * torch.exp2(scales.float() - 127)[:, None, :, None]
    ).reshape(2, 128, 512)
    linear.quant_method.process_weights_after_loading(linear)
    linear.quant_method.process_weights_after_loading(linear)
    assert isinstance(linear.quant_method.kernel, DeepGemmMxfp8BmmLinearKernel)
    assert linear.weight.shape == (2, 128, 512)
    torch.testing.assert_close(
        linear.weight.view(torch.uint8).flatten(),
        weight.view(torch.uint8).flatten(),
        rtol=0,
        atol=0,
    )
    unpacked_scales = torch.stack(
        [(linear.weight_scale >> (8 * i)) & 0xFF for i in range(4)], dim=-1
    ).flatten(-2)
    torch.testing.assert_close(
        unpacked_scales.to(torch.uint8).reshape(256, 16),
        scales.repeat_interleave(32, dim=0),
        rtol=0,
        atol=0,
    )

    x = torch.randn(2, num_tokens, 512, device="cuda", dtype=torch.bfloat16)
    reference = torch.einsum("gmk,gnk->mgn", x.float(), reference_weight)
    inputs = x.transpose(0, 1)
    if prequantized:
        cache = torch.cat(
            (
                torch.ones(num_tokens, 32, device="cuda"),
                torch.zeros(num_tokens, 32, device="cuda"),
            ),
            dim=1,
        )
        inputs = fused_inv_rope_fp8_quant(
            x.permute(1, 0, 2),
            torch.arange(num_tokens, device="cuda"),
            cache,
            n_groups=2,
            heads_per_group=1,
            nope_dim=448,
            rope_dim=64,
            quant_group_size=32,
            tma_aligned_scales=current_platform.has_device_capability(100),
        )
    with torch.no_grad():
        output = linear(inputs)
    assert output.shape == reference.shape
    assert (output.float() - reference).norm() / reference.norm() < 0.06
    if prequantized:
        recipe, tma_aligned_scales = compute_fp8_einsum_recipe(block_size=32)
        projected = deep_gemm_fp8_o_proj(
            x.permute(1, 0, 2),
            torch.arange(num_tokens, device="cuda"),
            cache,
            linear,
            torch.nn.Identity(),
            n_groups=2,
            heads_per_group=1,
            nope_dim=448,
            rope_dim=64,
            o_lora_rank=128,
            einsum_recipe=recipe,
            tma_aligned_scales=tma_aligned_scales,
        )
        torch.testing.assert_close(projected, output.flatten(1), rtol=0, atol=0)
    compiled = torch.compile(linear, backend="eager", fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(inputs), output, rtol=0, atol=0)


def test_prepare_gated_trtllm_fp8_moe_weights_pads_each_projection(monkeypatch):
    monkeypatch.setattr(
        flashinfer_utils,
        "rotate_weights_for_fi_trtllm_fp8_per_tensor_moe",
        lambda *args: None,
    )
    intermediate = 17
    padded_intermediate = 32
    hidden_size = 4
    gate = torch.ones((1, intermediate, hidden_size), dtype=torch.float8_e4m3fn)
    up = torch.full_like(gate, 2)
    w13 = torch.cat((gate, up), dim=1)
    w2 = torch.ones((1, hidden_size, intermediate), dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=True),
        moe_config=SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=intermediate,
        ),
    )

    padded_w31, _, _, _ = prepare_fp8_moe_layer_for_fi(
        layer,
        w13,
        w2,
        w13_scale=torch.ones(1),
        w13_input_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        w2_input_scale=torch.ones(1),
        is_trtllm=True,
    )

    expected = w13.new_zeros((1, 2 * padded_intermediate, hidden_size))
    expected[:, :intermediate] = up
    expected[:, padded_intermediate : padded_intermediate + intermediate] = gate
    assert layer.moe_config.intermediate_size_per_partition == padded_intermediate
    assert torch.equal(padded_w31, expected)


def test_static_fp8_moe_input_scales_remain_scalar() -> None:
    a1_scale, a2_scale = process_fp8_input_tensor_strategy_moe(
        torch.tensor([0.25, 0.5]),
        torch.tensor([0.75, 0.6]),
        enable_eplb=False,
    )

    assert a1_scale.ndim == a2_scale.ndim == 0


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    "force_marlin", [True, False] if current_platform.is_cuda() else [False]
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_model_load_and_run(
    model_id: str,
    force_marlin: bool,
    use_rocm_aiter: bool,
    monkeypatch,
    dist_init,
    workspace_init,
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    kernel_config = KernelConfig(
        linear_backend="marlin" if force_marlin else "auto",
        moe_backend="marlin" if force_marlin else "auto",
    )
    model, vllm_config = load_model_without_vllm_runner(
        model_id,
        model_config_kwargs={"hf_overrides": {"num_hidden_layers": 3}},
        vllm_config_kwargs={"kernel_config": kernel_config},
    )
    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        model(input_ids, positions, None)


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "force_marlin", [True, False] if current_platform.is_cuda() else [False]
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_quantization(
    vllm_runner,
    kv_cache_dtype: str,
    force_marlin: bool,
    use_rocm_aiter: bool,
    monkeypatch,
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    kwargs = {}
    if force_marlin:
        kwargs["linear_backend"] = "marlin"
        kwargs["moe_backend"] = "marlin"

    model_dtype = "auto"
    if kv_cache_dtype == "fp8" and current_platform.is_device_capability_family(90):
        # FA3 requires BF16 output when the query input is FP8.
        model_dtype = "bfloat16"

    with vllm_runner(
        "facebook/opt-125m",
        quantization="fp8",
        dtype=model_dtype,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    ) as llm:

        def check_model(model):
            fc1 = model.model.decoder.layers[0].fc1
            assert isinstance(fc1.quant_method, Fp8PerTensorOnlineLinearMethod)
            if kv_cache_dtype == "fp8":
                attn = model.model.decoder.layers[0].self_attn.attn
                assert isinstance(attn.quant_method, Fp8KVCacheMethod)
                assert attn._k_scale == 1.0
                assert attn._v_scale == 1.0

            if current_platform.is_cuda() or current_platform.is_xpu():
                if current_platform.supports_fp8() and not force_marlin:
                    # For GPUs with hardware support, we keep weights in fp8
                    assert fc1.weight.dtype == torch.float8_e4m3fn
                    assert not isinstance(
                        fc1.quant_method.fp8_linear, MarlinFP8ScaledMMLinearKernel
                    )
                else:
                    # For GPUs without hardware support, we pack the fp8 weights
                    # for weight-only quantization using Marlin kernels
                    assert fc1.weight.dtype == torch.int32
                    assert isinstance(
                        fc1.quant_method.fp8_linear, MarlinFP8ScaledMMLinearKernel
                    )
            elif current_platform.is_rocm():
                if current_platform.supports_fp8() and not force_marlin:
                    # For GPUs with hardware support, we keep weights in fp8
                    assert fc1.weight.dtype == current_platform.fp8_dtype()
                else:  # unsupported ROCm platform
                    pytest.skip(
                        "Skip `test_load_fp16_model`. "
                        "It only runs on ROCm platform with FP8 compute."
                        " e.g. MI300X and above."
                    )
            else:  # unsupported platform
                pytest.skip(
                    "Skip `test_load_fp16_model`. "
                    "It only runs on CUDA and ROCm platform."
                )

        llm.apply_model(check_model)

        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quant_peak_mem(
    vllm_runner,
    caplog_mp_spawn,
    monkeypatch,
) -> None:
    # Note: `allenai/OLMoE-1B-7B-0125-Instruct` was selected because:
    # 1. it covers both Linear and MoE paths
    # 2. it is already used by other tests in CI, so adding it here
    #    does not increase disk space for CI runners
    # I really wanted to use `ibm-granite/granite-3.0-1b-a400m-base`
    # which I think is the smallest MoE model in vLLM (2.5 GiB bf16,
    # 1.3 GiB fp8), but could not as adding one more model makes CI
    # run out of disk space.
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"

    # Force spawn to ensure caplog_mp_spawn works consistently
    # (it relies on VLLM_LOGGING_CONFIG_PATH which spawn reads but fork ignores)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    with (
        caplog_mp_spawn(logging.DEBUG) as log_holder,
        vllm_runner(
            model_name,
            quantization="fp8",
            enforce_eager=True,
        ) as llm,
    ):
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])

    log_text = log_holder.text

    # Parse memory usage from captured logs
    model_memory_gib = None
    peak_memory_gib = None
    for line in log_text.splitlines():
        if model_memory_gib is None:
            match = re.search(r"Model loading took ([\d.]+) GiB memory", line)
            if match:
                model_memory_gib = float(match.group(1))
        if peak_memory_gib is None:
            match = re.search(
                r"Peak GPU memory after loading weights: ([\d.]+) GiB", line
            )
            if match:
                peak_memory_gib = float(match.group(1))

    assert model_memory_gib is not None, "Could not find model loading memory log"
    assert peak_memory_gib is not None, "Could not find peak memory log"
    print(f"GPU memory used after loading weights: {model_memory_gib} GiB")
    print(f"Peak GPU memory usage while loading weights: {peak_memory_gib} GiB")

    # model specific, allenai/OLMoE-1B-7B-0125-Instruct fp8 online quant
    # uses 6.65 GiB for weight loading (bf16 checkpoint is ~12.89 GiB)
    expected_model_memory_gib = 6.7

    # for allenai/OLMoE-1B-7B-0125-Instruct the number we see today is 9.06
    # GiB, which is 1.36x above model_memory_gib. A slightly higher number is
    # expected as when we load and quantize weights in a streaming fashion we
    # need to have individual weights in bf16 + fp8 alive at the same time.
    expected_peak_memory_gib = expected_model_memory_gib * 1.4

    assert model_memory_gib < expected_model_memory_gib, (
        f"{model_memory_gib=} higher than {expected_model_memory_gib}"
    )
    assert peak_memory_gib < expected_peak_memory_gib, (
        f"{peak_memory_gib=} higher than {expected_peak_memory_gib}"
    )


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quant_load_format_dummy(
    vllm_runner,
    monkeypatch,
    caplog,
) -> None:
    with vllm_runner(
        "ibm-granite/granite-3.0-1b-a400m-base",
        quantization="fp8",
        enforce_eager=True,
        load_format="dummy",
    ) as llm:
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:
    def quantize_ref(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(current_platform.fp8_dtype())
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(current_platform.fp8_dtype())
        return qweight

    def per_tensor_dequantize(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight

    # Note that we use a shape % 4 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 4.
    x = (torch.randn(size=(11, 11), device=DEVICE_TYPE) * 13).to(dtype)

    # Dynamic quantization
    ref_y, inv_scale = ops.scaled_fp8_quant(x, None)
    ref_y = per_tensor_dequantize(ref_y, inv_scale, dtype)

    # Reference dynamic quantization
    y = quantize_ref(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Static quantization
    y, _ = ops.scaled_fp8_quant(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Padding
    y, _ = ops.scaled_fp8_quant(x, inv_scale, num_token_padding=17)
    assert y.shape[0] == 17
    torch.testing.assert_close(
        ref_y,
        per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale, dtype),
    )

    # non-contiguous input with padding
    m, n, padded_stride = 975, 512, 576
    padded_tensor = (torch.randn(size=(m, padded_stride), device=DEVICE_TYPE) * 13).to(
        dtype
    )
    x_nc = padded_tensor[:, :n]  # shape (m, n) with stride (padded_stride, 1)

    assert not x_nc.is_contiguous()
    assert x_nc.stride(0) == padded_stride

    # dynamic quantization
    ref_y_nc, inv_scale_nc = ops.scaled_fp8_quant(x_nc, None)
    ref_y_nc = per_tensor_dequantize(ref_y_nc, inv_scale_nc, dtype)

    # reference dynamic quantization
    y_nc = quantize_ref(x_nc, inv_scale_nc)
    torch.testing.assert_close(
        ref_y_nc, per_tensor_dequantize(y_nc, inv_scale_nc, dtype)
    )

    # static quantization
    y_nc, _ = ops.scaled_fp8_quant(x_nc, inv_scale_nc)
    torch.testing.assert_close(
        ref_y_nc, per_tensor_dequantize(y_nc, inv_scale_nc, dtype)
    )

    # padding after non-contiguous input quantization
    y_nc_pad, _ = ops.scaled_fp8_quant(x_nc, inv_scale_nc, num_token_padding=m + 10)
    assert y_nc_pad.shape[0] == m + 10
    torch.testing.assert_close(
        ref_y_nc,
        per_tensor_dequantize(
            torch.narrow(y_nc_pad, 0, 0, x_nc.shape[0]), inv_scale_nc, dtype
        ),
    )


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="FP8 e4m3fn weight reloading is not supported on e4m3fnuz platforms",
)
@pytest.mark.parametrize("method_cls", [Fp8LinearMethod, Fp8MoEMethod])
# FP8 weight reloading does not support online quantization
@pytest.mark.parametrize("is_checkpoint_fp8_serialized", [True])  # skip False
@pytest.mark.parametrize("weight_block_size", [None, [128, 128]])
# any postprocessing that is applied to the weights such as padding and repacking
# (excluding device sharding) must also be applied to the reloaded weights
#
# this is the case for marlin as well as per-tensor Fp8MoEMethod
@pytest.mark.parametrize("use_marlin", [False])  # skip True
def test_fp8_reloading(
    default_vllm_config,
    method_cls,
    is_checkpoint_fp8_serialized,
    weight_block_size,
    use_marlin,
    dist_init,
    monkeypatch,
):
    # NOTE(rob): this test fails when using DeepGEMM because the
    # shapes are invalid. Previously the test was passing because
    # we set fp8_backend to None, which sidestepped the issue.
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")

    if is_checkpoint_fp8_serialized is False:
        pytest.skip("FP8 weight reloading does not support online quantization")

    if method_cls is Fp8MoEMethod and weight_block_size is None:
        pytest.skip(
            "FP8 Tensor weight reloading does not support fusing w13_weight_scale. "
            "If this is your use case, consider using a restore function like #26327"
        )

    # Set model config as model_config.dtype is required in Fp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.moe_backend = "triton"
    layer_size = 128 if weight_block_size is not None else 1
    with torch.device(f"{DEVICE_TYPE}:0"):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            weight_block_size=weight_block_size,
        )

        if method_cls is Fp8LinearMethod:
            layer = torch.nn.Linear(layer_size, layer_size)
            method = method_cls(config)
            method.create_weights(
                layer=layer,
                input_size_per_partition=layer_size,
                output_partition_sizes=[layer_size],
                input_size=layer_size,
                output_size=layer_size,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )
            method.use_marlin = use_marlin

        else:
            layer = FusedMoEFactory(
                num_experts=1,
                top_k=1,
                hidden_size=layer_size,
                intermediate_size=layer_size,
            )
            layer = layer.routed_experts
            method = method_cls(config, layer)
            method.create_weights(
                layer=layer,
                num_experts=1,
                hidden_size=layer_size,
                intermediate_size_per_partition=layer_size,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )

    # capture weights format during loading
    original_metadata = [
        (name, param.shape, getattr(param, "weight_loader", default_weight_loader))
        for name, param in layer.named_parameters()
    ]

    # test loading
    for name, shape, _ in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)

    # test reloading works after loading
    for name, shape, _ in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA FP8 kernels")
@pytest.mark.parametrize(
    "backend,moe,block,eplb",
    [
        ("deep_gemm", False, True, False),
        ("cutlass", False, True, False),
        ("triton", False, True, False),
        ("torch", False, True, False),
        ("cutlass", False, False, False),
        ("flashinfer", False, False, False),
        ("torch", False, False, False),
        ("marlin", False, True, False),
        ("marlin", False, False, False),
        ("deep_gemm", True, True, False),
        ("triton", True, True, False),
        ("triton", True, False, False),
        ("marlin", True, True, False),
        ("marlin", True, False, False),
        ("triton", True, True, True),
        ("flashinfer_cutlass", True, True, False),
        ("flashinfer_cutlass", True, False, False),
        ("deep_gemm", True, True, True),
        ("flashinfer_cutlass", True, True, True),
        ("flashinfer_cutlass", True, False, True),
    ],
)
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_matches_cold_load(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    backend,
    moe,
    block,
    eplb,
    preserve,
    check_forward=True,
    e8m0_scales=False,
    ragged=False,
    is_bmm=False,
    marlin_static=False,
    legacy_reference=False,
    capture_graph=True,
    cold_events=None,
):
    """Real backend conversion must match cold load without replacing targets."""
    from transformers import LlamaConfig

    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
        AiterFp8BlockScaledMMKernel,
        AiterPreshuffledFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.b12x import (
        B12xFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
        CPUFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFp8BlockScaledMMKernel,
        CutlassFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
        DeepGemmFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        FlashInferFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.humming import (
        HummingFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
        BlockWiseTorchFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
        ROCmFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
        XPUFp8BlockScaledMMKernel,
        XPUW8A8FP8LinearKernel,
        XPUW8A16FP8LinearKernel,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
    from vllm.utils.torch_utils import set_default_torch_dtype

    if backend == "deep_gemm" and not is_deep_gemm_supported():
        pytest.skip("DeepGEMM is unavailable")
    if backend == "flashinfer_cutlass" and not has_flashinfer_cutlass_fused_moe():
        pytest.skip("FlashInfer CUTLASS is unavailable")
    if (
        backend == "torch"
        and not moe
        and block
        and check_forward
        and torch.version.cuda is not None
        and Version(torch.version.cuda) < Version("12.9")
    ):
        pytest.skip("PyTorch block-scaled GEMM requires a CUDA >= 12.9 build")
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=32,
    ).save_pretrained(tmp_path)
    default_vllm_config.model_config = ModelConfig(
        model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
    )
    if moe:
        # Exercise conversion even when the optional compute backend is absent.
        default_vllm_config.kernel_config.moe_backend = (
            "triton"
            if backend.startswith("batched_")
            or (
                backend
                in (
                    "flashinfer_trtllm",
                    "hpc",
                    "humming",
                    "cpu",
                    "vllm_cutlass",
                    "aiter",
                    "xpu",
                )
                and not check_forward
            )
            else backend
        )
    default_vllm_config.parallel_config.enable_eplb = eplb
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=(
            "dynamic"
            if block or (backend == "marlin" and not marlin_static)
            else "static"
        ),
        weight_block_size=[128, 128] if block else None,
    )
    if e8m0_scales:
        config.is_scale_e8m0 = True

    layer_index = 0

    def make_layer():
        nonlocal layer_index
        layer_index += 1
        with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
            if moe:
                layer = FusedMoEFactory(
                    num_experts=2,
                    top_k=1,
                    hidden_size=256,
                    intermediate_size=(
                        256
                        if block
                        or backend.startswith("batched_")
                        or backend in ("flashinfer_trtllm", "hpc")
                        else 260
                    ),
                    quant_config=config,
                    prefix=f"trace_{layer_index}.experts",
                    enable_eplb=eplb,
                ).routed_experts
                if eplb:
                    layer.eplb_state.logical_to_physical_map = torch.tensor(
                        [[0], [1]], device="cuda"
                    )
                method = layer.quant_method
                if backend == "flashinfer_trtllm" and not check_forward:
                    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (  # noqa: E501
                        TrtLlmFp8ExpertsMonolithic,
                    )
                    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                        Fp8MoeBackend,
                    )

                    method.fp8_backend = Fp8MoeBackend.FLASHINFER_TRTLLM
                    method.experts_cls = TrtLlmFp8ExpertsMonolithic
                elif backend == "hpc" and not check_forward:
                    from vllm.model_executor.layers.fused_moe.hpc_moe import HPCExperts
                    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                        Fp8MoeBackend,
                    )

                    method.fp8_backend = Fp8MoeBackend.HPC
                    method.experts_cls = HPCExperts
                elif backend == "humming" and not check_forward:
                    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                        Fp8MoeBackend,
                    )

                    method.fp8_backend = Fp8MoeBackend.HUMMING
                elif backend.startswith("batched_") or (
                    backend in ("cpu", "vllm_cutlass", "aiter", "xpu")
                    and not check_forward
                ):
                    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
                        Fp8MoeBackend,
                        backend_to_kernel_cls,
                    )

                    method.fp8_backend = {
                        "cpu": Fp8MoeBackend.CPU,
                        "vllm_cutlass": Fp8MoeBackend.VLLM_CUTLASS,
                        "batched_triton": Fp8MoeBackend.BATCHED_TRITON,
                        "batched_cutlass": Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
                        "aiter": Fp8MoeBackend.AITER,
                        "xpu": Fp8MoeBackend.XPU,
                    }[backend]
                    method.experts_cls = backend_to_kernel_cls(method.fp8_backend)[0]
            else:
                layer = torch.nn.Module()
                method = Fp8LinearMethod(config)
                widths = [128, 132] if backend == "cutlass" and not block else [256]
                if ragged:
                    widths = [272]
                method.create_weights(
                    layer,
                    256,
                    widths,
                    256,
                    sum(widths),
                    torch.bfloat16,
                    weight_loader=default_weight_loader,
                )
                kernel_cls = {
                    "deep_gemm": DeepGemmFp8BlockScaledMMKernel,
                    "cutlass": (
                        CutlassFp8BlockScaledMMKernel
                        if block
                        else CutlassFP8ScaledMMLinearKernel
                    ),
                    "triton": TritonFp8BlockScaledMMKernel,
                    "torch": (
                        BlockWiseTorchFP8ScaledMMLinearKernel
                        if block
                        else PerTensorTorchFP8ScaledMMLinearKernel
                    ),
                    "flashinfer": FlashInferFP8ScaledMMLinearKernel,
                    "marlin": MarlinFP8ScaledMMLinearKernel,
                    "humming": HummingFP8ScaledMMLinearKernel,
                    "b12x": B12xFp8BlockScaledMMKernel,
                    "xpu": (
                        XPUFp8BlockScaledMMKernel if block else XPUW8A8FP8LinearKernel
                    ),
                    "xpu_w8a16": XPUW8A16FP8LinearKernel,
                    "aiter_block": AiterFp8BlockScaledMMKernel,
                    "aiter_direct": AiterPreshuffledFp8BlockScaledMMKernel,
                    "cpu_direct": CPUFp8BlockScaledMMKernel,
                    "rocm": ROCmFP8ScaledMMLinearKernel,
                }[backend]
                if block and backend not in ("marlin", "humming"):
                    method.fp8_linear = kernel_cls(method.fp8_linear.config)
                else:
                    with pytest.MonkeyPatch.context() as patch:
                        if (
                            backend in ("xpu", "xpu_w8a16", "rocm")
                            and not check_forward
                        ):
                            patch.setattr(
                                kernel_cls,
                                "is_supported",
                                classmethod(lambda cls, *args: (True, None)),
                            )
                        method.fp8_linear = kernel_cls(
                            method.fp8_linear.config,
                            (
                                "weight",
                                "weight_scale_inv" if block else "weight_scale",
                                "input_scale",
                                "input_scale_ub",
                            ),
                        )
                method.use_marlin = backend == "marlin"
                layer.quant_method = method
                if backend == "aiter_direct":
                    layer.skip_weight_relayout = True
                if backend == "cpu_direct":
                    layer._cpu_skip_gemm_dispatch = True
                if is_bmm:
                    layer.is_bmm = True
                    layer.bmm_batch_size = 2
                layer.bias = torch.nn.Parameter(
                    torch.zeros(sum(widths)), requires_grad=False
                )
                layer.bias.weight_loader = default_weight_loader
                if backend == "humming":
                    layer.output_partition_sizes = widths
                    layer.params_dtype = torch.bfloat16
                    layer.has_bias = True
        return layer, method

    def load(layer, sources):
        checkpoint = []
        for name, source in sources.items():
            param = getattr(layer, name)
            if not moe:
                param.weight_loader(param, source)
                continue
            for expert in range(2):
                shards = ("w1", "w3") if name.startswith("w13_") else ("w2",)
                pieces = (
                    source[expert].chunk(2, dim=0)
                    if len(shards) == 2 and source[expert].ndim > 0
                    else (source[expert],) * len(shards)
                )
                for shard, piece in zip(shards, pieces):
                    proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[
                        shard
                    ]
                    suffix = name.split("_", 1)[1]
                    checkpoint.append((f"{expert}.{proj}.{suffix}", piece))
        if moe:
            assert list(layer.load_weights(checkpoint))

    layer, method = make_layer()
    trace = ModelReloadTracer()
    trace.register_fp8("fp8", layer)
    state = trace.states["fp8"]
    source_a = {}
    for role in state.roles:
        param = getattr(layer, role)
        value = torch.rand(param.shape, dtype=torch.float32, device="cuda")
        source_a[role] = (
            (value * 4 - 2).to(param.dtype)
            if "scale" not in role
            else (value * 0.1 + 0.01).to(param.dtype)
        )
        if backend == "flashinfer_cutlass" and block and "scale" in role:
            source_a[role].view(-1)[0] = 1e-23
        if (
            moe
            and current_platform.is_fp8_fnuz()
            and "weight" in role
            and "scale" not in role
        ):
            source_a[role].view(torch.int8).view(-1)[0] = -128
    with trace.observe():
        load(layer, source_a)
    method.process_weights_after_loading(layer)
    trace.bind_runtime()
    identities = {
        name: (target.tensor, target.tensor.data_ptr())
        for name, target in state.targets.items()
    }
    prepare_calls = []
    tracks_preparation = hasattr(state.policy, "prepare_for_load")
    if tracks_preparation:
        prepare_for_load = state.policy.prepare_for_load

        def checked_prepare(current):
            assert not current.checkpoint
            prepare_for_load(current)
            prepare_calls.append(True)
            assert set(current.checkpoint) == set(current.roles)
            for role in current.roles:
                source = current.checkpoint[role]
                if moe and backend == "flashinfer_cutlass":
                    target = current.targets[role].tensor
                    expected_alias = not preserve and (
                        block or role in ("w13_weight", "w2_weight")
                    )
                    assert (
                        source.untyped_storage().data_ptr()
                        == target.untyped_storage().data_ptr()
                    ) == expected_alias
                if backend in ("marlin", "humming"):
                    target = current.targets.get(role)
                    aliases = target is not None and (
                        source.untyped_storage().data_ptr()
                        == target.tensor.untyped_storage().data_ptr()
                    )
                    if preserve or role in ("weight", "w13_weight", "w2_weight"):
                        assert not aliases
                    elif not moe and role == "bias":
                        assert aliases
                assert source.shape == current.metadata[role].shape
                assert source.stride() == current.metadata[role].stride()
                assert source.dtype == current.metadata[role].dtype
            for target in current.targets.values():
                target.validate()

        state.policy.prepare_for_load = checked_prepare
    kernel = method.moe_kernel if moe else method.fp8_linear
    processing_plan = getattr(method, "processing_plan", None)
    policy_plan = getattr(state.policy, "plan", None)
    if processing_plan is not None:

        def reject_kernel_reinitialization(*args, **kwargs):
            pytest.fail("Reload must not execute cold-only kernel initialization")

        method._init_moe_kernel = reject_kernel_reinitialization
        method._setup_kernel = reject_kernel_reinitialization
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    topk_ids = (torch.arange(128, device="cuda", dtype=torch.int32) % 2)[:, None]
    topk_weights = torch.ones(128, 1, device="cuda")

    def forward(method, layer):
        if moe:
            return method.apply(layer, x, topk_weights, topk_ids, None, None)
        return method.apply(layer, x, layer.bias)

    if check_forward:
        for _ in range(3):
            forward(method, layer)
        torch.accelerator.synchronize()
        if capture_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = forward(method, layer)

    for factor in (0.5, 1.5):
        source_b = {
            name: (value.float() * factor).to(value.dtype)
            for name, value in source_a.items()
        }
        reference, reference_method = make_layer()
        placement = [1, 0] if eplb and factor == 0.5 else [0, 1]
        if eplb:
            for target in (layer, reference):
                target.eplb_state.logical_to_physical_map.copy_(
                    torch.tensor(placement, device="cuda")[:, None]
                )
        load(reference, source_b)
        if legacy_reference:
            # Compare against the pre-split cold path, not plan against itself.
            reference_method._create_processing_plan = lambda _: None
        reference_method.process_weights_after_loading(reference)
        cold_calls_before = list(cold_events or ())
        with trace.round(preserve_checkpoint=preserve):
            load(layer, dict(reversed(list(source_b.items()))))
            assert state.complete
            assert bool(state.checkpoint) is preserve
        assert list(cold_events or ()) == cold_calls_before
        assert getattr(state.policy, "plan", None) is policy_plan
        if tracks_preparation:
            assert len(prepare_calls) == (1 if factor == 0.5 else 2)
        assert (method.moe_kernel if moe else method.fp8_linear) is kernel
        assert getattr(method, "processing_plan", None) is processing_plan
        if moe and backend == "aiter":
            assert layer.w13_weight.is_shuffled and layer.w2_weight.is_shuffled
        for name, target in state.targets.items():
            original, pointer = identities[name]
            assert target.resolve() is original
            assert original.data_ptr() == pointer
            if name == "humming_locks":
                expected = reference_method.fp8_linear.locks
            elif name.startswith("humming."):
                expected = getattr(reference, name.removeprefix("humming."))
            elif name in state.roles or name in (
                "workspace",
                "bmm_weight",
                "bmm_scale",
            ):
                expected = getattr(reference, state.runtime_names.get(name, name))
            elif name in ("_g1_alphas", "_g2_alphas", "_g1_scale_c"):
                expected = getattr(reference_method.moe_kernel.fused_experts, name)
            else:
                expected = getattr(reference_method.moe_quant_config, name)
            torch.testing.assert_close(
                target.tensor.reshape(-1).contiguous().view(torch.uint8),
                expected.reshape(-1).contiguous().view(torch.uint8),
                rtol=0,
                atol=0,
            )
        if check_forward:
            if capture_graph:
                graph.replay()
                actual_output = graph_output
            else:
                actual_output = forward(method, layer)
            torch.testing.assert_close(
                actual_output, forward(reference_method, reference), rtol=0, atol=0
            )
        if preserve:
            for name, source in source_b.items():
                expected_checkpoint = source[placement] if moe else source
                torch.testing.assert_close(
                    state.checkpoint[name].contiguous().view(torch.uint8),
                    expected_checkpoint.contiguous().view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
    if moe and backend == "aiter":
        from vllm.model_executor.model_loader.reload.trace import ReloadError

        layer.w13_weight.is_shuffled = False
        with pytest.raises(ReloadError, match="shuffled layout marker"):
            state.policy.validate(state)
        layer.w13_weight.is_shuffled = True
    if backend in ("marlin", "humming"):
        from vllm.model_executor.model_loader.reload.trace import ReloadError

        if backend == "humming" and moe:
            for mapping in (
                layer.humming_configs,
                layer.weight_schemas,
                layer.input_schemas,
            ):
                original = mapping["w13"]
                mapping["w13"] = object()
                with pytest.raises(ReloadError, match="configuration changed"):
                    state.policy.validate(state)
                mapping["w13"] = original
        owner, attribute = (
            (kernel, "processing_plan")
            if backend == "humming" and not moe
            else (layer, f"fp8_{backend}_processing_plan")
        )
        original_plan = getattr(owner, attribute)
        setattr(owner, attribute, object())
        with pytest.raises(ReloadError):
            state.policy.validate(state)
        setattr(owner, attribute, original_plan)


def test_fp8_processing_plan_cutlass_canonical_input():
    """Each round starts at W13; neither previous output nor a layer is needed."""
    from vllm.model_executor.layers.quantization.utils.fp8_processing import (
        Fp8MoEProcessingPlan,
        Fp8MoEWeights,
    )

    plan = Fp8MoEProcessingPlan(
        backend="flashinfer_cutlass",
        block_shape=(128, 128),
        is_act_and_mul=True,
        is_gated=True,
        shard_size=128,
        num_experts=1,
        static_input=False,
        enable_eplb=False,
        use_e8m0=False,
    )
    # Layout and scale clamping do not require a GPU kernel.
    for gate, up in ((1.0, 2.0), (3.0, 4.0)):
        w13 = torch.cat(
            (torch.full((1, 128, 128), gate), torch.full((1, 128, 128), up)), dim=1
        )
        source = Fp8MoEWeights(
            w13,
            torch.ones(1, 128, 128),
            torch.tensor([[[1e-23], [0.25]]]),
            torch.zeros(1, 1, 1),
            None,
            None,
        )
        result = plan.process(source)
        torch.testing.assert_close(result.w13[:, :128], torch.full((1, 128, 128), up))
        torch.testing.assert_close(result.w13[:, 128:], torch.full((1, 128, 128), gate))
        torch.testing.assert_close(
            result.w13_scale, torch.tensor([[[0.25], [1e-10]]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.w2_scale, torch.full((1, 1, 1), 1e-10), rtol=0, atol=0
        )
        assert result.w13_input_scale is None
        assert result.w2_input_scale is None


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_fp8_reload_trace_fnuz_moe_conversion(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    block,
    preserve,
    eplb,
):
    """Exercise the real FN-to-FNUZ conversion, not an AMD forward on CUDA."""
    monkeypatch.setattr(current_platform, "is_fp8_fnuz", lambda: True)
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "triton",
        True,
        block,
        eplb,
        preserve,
        check_forward=False,
    )


def _stub_marlin_processing_ops(monkeypatch):
    """Keep production conversion/workspace code; record cold-only allocation."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8

    cold_events = []
    make_workspace = marlin_utils_fp8.marlin_make_workspace_new

    def workspace(*args, **kwargs):
        cold_events.append("workspace")
        return make_workspace(*args, **kwargs)

    def repack(b_q_weight, size_k, size_n, num_bits):
        assert num_bits == 8
        return b_q_weight.reshape(size_k // 16, size_n * 4).roll(1, -1)

    monkeypatch.setattr(ops, "gptq_marlin_repack", repack)
    monkeypatch.setattr(marlin_utils_fp8, "marlin_make_workspace_new", workspace)
    return cold_events


@pytest.mark.parametrize("moe,eplb", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_marlin_packing_lifecycle(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    moe,
    eplb,
    block,
    preserve,
):
    """Keep real scale/bias conversion; replace only the native repack operator."""
    cold_events = _stub_marlin_processing_ops(monkeypatch)
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "marlin",
        moe,
        block,
        eplb,
        preserve,
        check_forward=False,
        cold_events=cold_events,
    )


@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_marlin_static_lifecycle(
    default_vllm_config, dist_init, workspace_init, tmp_path, monkeypatch, preserve
):
    """A static checkpoint scale is tracked even though W8A16 discards it."""
    cold_events = _stub_marlin_processing_ops(monkeypatch)
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "marlin",
        False,
        False,
        False,
        preserve,
        check_forward=False,
        marlin_static=True,
        cold_events=cold_events,
    )


@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_marlin_static(
    default_vllm_config, dist_init, workspace_init, tmp_path, preserve
):
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "marlin",
        False,
        False,
        False,
        preserve,
        marlin_static=True,
    )


def _stub_humming_processing_library(monkeypatch):
    """Exercise production plan creation and processing, not a mock layer hook."""
    from vllm.utils import humming

    cold_events = []

    class StandardSchema:
        pass

    standard = StandardSchema()

    class WeightSchema:
        def __init__(self, config):
            self.config = config

        @classmethod
        def from_config(cls, config):
            cold_events.append("weight_schema")
            return cls(config)

        def convert_humming(self, *, tensors, **kwargs):
            scale = tensors.get("weight_scale", tensors.get("weight_scale_inv"))
            weight = tensors["weight"]
            if (
                self.config.get("strategy") == "tensor"
                and len(kwargs["shape_n_stacks"]) == 2
                and kwargs.get("num_experts") is not None
            ):
                # The normalized W13 scale is repeated for the W1/W3 stacks,
                # not replaced by their different original checkpoint scales.
                assert scale.shape == (kwargs["num_experts"], 2)
                torch.testing.assert_close(scale[:, 0], scale[:, 1], rtol=0, atol=0)
            values = {
                "weight": weight.view(torch.int32),
                "weight_scale": scale.clone(),
            }
            if "bias" in tensors:
                values["bias"] = tensors["bias"]
            return standard, values

    class InputSchema:
        def __init__(self):
            cold_events.append("input_schema")

        @classmethod
        def from_config(cls, config):
            return cls()

        def convert_humming(self, **kwargs):
            return self, {}

    def prepare_config(**kwargs):
        cold_events.append("config")
        return SimpleNamespace(**kwargs)

    def transform(config, tensors):
        values = {
            "weight": tensors["weight"].roll(1, -1),
            "weight_scale": tensors["weight_scale"].clone().mul_(2),
            "weight_scale_2": tensors["weight_scale"].float().sum().reshape(1),
        }
        if "bias" in tensors:
            values["bias"] = tensors["bias"].clone()
        return values

    for name, value in {
        "BaseWeightSchema": WeightSchema,
        "BaseInputSchema": InputSchema,
        "HummingWeightSchema": StandardSchema,
        "HummingInputSchema": InputSchema,
        "prepare_layer_config": prepare_config,
        "transform_humming_tensors": transform,
    }.items():
        # The facade lazily imports optional modules; override without resolving
        # those imports so this test also works with an older Humming package.
        monkeypatch.setitem(humming.__dict__, name, value)
    return cold_events


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int32])
def test_fp8_humming_processing_plan_standardization(transpose, dtype):
    """Saved input dimensions match cold renaming for FP8 and packed inputs."""
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        convert_linear_layer_to_humming_standard,
    )

    weight = torch.arange(32).reshape(4, 8).to(dtype)
    scale = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    values = {
        "qweight": weight.t() if transpose else weight,
        "scales": scale.t() if transpose else scale,
    }
    layer = torch.nn.Module()
    for name, value in values.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
        param = getattr(layer, name)
        param.input_dim = 0 if transpose else 1
        param.output_dim = 1 if transpose else 0
    layout = convert_linear_layer_to_humming_standard(
        layer, {"weight": "qweight", "weight_scale": "scales"}
    )
    expected = {"weight": weight.view(torch.int32), "weight_scale": scale}
    replay = layout.process(values)
    for name, value in expected.items():
        torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)
        torch.testing.assert_close(replay[name], value, rtol=0, atol=0)
    assert not hasattr(layer, "qweight") and not hasattr(layer, "scales")


def test_fp8_humming_processing_plan_rejects_schema_drift(monkeypatch):
    """A changed schema must not reach the fixed-config transform."""
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        HummingTensorProcessingPlan,
    )
    from vllm.utils import humming

    def reject_transform(*args):
        pytest.fail("Changed schema reached the runtime transform")

    monkeypatch.setitem(humming.__dict__, "transform_humming_tensors", reject_transform)
    plan = HummingTensorProcessingPlan(
        source_schema=SimpleNamespace(convert_humming=lambda **kwargs: (object(), {})),
        weight_schema=object(),
        input_schema=object(),
        config=object(),
        shape_n_stacks=(4,),
        shape_k_stacks=(4,),
        param_dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="schema changed"):
        plan.process({})


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_fp8_reload_trace_humming_moe_lifecycle(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    block,
    preserve,
    eplb,
):
    """Exercise dynamic placement and derived tensors with cold-bound plans."""
    cold_events = _stub_humming_processing_library(monkeypatch)

    def init_kernel(method, layer):
        method.moe_quant_config = SimpleNamespace()
        method.moe_kernel = SimpleNamespace()

    monkeypatch.setattr(Fp8MoEMethod, "_init_moe_kernel", init_kernel)
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "humming",
        True,
        block,
        eplb,
        preserve,
        check_forward=False,
        cold_events=cold_events,
    )


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_fp8_reload_trace_humming_moe(
    default_vllm_config, dist_init, workspace_init, tmp_path, block, preserve, eplb
):
    from vllm.utils.import_utils import has_humming

    if not has_humming():
        pytest.skip("Humming is unavailable")
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "humming",
        True,
        block,
        eplb,
        preserve,
    )


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_humming_linear_lifecycle(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    block,
    preserve,
):
    """Test deleted/renamed roles without depending on Humming's native packing."""
    from vllm.utils.import_utils import has_humming

    if not has_humming():
        pytest.skip("Humming dtype definitions are unavailable")

    cold_events = _stub_humming_processing_library(monkeypatch)
    test_fp8_reload_trace_humming_linear(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        block,
        preserve,
        check_forward=False,
        cold_events=cold_events,
    )


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_humming_linear(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    block,
    preserve,
    check_forward=True,
    cold_events=None,
):
    from vllm.utils.import_utils import has_humming

    if not has_humming():
        pytest.skip("Humming is unavailable")
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "humming",
        False,
        block,
        False,
        preserve,
        check_forward=check_forward,
        cold_events=cold_events,
    )


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_cutlass_conversion(
    default_vllm_config, dist_init, workspace_init, tmp_path, block, eplb, preserve
):
    """Check real CUDA conversion/storage independently of CUTLASS forward JIT."""
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "flashinfer_cutlass",
        True,
        block,
        eplb,
        preserve,
        check_forward=False,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA tensors")
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_torch_block_conversion(
    default_vllm_config, dist_init, workspace_init, tmp_path, preserve
):
    """Validate reload even when the CUDA build lacks block-scaled torch GEMM."""
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "torch",
        False,
        True,
        False,
        preserve,
        check_forward=False,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA tensors")
@pytest.mark.parametrize(
    "backend,block,fnuz,e8m0_scales",
    [
        ("aiter_block", True, False, False),
        ("aiter_block", True, False, True),
        ("aiter_direct", True, False, True),
        ("aiter_block", True, True, False),
        ("aiter_block", True, True, True),
        ("aiter_direct", True, True, True),
        ("cpu_direct", True, False, False),
        ("rocm", False, False, False),
        ("rocm", False, True, False),
    ],
)
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_platform_linear_conversion(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    backend,
    block,
    fnuz,
    e8m0_scales,
    preserve,
):
    """Test platform-independent transforms, not AMX/ROCm compute or AITER shuffle."""
    monkeypatch.setattr(current_platform, "is_fp8_fnuz", lambda: fnuz)
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        backend,
        False,
        block,
        False,
        preserve,
        check_forward=False,
        e8m0_scales=e8m0_scales,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA tensors")
@pytest.mark.parametrize(
    "backend,block,ragged,is_bmm",
    [
        ("xpu", True, False, False),
        ("xpu", True, True, False),
        ("xpu", True, False, True),
        ("xpu", False, False, False),
        ("xpu_w8a16", False, False, False),
    ],
)
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_xpu_linear_conversion(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    backend,
    block,
    ragged,
    is_bmm,
    preserve,
):
    """Check XPU layout/BMM caches with real cold conversion, not an XPU GEMM."""
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        backend,
        False,
        block,
        False,
        preserve,
        check_forward=False,
        ragged=ragged,
        is_bmm=is_bmm,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA tensors")
@pytest.mark.parametrize("e8m0_scales", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_b12x_block_conversion(
    default_vllm_config, dist_init, workspace_init, tmp_path, preserve, e8m0_scales
):
    """Compare real B12x conversion, not its SM120-only GEMM, on CUDA."""
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        "b12x",
        False,
        True,
        False,
        preserve,
        check_forward=False,
        e8m0_scales=e8m0_scales,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA tensors")
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("backend", ["flashinfer_trtllm", "hpc"])
def test_fp8_reload_trace_optional_moe_conversion(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    block,
    eplb,
    preserve,
    backend,
):
    """Check optional backend packing/cached scales independently of its GEMM."""
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        backend,
        True,
        block,
        eplb,
        preserve,
        check_forward=False,
    )


@pytest.mark.parametrize(
    "backend", ["triton", "vllm_cutlass", "flashinfer_trtllm", "hpc", "cpu"]
)
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_fp8_reload_trace_planned_moe_matches_legacy(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    backend,
    block,
    preserve,
    eplb,
):
    """Reuse the cold plan across reloads; CPU uses a packing lifecycle stub."""
    if backend == "cpu":
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe

        def pack(w13, w2):
            return w13.view(torch.uint8).roll(1, -1).view(w13.dtype), (
                w2.view(torch.uint8).roll(1, -1).view(w2.dtype)
            )

        monkeypatch.setattr(cpu_moe, "prepare_fp8_moe_layer_for_cpu", pack)
        # The legacy oracle imported the same helper lazily.
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        backend,
        True,
        block,
        eplb,
        preserve,
        check_forward=backend == "triton",
        legacy_reference=True,
    )


@pytest.mark.parametrize(
    "backend,block",
    [("batched_triton", False), ("batched_triton", True), ("batched_cutlass", False)],
)
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_fp8_reload_trace_batched_moe(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    backend,
    block,
    preserve,
    eplb,
):
    """Run native batched kernels before/after reload with no-comms dispatch."""
    from vllm.model_executor.layers.fused_moe.oracle import fp8
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
        BatchedPrepareAndFinalize,
    )

    monkeypatch.setattr(
        fp8,
        "maybe_make_prepare_finalize",
        lambda **_: BatchedPrepareAndFinalize(128, 2, 1, 0),
    )
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        backend,
        True,
        block,
        eplb,
        preserve,
        legacy_reference=True,
        capture_graph=False,
    )


@pytest.mark.parametrize(
    "backend,fnuz", [("aiter", False), ("aiter", True), ("xpu", False)]
)
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_fp8_reload_trace_platform_moe_matches_legacy(
    default_vllm_config,
    dist_init,
    workspace_init,
    tmp_path,
    monkeypatch,
    backend,
    fnuz,
    block,
    preserve,
    eplb,
):
    """Compare platform conversion lifecycles without AMD/Intel compute kernels."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(current_platform, "is_fp8_fnuz", lambda: fnuz)

    def shuffle(*weights):
        return tuple(w.view(torch.uint8).roll(1, -1).view(w.dtype) for w in weights)

    def init_kernel(method, layer):
        method.moe_quant_config = method.get_fused_moe_quant_config(layer)
        method.moe_kernel = SimpleNamespace(
            fused_experts=SimpleNamespace(fused_moe_impl=None)
        )

    monkeypatch.setattr(rocm_aiter_ops, "shuffle_weights", shuffle)
    monkeypatch.setattr(Fp8MoEMethod, "_init_moe_kernel", init_kernel)
    test_fp8_reload_trace_matches_cold_load(
        default_vllm_config,
        dist_init,
        workspace_init,
        tmp_path,
        backend,
        True,
        block,
        eplb,
        preserve,
        check_forward=False,
        legacy_reference=True,
    )


@pytest.mark.parametrize("is_act_and_mul", [False, True])
def test_fp8_moe_derived_scales_change_with_new_weights(is_act_and_mul):
    """Derived scales must use this round's weights and activation scales."""
    from vllm.model_executor.layers.quantization.utils.fp8_processing import (
        compute_fp8_moe_per_tensor_scales,
        compute_fp8_moe_trtllm_scales,
    )

    for factor in (0.5, 1.5):
        w1, w2 = torch.tensor([2.0, 4.0]) * factor, torch.tensor([3.0, 5.0])
        a1, a2 = torch.tensor(0.25), torch.tensor(0.5) * factor
        scales = compute_fp8_moe_per_tensor_scales(w1, w2, a1, a2)
        torch.testing.assert_close(scales["g1_alphas"], w1 * a1)
        torch.testing.assert_close(scales["g2_alphas"], w2 * a2)
        torch.testing.assert_close(scales["a1_gscale"], a1.reciprocal())
        torch.testing.assert_close(scales["a2_gscale"], a2.reciprocal())
        scales = compute_fp8_moe_trtllm_scales(w1, w2, a1, a2, is_act_and_mul)
        torch.testing.assert_close(scales["_g1_alphas"], w1 * a1)
        torch.testing.assert_close(scales["_g2_alphas"], w2 * a2)
        expected = w1 * a1 / a2 if is_act_and_mul else torch.ones_like(w1) / a2
        torch.testing.assert_close(scales["_g1_scale_c"], expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA tensors")
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_fp8_reload_trace_xpu_moe_layout(block, preserve):
    """XPU's pure tensor conversion must preserve live targets across reloads."""
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        prepare_fp8_moe_layer_for_xpu,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.utils.fp8_processing import (
        Fp8MoEProcessingPlan,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_tensor_strategy_moe,
    )
    from vllm.model_executor.model_loader.reload.fp8 import PlannedMoEReloadPolicy
    from vllm.model_executor.model_loader.reload.trace import ReloadError, ReloadState
    from vllm.model_executor.utils import replace_parameter

    suffix = "weight_scale_inv" if block else "weight_scale"
    s13, s2 = f"w13_{suffix}", f"w2_{suffix}"
    shapes = {
        "w13_weight": (2, 256, 128),
        "w2_weight": (2, 128, 128),
        s13: (2, 2, 1) if block else (2, 2),
        s2: (2, 1, 1) if block else (2,),
    }
    sources = {
        name: (torch.rand(shape, device="cuda") + 0.1).to(
            torch.float32 if "scale" in name else torch.float8_e4m3fn
        )
        for name, shape in shapes.items()
    }

    def convert(values):
        w13, ws13 = values["w13_weight"].clone(), values[s13].clone()
        if not block:
            w13, ws13 = process_fp8_weight_tensor_strategy_moe(w13, ws13, 128, 2, True)
        w13, ws13, w2, ws2 = prepare_fp8_moe_layer_for_xpu(
            w13, ws13, values["w2_weight"].clone(), values[s2].clone()
        )
        return {"w13_weight": w13, "w2_weight": w2, s13: ws13, s2: ws2}

    layer = torch.nn.Module()
    # Only layout is under test: no XPU kernel is constructed or emulated.
    experts = SimpleNamespace(fused_moe_impl=None)
    layer.quant_method = SimpleNamespace(
        moe_kernel=SimpleNamespace(fused_experts=experts),
        moe_quant_config=object(),
        fp8_backend=Fp8MoeBackend.XPU,
        processing_plan=Fp8MoEProcessingPlan(
            backend="xpu",
            block_shape=(128, 128) if block else None,
            is_act_and_mul=True,
            is_gated=True,
            shard_size=128,
            num_experts=2,
            static_input=False,
            enable_eplb=False,
            use_e8m0=False,
        ),
    )
    for name, source in sources.items():
        param = torch.nn.Parameter(torch.empty_like(source), requires_grad=False)
        param.weight_loader = default_weight_loader
        layer.register_parameter(name, param)
    state = ReloadState(
        "experts",
        layer,
        tuple(shapes),
        PlannedMoEReloadPolicy(),
    )
    trace = ModelReloadTracer()
    trace.register_state(state)

    def load(values):
        for name, source in values.items():
            param = getattr(layer, name)
            param.weight_loader(param, source)

    with trace.observe():
        load(sources)
    for name, value in convert(sources).items():
        replace_parameter(layer, name, value)
    trace.bind_runtime()
    # Simulate first forward installing the dependency's tensor references.
    cached = {
        "w13": layer.w13_weight,
        "w2": layer.w2_weight,
        "gemm1_wei_scales": getattr(layer, s13),
        "gemm2_wei_scales": getattr(layer, s2),
    }
    experts.fused_moe_impl = SimpleNamespace(**cached)
    identities = {
        name: (target.tensor, target.tensor.data_ptr())
        for name, target in state.targets.items()
    }
    for factor in (0.5, 1.5):
        incoming = {
            name: (source.float() * factor).to(source.dtype)
            for name, source in sources.items()
        }
        with trace.round(preserve_checkpoint=preserve):
            load(dict(reversed(list(incoming.items()))))
            assert state.complete
            assert bool(state.checkpoint) is preserve
        for name, expected in convert(incoming).items():
            actual = getattr(layer, name)
            assert actual is identities[name][0]
            assert actual.data_ptr() == identities[name][1]
            torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
            if preserve:
                torch.testing.assert_close(
                    state.checkpoint[name].float(),
                    incoming[name].float(),
                    rtol=0,
                    atol=0,
                )
    for name, tensor in cached.items():
        setattr(experts.fused_moe_impl, name, tensor.clone())
        with pytest.raises(ReloadError, match="no longer references"):
            state.policy.validate(state)
        setattr(experts.fused_moe_impl, name, tensor)


def test_xpu_moe_forwards_current_expert_map(monkeypatch):
    """A cached XPU executor must receive each call's current expert mapping."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import xpu_moe

    initial = {}
    calls = []

    def factory(**kwargs):
        initial.update(kwargs)
        return SimpleNamespace(apply=lambda **args: calls.append(args))

    monkeypatch.setattr(xpu_moe, "XpuFusedMoe", factory, raising=False)
    experts = SimpleNamespace(
        fused_moe_impl=None,
        quant_config=None,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        moe_config=SimpleNamespace(num_local_experts=2, ep_rank=0, ep_size=2),
        gemm1_clamp_limit=None,
    )
    tensor = torch.zeros(1, 1)
    args = dict(
        output=tensor,
        hidden_states=tensor,
        w1=tensor,
        w2=tensor,
        topk_weights=tensor,
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=4,
        a1q_scale=None,
        a2_scale=None,
        workspace13=tensor,
        workspace2=tensor,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    first = torch.tensor([0, 1, -1, -1])
    second = torch.tensor([-1, -1, 1, 0])
    xpu_moe.XPUExperts.apply(experts, **args, expert_map=first)
    impl = experts.fused_moe_impl
    xpu_moe.XPUExperts.apply(experts, **args, expert_map=second)
    assert experts.fused_moe_impl is impl
    assert initial["expert_map"] is first
    assert calls[0]["expert_map"] is first
    assert calls[1]["expert_map"] is second


def test_fp8_reload_trace_rejects_uncoordinated_eplb_collectives(monkeypatch):
    """Local readiness cannot safely order per-tensor scale collectives across EP."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization import fp8

    monkeypatch.setattr(current_platform, "is_fp8_fnuz", lambda: False)
    monkeypatch.setattr(fp8, "is_weights_pre_processed", lambda: False)
    method = SimpleNamespace(
        quant_config=Fp8Config(True, "static"),
        moe=SimpleNamespace(
            has_bias=False,
            moe_parallel_config=SimpleNamespace(enable_eplb=True, ep_size=2),
        ),
        weight_scale_refine=None,
        weight_scale_name="weight_scale",
        block_quant=False,
        fp8_backend=Fp8MoeBackend.FLASHINFER_CUTLASS,
    )
    layer = SimpleNamespace(
        expert_map_manager=SimpleNamespace(num_fused_shared_experts=0)
    )
    with pytest.raises(NotImplementedError, match="collectives are coordinated"):
        Fp8MoEMethod.create_reload_state(method, layer, "experts")


def test_kv_cache_scale_sync_to_host_copies():
    """Test device-to-host sync of the k/v quantization scales, for both the
    checkpoint-load and runtime-calc paths that produce them.
    """
    layer = torch.nn.Module()
    set_default_quant_scales(layer, register_buffer=True)
    layer.kv_cache_dtype = "fp8"

    method = BaseKVCacheMethod(quant_config=None)
    method.create_weights(layer)
    # 0.3 stays != 1.0 even after the fp8_fnuz x2 rescale.
    checkpoint_scale = torch.tensor(0.3, dtype=torch.float32)
    layer.k_scale.weight_loader(layer.k_scale, checkpoint_scale)
    layer.v_scale.weight_loader(layer.v_scale, checkpoint_scale)
    method.process_weights_after_loading(layer)

    assert layer._k_scale_float != 1.0
    assert layer._v_scale_float != 1.0
    # Host copy must mirror both the float and the device scale tensor.
    assert layer._k_scale_cpu.item() == pytest.approx(layer._k_scale_float)
    assert layer._v_scale_cpu.item() == pytest.approx(layer._v_scale_float)
    assert layer._k_scale_cpu.item() == pytest.approx(layer._k_scale.item())
    assert layer._v_scale_cpu.item() == pytest.approx(layer._v_scale.item())


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_kv_cache_dtype_skip_layers(monkeypatch, dist_init, workspace_init):
    """Test that kv_cache_dtype_skip_layers skips quantization for specified layers."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    model, _ = load_model_without_vllm_runner(
        "facebook/opt-125m",
        vllm_config_kwargs={
            "cache_config": CacheConfig(
                cache_dtype="fp8", kv_cache_dtype_skip_layers=["0", "2"]
            )
        },
    )
    for i, layer in enumerate(model.model.decoder.layers):
        expected = "auto" if str(i) in ["0", "2"] else "fp8"
        assert layer.self_attn.attn.kv_cache_dtype == expected
