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
    from vllm.models.deepseek_v4_1 import quant_config as quant_module
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

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
    from vllm.models.deepseek_v4_1.nvidia import model as model_module
    from vllm.models.deepseek_v4_1.nvidia import vl_model as vl_module

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
    from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4Model
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
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
