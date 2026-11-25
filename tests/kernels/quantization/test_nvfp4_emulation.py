# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

import pytest
import torch
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
    kE2M1ToFloat,
)
from safetensors import safe_open

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
    fused_moe_nvfp4_emulation_kernel,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils import (
    nvfp4_emulation_utils,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _dequantize_nvfp4_kernel,
    _e4m3_inline,
    _native_e4m3,
    _nvfp4_gathered_bias_kernel,
    _nvfp4_quant_dequant_kernel,
    _round_to_e4m3,
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950
else:

    def on_gfx950() -> bool:
        return False


MOE_MODEL_CONFIGS = {
    "nvidia/Qwen3-30B-A3B-NVFP4": {
        "shards": ["model-00001-of-00004.safetensors"],
        "expert_prefix": "model.layers.9.mlp.experts.",
        # Position of the expert index in the dot-split key.
        "expert_idx_pos": 5,
    }
}


@pytest.fixture(scope="module")
def loaded_model_files():
    return {
        model_id: hf_api().snapshot_download(
            repo_id=model_id, allow_patterns=config["shards"]
        )
        for model_id, config in MOE_MODEL_CONFIGS.items()
    }


class Nvfp4QuantizationEmulationTritonExpertsReference(TritonExperts):
    """Extension of TritonExperts to support emulated NVFP4 MoE experts.

    It may be used for NVFP4 models when the device does not have
    native support for this dtype.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        # `TritonExperts.apply` expects pre-dequantized weights,
        # which we handle in `apply` below.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

        self.quantization_emulation = True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return "nvfp4"

    @property
    def a1_scale(self) -> torch.Tensor | None:
        return self.quant_config.a1_gscale

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        # Dequantize w1 from packed NVFP4 to fp16/bf16
        w13_global_scale = self.quant_config.g1_alphas

        w1_dequant = dequantize_to_dtype(
            tensor_fp4=w1,
            tensor_sf=self.w1_scale_val,
            global_scale=w13_global_scale,
            dtype=hidden_states.dtype,
            block_size=16,
            swizzle=False,
        )

        # Dequantize w2 from packed NVFP4 to fp16/bf16
        w2_global_scale = self.quant_config.g2_alphas

        w2_dequant = dequantize_to_dtype(
            tensor_fp4=w2,
            tensor_sf=self.w2_scale_val,
            global_scale=w2_global_scale,
            dtype=hidden_states.dtype,
            block_size=16,
            swizzle=False,
        )

        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_dequant,
            w2=w2_dequant,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=self.quant_config.a2_gscale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


@pytest.mark.parametrize(
    ("config_kwargs", "expected_reason"),
    [
        ({"has_bias": True}, "kernel does not support bias"),
        ({"is_lora_enabled": True}, "kernel does not support LoRA"),
    ],
)
def test_nvfp4_emulation_support_check_rejects_bias_and_lora(
    config_kwargs: dict[str, bool],
    expected_reason: str,
) -> None:
    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=16,
        intermediate_size=16,
        num_local_experts=2,
        num_logical_experts=2,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=current_platform.device_type,
        routing_method=RoutingMethodType.TopK,
        **config_kwargs,
    )

    supported, reason = Nvfp4QuantizationEmulationTritonExperts.is_supported_config(
        Nvfp4QuantizationEmulationTritonExperts,
        moe_config,
        kNvfp4Static,
        kNvfp4Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == expected_reason


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
def test_triton_dequantize_nvfp4(monkeypatch, loaded_model_files) -> None:
    """Test the Triton dequantization kernel against the CPU reference
    using real NVFP4 weights from a checkpoint.

    Tests both 2D (attention projection) and 3D (stacked MoE experts).
    """
    checkpoint_path = loaded_model_files["nvidia/Qwen3-30B-A3B-NVFP4"]
    shards = cast(list[str], MOE_MODEL_CONFIGS["nvidia/Qwen3-30B-A3B-NVFP4"]["shards"])
    shard_path = f"{checkpoint_path}/{shards[0]}"
    block_size = 16

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        # 2D case: attention projection
        tensor_fp4_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight")
        tensor_sf_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight_scale")
        global_scale_2d = f.get_tensor("model.layers.9.self_attn.k_proj.weight_scale_2")

        # 3D case: stack ALL experts for layer 9 up_proj
        expert_prefix = "model.layers.9.mlp.experts."
        expert_indices = sorted(
            int(key.split(".")[5])
            for key in all_keys
            if key.startswith(expert_prefix) and key.endswith(".up_proj.weight")
        )
        assert len(expert_indices) > 0

        all_fp4 = []
        all_sf = []
        all_global_scale = []
        for index in expert_indices:
            name = f"{expert_prefix}{index}.up_proj"
            all_fp4.append(f.get_tensor(f"{name}.weight"))
            all_sf.append(f.get_tensor(f"{name}.weight_scale"))
            all_global_scale.append(f.get_tensor(f"{name}.weight_scale_2"))

    tensor_fp4_3d = torch.stack(all_fp4)
    tensor_sf_3d = torch.stack(all_sf)
    global_scale_3d = torch.stack(all_global_scale)

    test_cases = [
        ("2D base", tensor_fp4_2d, tensor_sf_2d, global_scale_2d),
        (
            "2D 2x rows",
            tensor_fp4_2d.repeat(2, 1),
            tensor_sf_2d.repeat(2, 1),
            global_scale_2d,
        ),
        (
            "2D 4x rows",
            tensor_fp4_2d.repeat(4, 1),
            tensor_sf_2d.repeat(4, 1),
            global_scale_2d,
        ),
        (
            "2D 2x cols",
            tensor_fp4_2d.repeat(1, 2),
            tensor_sf_2d.repeat(1, 2),
            global_scale_2d,
        ),
        ("3D base", tensor_fp4_3d, tensor_sf_3d, global_scale_3d),
        (
            "3D 2x experts",
            tensor_fp4_3d.repeat(2, 1, 1),
            tensor_sf_3d.repeat(2, 1, 1),
            global_scale_3d.repeat(2),
        ),
        (
            "3D 2x rows",
            tensor_fp4_3d.repeat(1, 2, 1),
            tensor_sf_3d.repeat(1, 2, 1),
            global_scale_3d,
        ),
        (
            "3D 2x cols",
            tensor_fp4_3d.repeat(1, 1, 2),
            tensor_sf_3d.repeat(1, 1, 2),
            global_scale_3d,
        ),
    ]

    quantiles = [0.5, 0.001, 0.999]

    # Move the E2M1 lookup table to CUDA ahead of time, as would normally
    # happen during model loading (process_weights_after_loading).  Both the
    # Triton and PyTorch reference paths run on CUDA.
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.cuda()
    )

    for label, tensor_fp4, tensor_sf, global_scale in test_cases:
        fp4_cuda = tensor_fp4.cuda()
        sf_cuda = tensor_sf.cuda()
        gs_cuda = global_scale.cuda()

        # Triton path
        triton_result = dequantize_to_dtype(
            fp4_cuda,
            sf_cuda,
            gs_cuda,
            torch.bfloat16,
            block_size,
            swizzle=False,
        )

        # Reference path (PyTorch ops on CUDA, Triton dispatch disabled)
        with monkeypatch.context() as m:
            m.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            reference = dequantize_to_dtype(
                fp4_cuda,
                sf_cuda,
                gs_cuda,
                torch.bfloat16,
                block_size,
                swizzle=False,
            )

        torch.testing.assert_close(triton_result, reference, atol=0, rtol=0)

        # Benchmark
        shape = list(tensor_fp4.shape)

        def _triton_bench(
            fp4_cuda=fp4_cuda,
            scale_cuda=sf_cuda,
            global_scale_cuda=gs_cuda,
            block_size=block_size,
        ):
            return dequantize_to_dtype(
                fp4_cuda,
                scale_cuda,
                global_scale_cuda,
                torch.bfloat16,
                block_size,
                swizzle=False,
            )

        triton_ms, triton_min, triton_max = triton.testing.do_bench(
            _triton_bench, quantiles=quantiles
        )

        def _reference_bench(
            fp4_cuda=fp4_cuda,
            scale_cuda=sf_cuda,
            global_scale_cuda=gs_cuda,
            block_size=block_size,
        ):
            with monkeypatch.context() as m2:
                m2.setattr(
                    nvfp4_emulation_utils.current_platform,
                    "is_cuda_alike",
                    lambda: False,
                )
                dequantize_to_dtype(
                    fp4_cuda,
                    scale_cuda,
                    global_scale_cuda,
                    torch.bfloat16,
                    block_size,
                    swizzle=False,
                )

        ref_ms, ref_min, ref_max = triton.testing.do_bench(
            _reference_bench, quantiles=quantiles
        )

        speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
        print(f"  dequantize {label} {shape}:")
        print(
            f"    triton:    median={triton_ms:.3f}ms, "
            f"min={triton_min:.3f}ms, max={triton_max:.3f}ms"
        )
        print(
            f"    reference: median={ref_ms:.3f}ms, "
            f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
        )
        print(f"    speedup:   {speedup:.2f}x")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
@pytest.mark.parametrize(
    "m, k",
    [
        (1, 16),
        (1, 4096),
        (2, 4096),
        (4, 4096),
        (8, 4096),
        (16, 4096),
        (24, 4096),
        (32, 4096),
        (1, 8192),
        (2, 8192),
        (4, 8192),
        (8, 8192),
        (16, 8192),
        (24, 8192),
        (32, 8192),
        (1, 32),
        (2, 48),
        (7, 64),
        (16, 128),
        (33, 160),
        (128, 256),
        (256, 512),
        (1024, 1024),
        (5120, 2048),
        (2048, 4096),
        (4096, 7168),
        (8192, 8192),
        (128, 16384),
    ],
)
@pytest.mark.parametrize("global_scale_value", [0.5, 1.0, 0.001])
def test_triton_nvfp4_quant_dequant(
    monkeypatch, m: int, k: int, global_scale_value: float
) -> None:
    """Test the Triton quant-dequant kernel against the CPU reference."""
    block_size = 16
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor(global_scale_value, dtype=torch.float32, device="cuda")

    # Triton path
    triton_result = ref_nvfp4_quant_dequant(x, global_scale, block_size)

    # CPU reference path
    with monkeypatch.context() as mp:
        mp.setattr(
            nvfp4_emulation_utils.current_platform,
            "is_cuda_alike",
            lambda: False,
        )
        reference = ref_nvfp4_quant_dequant(x.cpu(), global_scale.cpu(), block_size)

    torch.testing.assert_close(triton_result.cpu(), reference, atol=0, rtol=0)

    # Benchmark (both paths on CUDA tensors for fair comparison)
    quantiles = [0.5, 0.001, 0.999]

    def _triton_bench(
        input_tensor=x, input_global_scale=global_scale, input_block_size=block_size
    ):
        return ref_nvfp4_quant_dequant(
            input_tensor, input_global_scale, input_block_size
        )

    triton_ms, triton_min, triton_max = triton.testing.do_bench(
        _triton_bench, quantiles=quantiles
    )

    def _reference_bench(
        input_tensor=x, input_global_scale=global_scale, input_block_size=block_size
    ):
        with monkeypatch.context() as mp2:
            mp2.setattr(
                nvfp4_emulation_utils.current_platform,
                "is_cuda_alike",
                lambda: False,
            )
            ref_nvfp4_quant_dequant(input_tensor, input_global_scale, input_block_size)

    ref_ms, ref_min, ref_max = triton.testing.do_bench(
        _reference_bench, quantiles=quantiles
    )

    speedup = ref_ms / triton_ms if triton_ms > 0 else float("inf")
    print(f"  quant_dequant [{m}x{k}] gs={global_scale_value}:")
    print(
        f"    triton:    median={triton_ms:.3f}ms, "
        f"min={triton_min:.3f}ms, max={triton_max:.3f}ms"
    )
    print(
        f"    reference: median={ref_ms:.3f}ms, "
        f"min={ref_min:.3f}ms, max={ref_max:.3f}ms"
    )
    print(f"    speedup:   {speedup:.2f}x")


def _load_nvfp4_moe_weights(
    model_files: dict[str, str],
    model_id: str,
    tensor_parallel_size: int,
    max_experts: int | None = None,
):
    """Load and stack NVFP4 MoE weights from checkpoint shards.

    Returns (w1, w1_scale, w1_gscale, w2, w2_scale, w2_gscale,
             a1_gscale, a2_gscale, num_experts, hidden_dim,
             intermediate_size).

    When max_experts is set, only the first max_experts experts are loaded.

    When tensor_parallel_size > 1, the N dimension of w1 and the K
    dimension of w2 are narrowed to the first TP shard (simulating
    column-parallel on w1 / row-parallel on w2).
    """
    cfg = MOE_MODEL_CONFIGS[model_id]
    shards = cast(list[str], cfg["shards"])
    checkpoint_path = model_files[model_id]

    expert_prefix = cfg["expert_prefix"]
    idx_pos = cast(int, cfg["expert_idx_pos"])

    # Collect all tensors across shards into a flat dict — an expert's
    # tensors may be split across multiple shard files.
    all_tensors: dict[str, torch.Tensor] = {}
    for shard_name in shards:
        shard_path = f"{checkpoint_path}/{shard_name}"
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(expert_prefix):
                    all_tensors[key] = f.get_tensor(key)

    expert_indices = sorted(
        {
            int(key.split(".")[idx_pos])
            for key in all_tensors
            if key.endswith(".gate_proj.weight")
        }
    )
    if max_experts is not None:
        expert_indices = expert_indices[:max_experts]
    num_experts = len(expert_indices)

    gate_weights, up_weights, down_weights = [], [], []
    gate_scales, up_scales, down_scales = [], [], []
    gate_gscales, up_gscales, down_gscales = [], [], []
    a1_scales, a2_scales = [], []

    for idx in expert_indices:
        prefix = f"{expert_prefix}{idx}"
        gate_weights.append(all_tensors[f"{prefix}.gate_proj.weight"])
        gate_scales.append(all_tensors[f"{prefix}.gate_proj.weight_scale"])
        gate_gscales.append(all_tensors[f"{prefix}.gate_proj.weight_scale_2"])
        up_weights.append(all_tensors[f"{prefix}.up_proj.weight"])
        up_scales.append(all_tensors[f"{prefix}.up_proj.weight_scale"])
        up_gscales.append(all_tensors[f"{prefix}.up_proj.weight_scale_2"])
        down_weights.append(all_tensors[f"{prefix}.down_proj.weight"])
        down_scales.append(all_tensors[f"{prefix}.down_proj.weight_scale"])
        down_gscales.append(all_tensors[f"{prefix}.down_proj.weight_scale_2"])
        a1_scales.append(all_tensors[f"{prefix}.gate_proj.input_scale"])
        a2_scales.append(all_tensors[f"{prefix}.down_proj.input_scale"])

    # Stack into MoE format.
    # w1 = [E, 2*intermediate, hidden//2]  (gate + up concatenated)
    w1 = torch.stack(
        [torch.cat([g, u], dim=0) for g, u in zip(gate_weights, up_weights)]
    ).cuda()
    w1_scale = torch.stack(
        [torch.cat([g, u], dim=0) for g, u in zip(gate_scales, up_scales)]
    ).cuda()
    w1_gscale = torch.stack(gate_gscales).cuda()

    # w2 = [E, hidden, intermediate//2]
    w2 = torch.stack(down_weights).cuda()
    w2_scale = torch.stack(down_scales).cuda()
    w2_gscale = torch.stack(down_gscales).cuda()

    a13_scale_raw = torch.stack(a1_scales).cuda()
    a2_scale_raw = torch.stack(a2_scales).cuda()

    # Apply EMULATION transforms (matches oracle/nvfp4.py).
    nvfp4_emulation_utils.kE2M1ToFloat_handle.val = (
        nvfp4_emulation_utils.kE2M1ToFloat_handle.val.cuda()
    )
    a1_gscale = 1.0 / a13_scale_raw.max().to(torch.float32)
    a2_gscale = 1.0 / a2_scale_raw.max().to(torch.float32)

    # ── Simulate TP sharding ──
    # w1 (gate_up): column-parallel → shard the N dimension (dim 1).
    # w2 (down):    row-parallel   → shard the K dimension (dim 2,
    #               which is the packed K//2 dim).
    # Scales follow the same sharding on the corresponding dimension.
    tp = tensor_parallel_size
    if tp > 1:
        n1 = w1.size(1) // tp
        w1 = w1[:, :n1, :].contiguous()
        w1_scale = w1_scale[:, :n1, :].contiguous()

        k2_packed = w2.size(2) // tp
        k2_scale = w2_scale.size(2) // tp
        w2 = w2[:, :, :k2_packed].contiguous()
        w2_scale = w2_scale[:, :, :k2_scale].contiguous()

    hidden_dim = w1.size(2) * 2
    intermediate_size = w1.size(1) // 2

    return (
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        num_experts,
        hidden_dim,
        intermediate_size,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 1024])
@pytest.mark.parametrize("top_k", [4])
@pytest.mark.parametrize("model_id", list(MOE_MODEL_CONFIGS.keys()))
@pytest.mark.parametrize(
    "tensor_parallel_size",
    [pytest.param(val, id=f"tensor_parallel_size:{val}") for val in [1, 2, 4, 8]],
)
def test_nvfp4_moe_correctness(
    loaded_model_files,
    num_tokens: int,
    top_k: int,
    model_id: str,
    tensor_parallel_size: int,
) -> None:
    """Compare Nvfp4QuantizationEmulationTritonExperts (fused weight dequant + compute)
    against the unfused reference Nvfp4QuantizationEmulationTritonExpertsReference.

    Both must produce bit-identical results.
    """
    num_test_experts = max(8, top_k)
    (
        w1,
        w1_scale,
        w1_gscale,
        w2,
        w2_scale,
        w2_gscale,
        a1_gscale,
        a2_gscale,
        num_experts,
        hidden_dim,
        intermediate_size,
    ) = _load_nvfp4_moe_weights(
        loaded_model_files,
        model_id,
        tensor_parallel_size,
        max_experts=num_test_experts,
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )

    def _make_quant_config():
        return nvfp4_moe_quant_config(
            g1_alphas=w1_gscale.clone(),
            g2_alphas=w2_gscale.clone(),
            a1_gscale=a1_gscale.clone(),
            a2_gscale=a2_gscale.clone(),
            w1_scale=w1_scale.clone(),
            w2_scale=w2_scale.clone(),
        )

    ref_experts = Nvfp4QuantizationEmulationTritonExpertsReference(
        moe_config=moe_config,
        quant_config=_make_quant_config(),
    )
    fused_experts = Nvfp4QuantizationEmulationTritonExperts(
        moe_config=moe_config,
        quant_config=_make_quant_config(),
    )

    torch.manual_seed(42)
    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )

    topk_weights = torch.randn(
        num_tokens, top_k, dtype=torch.float32, device="cuda"
    ).softmax(dim=-1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    N = w1.size(1)  # 2 * intermediate
    K = hidden_dim

    ws13_size = num_tokens * top_k * max(intermediate_size, K)
    ws2_size = num_tokens * top_k * max(N, K)

    workspace13_ref = torch.zeros(ws13_size, dtype=torch.bfloat16, device="cuda")
    workspace2_ref = torch.zeros(ws2_size, dtype=torch.bfloat16, device="cuda")
    output_ref = torch.zeros(num_tokens, K, dtype=torch.bfloat16, device="cuda")

    workspace13_fused = torch.zeros_like(workspace13_ref)
    workspace2_fused = torch.zeros_like(workspace2_ref)
    output_fused = torch.zeros_like(output_ref)

    apply_kwargs = dict(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    # Unfused reference.
    ref_experts.apply(
        output=output_ref,
        workspace13=workspace13_ref,
        workspace2=workspace2_ref,
        **apply_kwargs,
    )

    # Fused implementation.
    fused_experts.apply(
        output=output_fused,
        workspace13=workspace13_fused,
        workspace2=workspace2_fused,
        **apply_kwargs,
    )

    # Not strict equality on H100, MI325, MI300 (< 0.1% elements).
    # The fused on-the-fly dequant path can lower to a slightly
    # different Triton/MMA tiling than the pre-dequantized
    # reference; experiments with reference-like tiling/masking
    # reduced some diffs were not kept because they regress
    # the fused kernel speed.
    # Strict equality validated on MI355.
    torch.testing.assert_close(
        output_fused,
        output_ref,
        atol=0.0 if on_gfx950() else 0.02,
        rtol=0,
    )


def _dense_nvfp4_reference(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Decode E2M1 nibbles and both scale levels without production helpers."""
    levels = kE2M1ToFloat.cpu()
    weight = break_fp4_bytes(layer.weight.cpu(), torch.float32)
    weight *= (
        layer.weight_scale.cpu().float() * layer.weight_global_scale.cpu()
    ).repeat_interleave(16, dim=-1)
    blocks = x.cpu().float().reshape(x.shape[0], -1, 16)
    global_scale = layer.input_global_scale_inv.cpu()
    scales = (
        (blocks.abs().amax(-1, keepdim=True) * (1.0 / FLOAT4_E2M1_MAX) * global_scale)
        .clamp(max=FLOAT8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    inverse = torch.where(scales == 0, 0, global_scale / scales)
    scaled = (blocks * inverse).abs().clamp(max=FLOAT4_E2M1_MAX)
    distances = (scaled.unsqueeze(-1) - levels).abs()
    # E2M1 ties round to the even significand (even encoding).
    nearest_mask = distances == distances.amin(-1, keepdim=True)
    even_mask = nearest_mask & (torch.arange(8, device="cpu") % 2 == 0)
    candidates = torch.where(even_mask.any(-1, keepdim=True), even_mask, nearest_mask)
    nearest = candidates.to(torch.int8).argmax(-1)
    activation = levels[nearest] * blocks.sign() * (scales / global_scale)
    expected = activation.reshape_as(x).to(x.dtype) @ weight.to(x.dtype).T
    if bias is not None:
        expected += bias.cpu()
    return expected


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm NVFP4 fallback")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k", [(1, 32, 64), (17, 128, 256)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_nvfp4_dense_emulation(
    dtype: torch.dtype, m: int, n: int, k: int, with_bias: bool
) -> None:
    """W4A4 fallback preserves packed scales, bias and outputs across replay."""
    set_random_seed(47)
    layer = torch.nn.Module()
    # Every signed E2M1 code appears, including both zero encodings.
    codes = (torch.randperm(n * k, device="cuda") % 16).reshape(n, k)
    layer.weight = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    layer.weight_scale = (
        torch.arange(n * k // 16, device="cuda").reshape(n, k // 16) % 7 / 8
    ).to(torch.float8_e4m3fn)
    layer.weight_global_scale = torch.tensor(0.25, device="cuda")
    layer.input_global_scale_inv = torch.tensor(3.0, device="cuda")
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    x[:, :16] = 0
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="emulation"))
    ):
        kernel = init_nvfp4_linear_kernel()
    assert isinstance(kernel, EmulationNvFp4LinearKernel)
    kernel.process_weights_after_loading(layer)
    original_weight = layer.weight.clone()
    original_scale = layer.weight_scale.view(torch.uint8).clone()
    actual = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(
        actual.cpu(), _dense_nvfp4_reference(layer, x, bias), atol=0.03, rtol=0.01
    )
    assert actual.abs().max() > 0.1
    saved = actual.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.apply_weights(layer, x, bias)
    for sign in (-1, 1):
        x.mul_(sign * 0.75)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            captured.cpu(), _dense_nvfp4_reference(layer, x, bias), atol=0.03, rtol=0.01
        )
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)
    assert torch.equal(layer.weight, original_weight)
    assert torch.equal(layer.weight_scale.view(torch.uint8), original_scale)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("invert_global_scales", [True, False])
@pytest.mark.parametrize("experts_per_chunk", [1, 2, 3, 8])
def test_dequantize_nvfp4_moe_weights_chunked_matches_full(
    experts_per_chunk: int, invert_global_scales: bool, monkeypatch
) -> None:
    """Chunked dequantization must equal the single-chunk dequantization exactly.

    5 experts cover one expert per chunk, a smaller last chunk (2, 3) and a
    chunk larger than the number of experts (8).
    """
    set_random_seed(0)
    device = torch.device(current_platform.device_type)
    num_experts, hidden, inter, group = 5, 64, 32, 16

    def packed(rows: int, cols: int) -> torch.Tensor:
        return torch.randint(
            0, 256, (num_experts, rows, cols // 2), dtype=torch.uint8, device=device
        )

    def block_scale(rows: int, cols: int) -> torch.Tensor:
        scale = torch.rand(num_experts, rows, cols // group, device=device) * 4 + 0.5
        return scale.to(torch.float8_e4m3fn)

    w13_packed, w13_scale = packed(2 * inter, hidden), block_scale(2 * inter, hidden)
    w2_packed, w2_scale = packed(hidden, inter), block_scale(hidden, inter)
    w13_global = torch.rand(num_experts, 2, device=device) + 0.5
    w2_global = torch.rand(num_experts, device=device) + 0.5

    args = (w13_packed, w13_scale, w13_global, w2_packed, w2_scale, w2_global)
    dtype = torch.float32
    # The default chunk budget holds all experts at once.
    ((_, expected_w13, expected_w2),) = (
        nvfp4_emulation_utils.dequantize_nvfp4_moe_weights(
            *args, dtype, invert_global_scales=invert_global_scales
        )
    )

    expert_bytes = dtype.itemsize * 2 * (w13_packed[0].numel() + w2_packed[0].numel())
    monkeypatch.setattr(
        nvfp4_emulation_utils,
        "_DEQUANT_CHUNK_BYTES",
        experts_per_chunk * expert_bytes,
    )
    chunks = list(
        nvfp4_emulation_utils.dequantize_nvfp4_moe_weights(
            *args, dtype, invert_global_scales=invert_global_scales
        )
    )

    expected_starts = list(range(0, num_experts, experts_per_chunk))
    assert [c[0].start for c in chunks] == expected_starts
    assert chunks[-1][0].stop == num_experts
    assert torch.equal(torch.cat([c[1] for c in chunks]), expected_w13)
    assert torch.equal(torch.cat([c[2] for c in chunks]), expected_w2)


@triton.jit
def _decode_e4m3_kernel(src_ptr, dst_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    codes = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, _e4m3_inline(codes))


@triton.jit
def _round_e4m3_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, _round_to_e4m3(x), mask=mask)


def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
    rounded = torch.empty_like(x)
    _round_e4m3_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, rounded, x.numel(), BLOCK=1024
    )
    return rounded


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int32) == b.view(torch.int32)) | (a.isnan() & b.isnan())


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton NVFP4 kernel requires CUDA.",
)
def test_portable_e4m3_helpers_match_torch() -> None:
    """The FP8-free E4M3 helpers used below SM89 must match torch exactly.

    They run on any GPU here, so CI without SM8x hardware still covers the
    path that A100 and other pre-SM89 GPUs take. Rounding runs under
    `torch.compile`, which re-generates the kernel source as the emulation
    linear path does.
    """
    device = "cuda"
    fp8 = torch.float8_e4m3fn

    codes = torch.arange(256, dtype=torch.uint8, device=device)
    decoded = torch.empty(256, device=device)
    _decode_e4m3_kernel[(1,)](codes, decoded, N=256)
    assert _bitwise_equal(decoded, codes.view(fp8).float()).all()

    # Every representable value, every tie, values 1 and 2 ulps either side of
    # each tie, and random values across the whole range including subnormals.
    values = codes.view(fp8).float()
    values = values[values.isfinite() & (values >= 0)].unique().sort().values
    ties = (values[:-1] + values[1:]) / 2
    nudged = [(ties.view(torch.int32) + k).view(torch.float32) for k in (-2, -1, 1, 2)]
    generator = torch.Generator(device=device).manual_seed(0)
    random = torch.exp2(
        torch.empty(1 << 20, device=device).uniform_(-14, 8.8, generator=generator)
    )
    x = torch.cat([values, ties, *nudged, random])
    x = torch.cat([x, -x]).clamp(-448.0, 448.0).contiguous()
    rounded = torch.compile(_round_e4m3, fullgraph=True)(x)
    expected = x.to(fp8).float()
    mismatch = ~_bitwise_equal(rounded, expected)
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} mismatches, e.g. {x[mismatch][:4].tolist()} -> "
        f"{rounded[mismatch][:4].tolist()} (expected {expected[mismatch][:4].tolist()})"
    )


# Pointer arguments of each E4M3-scaled kernel; all other runtime arguments are
# int32 strides or sizes, except where listed.
_E4M3_KERNELS: dict[str, tuple[Any, dict[str, str], dict[str, Any]]] = {
    "gathered_bias": (
        _nvfp4_gathered_bias_kernel,
        {
            "markov_ptr": "*bf16",
            "packed_weight_ptr": "*u8",
            "weight_scale_ptr": "*u8",
            "global_scale_ptr": "*fp32",
            "values_ptr": "*fp32",
            "index_ptr": "*i64",
            "logits_ptr": "*fp32",
            "alpha": "fp32",
        },
        {"TOPK": 2, "K": 64, "NUM_BLOCKS": 4},
    ),
    "dequantize": (
        _dequantize_nvfp4_kernel,
        {
            "fp4_ptr": "*u8",
            "scale_ptr": "*u8",
            "global_scale_ptr": "*fp32",
            "output_ptr": "*bf16",
        },
        {
            "rows_per_batch": 1,
            "num_blocks": 4,
            "BLOCK_SIZE": 16,
            "has_batch_global_scale": False,
            "TILE_BLOCKS": 4,
        },
    ),
    "quant_dequant": (
        _nvfp4_quant_dequant_kernel,
        {"input_ptr": "*bf16", "output_ptr": "*bf16", "global_scale_ptr": "*fp32"},
        {
            "k": 64,
            "num_blocks": 4,
            "BLOCK_SIZE": 16,
            "FP4_MAX_RECIPROCAL": 1 / 6,
            "TILE_BLOCKS": 4,
        },
    ),
    "moe": (
        fused_moe_nvfp4_emulation_kernel,
        {
            "a_ptr": "*bf16",
            "b_ptr": "*u8",
            "c_ptr": "*bf16",
            "b_scale_ptr": "*u8",
            "w_global_scale_ptr": "*fp32",
            "topk_weights_ptr": "*fp32",
            "sorted_token_ids_ptr": "*i32",
            "expert_ids_ptr": "*i32",
            "num_tokens_post_padded_ptr": "*i32",
        },
        {
            "N": 64,
            "K": 64,
            "block_k_diviable": True,
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "MUL_ROUTED_WEIGHT": False,
            "top_k": 1,
            "group_size": 16,
        },
    ),
}
# Kernels whose constexpr `compute_type` is a Triton dtype, resolved at use.
_E4M3_KERNEL_COMPUTE_TYPE = {"moe": "bfloat16"}


def _compile_for(kernel_name: str, arch: int, native_e4m3: bool) -> None:
    """Compile a kernel for an explicit CUDA target, without a GPU or launch."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn, runtime_types, constants = _E4M3_KERNELS[kernel_name]
    constants = {**constants, "NATIVE_E4M3": native_e4m3}
    if kernel_name in _E4M3_KERNEL_COMPUTE_TYPE:
        constants["compute_type"] = getattr(tl, _E4M3_KERNEL_COMPUTE_TYPE[kernel_name])
    signature = {
        name: "constexpr" if name in constants else runtime_types.get(name, "i32")
        for name in fn.arg_names
    }
    source = ASTSource(fn, signature=signature, constexprs=constants)
    triton.compile(source, target=GPUTarget("cuda", arch, 32))


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Compiles for explicit NVIDIA targets with Triton's CUDA backend.",
)
@pytest.mark.parametrize("kernel_name", list(_E4M3_KERNELS))
@pytest.mark.parametrize(
    "archs", [(80, 90), (90, 80)], ids=["sm80_first", "sm90_first"]
)
def test_e4m3_kernels_compile_per_target(kernel_name: str, archs) -> None:
    """Each launch picks its E4M3 path for its own GPU, in either order.

    The portable variant must compile for SM80 whether an SM89+ variant of the
    same kernel was compiled before or after it in the process.
    """
    for arch in archs:
        _compile_for(kernel_name, arch, native_e4m3=arch >= 89)


def test_native_e4m3_follows_each_device(monkeypatch) -> None:
    """The E4M3 path is chosen per launch device, not once per process."""
    capabilities = {0: (9, 0), 1: (8, 0), 2: (8, 9), 3: (12, 0)}
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: capabilities[torch.device(device).index],
    )
    for order in ([0, 1, 2, 3], [1, 0, 3, 2]):
        assert [_native_e4m3(torch.device("cuda", i)) for i in order] == [
            capabilities[i] >= (8, 9) for i in order
        ]
