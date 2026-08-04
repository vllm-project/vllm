# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import huggingface_hub
import pytest
import torch
from safetensors import safe_open

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
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
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils import (
    nvfp4_emulation_utils,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    ref_nvfp4_quant_dequant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

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
        model_id: huggingface_hub.snapshot_download(
            repo_id=model_id, allow_patterns=config["shards"]
        )
        for model_id, config in MOE_MODEL_CONFIGS.items()
    }


class Nvfp4QuantizationEmulationTritonExpertsReference(TritonExperts):
    """
    Extension of TritonExperts to support emulated NVFP4 MoE experts.

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
        device="cuda",
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
