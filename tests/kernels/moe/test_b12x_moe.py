# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the native B12X tensor-parallel MoE integration."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.b12x_moe as b12x_moe
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle
import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    mxfp4_w4a16_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.b12x_moe import (
    prepare_nvfp4_moe_layer_for_b12x,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights_q = []
    scales = []
    global_scales = []
    for expert_weight in weight:
        global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / expert_weight.abs().max()
        ).to(torch.float32)
        weight_q, scale = ops.scaled_fp4_quant(
            expert_weight,
            global_scale,
            is_sf_swizzled_layout=False,
        )
        weights_q.append(weight_q)
        scales.append(scale)
        global_scales.append(global_scale)
    return torch.stack(weights_q), torch.stack(scales), torch.stack(global_scales)


def _dequantize_nvfp4_linear(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows, packed_cols = tensor_fp4.shape
    cols = packed_cols * 2
    values = break_fp4_bytes(tensor_fp4, torch.float32)
    values = values.reshape(rows, cols // 16, 16)
    scales = tensor_sf.view(torch.float8_e4m3fn).to(torch.float32)
    return (
        (values * (scales[:, : cols // 16] / global_scale).unsqueeze(-1))
        .reshape(rows, cols)
        .to(dtype)
    )


def _nvfp4_activation_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    routed_input = (
        hidden_states[:, None, :]
        .expand(-1, topk, -1)
        .reshape(tokens * topk, hidden_size)
    )
    routed_output = torch.zeros(
        tokens * topk,
        hidden_size,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    flat_ids = topk_ids.reshape(-1)

    for expert in range(w1.shape[0]):
        mask = flat_ids == expert
        if not mask.any():
            continue
        a1_q, a1_block_scale = ops.scaled_fp4_quant(
            routed_input[mask],
            a1_scale[expert],
            is_sf_swizzled_layout=False,
        )
        a1 = _dequantize_nvfp4_linear(
            a1_q,
            a1_block_scale,
            a1_scale[expert],
            torch.float32,
        )
        fc1 = a1 @ w1[expert].float().t()
        gate, up = fc1.chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        a2_q, a2_block_scale = ops.scaled_fp4_quant(
            intermediate,
            a2_scale[expert],
            is_sf_swizzled_layout=False,
        )
        a2 = _dequantize_nvfp4_linear(
            a2_q,
            a2_block_scale,
            a2_scale[expert],
            torch.float32,
        )
        routed_output[mask] = a2 @ w2[expert].float().t()

    return (
        routed_output.view(tokens, topk, hidden_size)
        .mul(topk_weights[..., None])
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


def _e8m0_bytes_to_float(scale_bytes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale_bytes.view(torch.uint8).to(torch.float32) - 127.0)


def _e8m0_scale_bytes_from_amax(amax: torch.Tensor) -> torch.Tensor:
    scale = amax.to(torch.float32) / 6.0
    scale = torch.where(scale > 0, scale, torch.full_like(scale, 2.0**-127))
    exponent = torch.ceil(torch.log2(scale)).clamp(min=-127.0, max=120.0)
    return (exponent + 127.0).to(torch.uint8)


def _mxfp4_decode_packed(packed: torch.Tensor, cols: int) -> torch.Tensor:
    fp4_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    packed = packed.view(torch.uint8)
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    return fp4_lut[codes[:, :cols]]


def _mxfp4_encode_values(values: torch.Tensor) -> torch.Tensor:
    mag = values.abs()
    codes = torch.zeros_like(mag, dtype=torch.uint8)
    codes = torch.where((mag > 0.25) & (mag < 0.75), 1, codes)
    codes = torch.where((mag >= 0.75) & (mag <= 1.25), 2, codes)
    codes = torch.where((mag > 1.25) & (mag < 1.75), 3, codes)
    codes = torch.where((mag >= 1.75) & (mag <= 2.5), 4, codes)
    codes = torch.where((mag > 2.5) & (mag < 3.5), 5, codes)
    codes = torch.where((mag >= 3.5) & (mag <= 5.0), 6, codes)
    codes = torch.where(mag > 5.0, 7, codes)
    return codes | ((values < 0).to(torch.uint8) << 3)


def _quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape[-2:]
    blocks = weight.float().reshape(-1, cols // 32, 32)
    scale_bytes = _e8m0_scale_bytes_from_amax(blocks.abs().amax(dim=-1))
    scales = _e8m0_bytes_to_float(scale_bytes).unsqueeze(-1)
    codes = _mxfp4_encode_values(blocks / scales).reshape(-1, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return (
        packed.reshape(*weight.shape[:-2], rows, cols // 2),
        scale_bytes.reshape(*weight.shape[:-2], rows, cols // 32),
    )


def _dequantize_mxfp4(
    packed: torch.Tensor,
    scale_bytes: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    values = _mxfp4_decode_packed(packed, cols)
    scales = _e8m0_bytes_to_float(scale_bytes).repeat_interleave(32, dim=1)
    return (values * scales[:, :cols]).to(dtype)


def _has_b12x_moe() -> bool:
    return (
        torch.cuda.is_available()
        and current_platform.is_device_capability_family(120)
        and B12xExperts._supports_current_device()
    )


def _count_fp4_negative_zeros(packed: torch.Tensor) -> int:
    low = (packed & 0x0F) == 0x08
    high = (packed & 0xF0) == 0x80
    return int(low.sum().item() + high.sum().item())


def _make_b12x_moe_kernel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
) -> mk.FusedMoEKernel:
    num_experts = w1.shape[0]
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_states.shape[1],
        intermediate_size=w2.shape[2] * 2,
        in_dtype=hidden_states.dtype,
        activation=activation,
    )
    experts = B12xExperts(moe_config, quant_config)
    experts.process_weights_after_loading(
        SimpleNamespace(
            activation=activation,
            w13_weight=w1,
            w2_weight=w2,
        )
    )
    return mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        experts,
    )


def _run_b12x_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    num_experts = w1.shape[0]
    kernel = _make_b12x_moe_kernel(
        hidden_states,
        w1,
        w2,
        topk,
        activation,
        quant_config,
    )
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )
    return kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


def _quant_config(weight_dtype: str, activation_dtype: str | None):
    scale = torch.ones(1, dtype=torch.float32)
    return FusedMoEQuantConfig.make(
        quant_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
    )


@pytest.mark.parametrize(
    "weight_dtype,activation_dtype,mode,source_format,w13_layout",
    [
        ("mxfp4", "mxfp8", "w4a8_mx", "fp4_e8m0_k32", "w31"),
        ("mxfp4", None, "w4a16", "fp4_e8m0_k32", "w31"),
        ("nvfp4", "nvfp4", "nvfp4", "modelopt_nvfp4", "w31"),
        ("nvfp4", "mxfp8", "w4a8_nvfp4", "modelopt_nvfp4", "w31"),
        ("nvfp4", None, "w4a16", "modelopt_nvfp4", "w13"),
    ],
)
def test_b12x_moe_quant_mode_contract(
    weight_dtype: str,
    activation_dtype: str | None,
    mode: str,
    source_format: str,
    w13_layout: str,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config(weight_dtype, activation_dtype),
    )

    assert experts._quant_mode() == mode
    assert experts._source_format() == source_format
    assert experts._w13_layout() == w13_layout


def test_b12x_moe_supports_only_tensor_parallel() -> None:
    parallel = FusedMoEParallelConfig.make_no_parallel()

    assert B12xExperts._supports_parallel_config(parallel)
    assert not B12xExperts._supports_parallel_config(
        replace(parallel, use_ep=True, ep_size=2)
    )
    all2all = replace(parallel, use_ep=True, dp_size=2)
    assert all2all.use_all2all_kernels
    assert not B12xExperts._supports_parallel_config(all2all)
    assert not B12xExperts._supports_parallel_config(
        replace(parallel, enable_eplb=True)
    )


def test_b12x_moe_rejects_unsupported_input_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=256,
        intermediate_size=64,
        in_dtype=torch.float32,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == "kernel does not support torch.float32 input/output dtype"


def test_b12x_moe_rejects_interleaved_swigluoai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=128,
        intermediate_size=64,
        activation=MoEActivation.SWIGLUOAI,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == "kernel does not support MoEActivation.SWIGLUOAI activation"


@pytest.mark.parametrize("activation_key", [kMxfp8Dynamic, kNvfp4Dynamic])
def test_b12x_moe_rejects_uninterleaved_swigluoai_for_w4a8(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=128,
        intermediate_size=64,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    weight_key = kMxfp4Static if activation_key == kMxfp8Dynamic else kNvfp4Static

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        weight_key,
        activation_key,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == "kernel does not support swigluoai_uninterleave with W4A8"


def test_b12x_moe_supports_uninterleaved_swigluoai_for_w4a16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=128,
        intermediate_size=64,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported
    assert reason is None


def test_b12x_moe_rejects_relu2_for_mxfp4_w4a8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=256,
        intermediate_size=64,
        activation=MoEActivation.RELU2_NO_MUL,
    )

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == "MXFP4 W4A8 supports only SiLU and SiTU"


def test_b12x_moe_rejects_unaligned_mxfp4_w4a8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == (
        "MXFP4 W4A8 requires hidden size divisible by 256 and per-rank "
        "intermediate size divisible by 32"
    )


def test_b12x_moe_rejects_mxfp4_scale_groups_crossing_tp_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=48)
    config.intermediate_size_per_partition = 64

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == (
        "MXFP4 requires the per-rank intermediate size to be divisible by 32"
    )


@pytest.mark.parametrize(
    "beta,linear_beta",
    [(3.0, 25.0), (4.0, 24.0), (None, None)],
)
def test_b12x_moe_rejects_nonstandard_situ_parameters(
    monkeypatch: pytest.MonkeyPatch,
    beta: float | None,
    linear_beta: float | None,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=256,
        intermediate_size=64,
        activation=MoEActivation.SITU,
    )
    config.activation_situ_beta = beta
    config.activation_situ_linear_beta = linear_beta

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason == "kernel supports only SiTU beta=4 and linear_beta=25"


def test_b12x_moe_supports_standard_situ_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        hidden_dim=256,
        intermediate_size=64,
        activation=MoEActivation.SITU,
    )
    config.activation_situ_beta = 4.0
    config.activation_situ_linear_beta = 25.0

    supported, reason = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        kMxfp4Static,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported
    assert reason is None


@pytest.mark.parametrize(
    "activation_key,force_a16,expected_backend",
    [
        (kMxfp8Dynamic, False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (None, False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (kMxfp8Dynamic, True, Mxfp4MoeBackend.B12X_MXFP4_BF16),
        (None, True, Mxfp4MoeBackend.B12X_MXFP4_BF16),
    ],
)
def test_explicit_b12x_mxfp4_selection(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
    force_a16: bool,
    expected_backend: Mxfp4MoeBackend,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(
        config,
        activation_key=activation_key,
    )

    assert backend == expected_backend
    assert experts_cls is B12xExperts


@pytest.mark.parametrize(
    "force_a16,expected_backend",
    [
        (False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (True, Mxfp4MoeBackend.B12X_MXFP4_BF16),
    ],
)
def test_deepseek_v4_b12x_activation_selection(
    monkeypatch: pytest.MonkeyPatch,
    force_a16: bool,
    expected_backend: Mxfp4MoeBackend,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == expected_backend
    assert experts_cls is B12xExperts


def test_b12x_mxfp4_falls_back_to_a16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "VLLM_B12X_MOE_FORCE_A16",
        False,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(config)

    assert backend == Mxfp4MoeBackend.B12X_MXFP4_BF16
    assert experts_cls is B12xExperts


@pytest.mark.parametrize(
    "activation_key,force_a16,expected_activation_key",
    [
        (kNvfp4Dynamic, False, kNvfp4Dynamic),
        (kMxfp8Dynamic, False, kMxfp8Dynamic),
        (None, False, None),
        (kNvfp4Dynamic, True, None),
        (kMxfp8Dynamic, True, None),
    ],
)
def test_explicit_b12x_nvfp4_selection(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
    force_a16: bool,
    expected_activation_key,
) -> None:
    selected_activation_keys = []

    def is_supported_config(cls, config, weight_key, activation_key, activation_format):
        selected_activation_keys.append(activation_key)
        return True, None

    monkeypatch.setattr(B12xExperts, "is_supported_config", is_supported_config)
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_nvfp4_moe_backend(
        config,
        weight_key=kNvfp4Static,
        activation_key=activation_key,
    )

    assert backend == NvFp4MoeBackend.B12X
    assert experts_cls is B12xExperts
    assert selected_activation_keys == [expected_activation_key]


@pytest.mark.parametrize(
    "force_a16,expected_quant_dtype", [(False, "nvfp4"), (True, None)]
)
def test_b12x_nvfp4_force_a16_updates_quant_config(
    monkeypatch: pytest.MonkeyPatch,
    force_a16: bool,
    expected_quant_dtype,
) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FORCE_A16",
        force_a16,
    )
    scale = torch.ones(1)

    quant_config = nvfp4_oracle.make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.B12X,
        w13_scale=scale,
        w2_scale=scale,
        w13_scale_2=scale,
        w2_scale_2=scale,
        a13_scale=scale,
        a2_scale=scale,
    )

    assert quant_config.quant_dtype == expected_quant_dtype


def test_b12x_nvfp4_force_a16_updates_weight_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "VLLM_B12X_MOE_FORCE_A16",
        True,
    )
    reorder_w13 = None

    def prepare_for_b12x(**kwargs):
        nonlocal reorder_w13
        reorder_w13 = kwargs["reorder_w13"]
        return (
            kwargs["w13"],
            kwargs["w13_scale"],
            kwargs["w13_scale_2"],
            kwargs["a13_scale"],
            kwargs["w2"],
            kwargs["w2_scale"],
            kwargs["w2_scale_2"],
            kwargs["a2_scale"],
        )

    monkeypatch.setattr(
        nvfp4_oracle,
        "prepare_nvfp4_moe_layer_for_b12x",
        prepare_for_b12x,
    )
    tensor = torch.ones(1)

    nvfp4_oracle.convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=NvFp4MoeBackend.B12X,
        layer=SimpleNamespace(),
        w13=tensor,
        w13_scale=tensor,
        w13_scale_2=tensor,
        a13_scale=tensor,
        w2=tensor,
        w2_scale=tensor,
        w2_scale_2=tensor,
        a2_scale=tensor,
        is_act_and_mul=True,
    )

    assert reorder_w13 is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b12x_nvfp4_preparation_pads_each_gated_half() -> None:
    device = torch.device("cuda")
    num_experts, hidden_size, intermediate_size = 2, 64, 48
    w13 = torch.ones(
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.ones(
        num_experts,
        2 * intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2 = torch.ones(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.ones(
        num_experts,
        hidden_size,
        intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    global_scale = torch.ones(num_experts, device=device)
    input_scale = torch.tensor([[1.0, 2.0], [3.0, 1.0]], device=device)

    prepared = prepare_nvfp4_moe_layer_for_b12x(
        w13,
        w13_scale,
        global_scale,
        input_scale,
        w2,
        w2_scale,
        global_scale,
        input_scale,
        is_act_and_mul=True,
    )

    prepared_w13, prepared_w13_scale, _, prepared_a13 = prepared[:4]
    prepared_w2, prepared_w2_scale, _, prepared_a2 = prepared[4:]
    assert prepared_w13.shape == (num_experts, 128, hidden_size // 2)
    assert prepared_w13_scale.shape == (num_experts, 128, hidden_size // 16)
    assert prepared_w2.shape == (num_experts, hidden_size, 32)
    assert prepared_w2_scale.shape == (num_experts, 128, 4)
    torch.testing.assert_close(prepared_a13, torch.tensor([2.0, 3.0], device=device))
    torch.testing.assert_close(prepared_a2, torch.tensor([2.0, 3.0], device=device))


def test_b12x_moe_warmup_counts_cover_serving_range() -> None:
    assert b12x_moe._b12x_moe_warmup_token_counts(
        max_tokens=10,
        token_counts=(3, 8, 12, 0),
    ) == (1, 2, 3, 4, 8, 10)


def test_b12x_moe_uses_minimax_swiglu_parameters() -> None:
    config = make_dummy_moe_config(
        hidden_dim=128,
        intermediate_size=64,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    config.swiglu_limit = 7.0
    config.swiglu_alpha = 1.702
    config.swiglu_beta = 1.0
    experts = B12xExperts(config, _quant_config("mxfp4", None))

    assert experts._swiglu_params(config.activation) == (7.0, 1.702, 1.0)


def test_b12x_moe_warmup_runs_each_planner_regime_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    meta = SimpleNamespace(
        w1=torch.empty(0),
        w2=torch.empty(0),
        activation=MoEActivation.SILU,
        quant_mode="w4a16",
        num_experts=4,
        hidden_size=128,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        topk=2,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    prepared = SimpleNamespace()
    planned_tokens = []
    launched_tokens = []

    monkeypatch.setattr(experts, "_warmup_metadata", lambda layer: meta)
    monkeypatch.setattr(experts, "_prepare_experts", lambda **kwargs: prepared)

    def fake_execution_plan(**kwargs):
        tokens = kwargs["tokens"]
        if tokens <= 2:
            signature = ("micro", "decode")
        elif tokens <= 4:
            signature = ("dynamic", "small")
        else:
            signature = ("dynamic", "large")
        return SimpleNamespace(
            implementation=signature[0],
            execution=signature[1],
        )

    def fake_plan(**kwargs):
        planned_tokens.append(kwargs["tokens"])
        return SimpleNamespace(
            scratch_specs=lambda: [SimpleNamespace(dtype=torch.uint8, shape=(64,))]
        )

    def fake_run(**kwargs):
        launched_tokens.append(kwargs["hidden_states"].shape[0])

    monkeypatch.setattr(b12x_moe, "_b12x_moe_execution_plan", fake_execution_plan)
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_plan", fake_run)
    monkeypatch.setattr(experts, "_plan", fake_plan)

    warmed = experts.warmup_launches(
        SimpleNamespace(),
        token_counts=(1, 2, 3, 4, 8),
    )

    assert warmed == 3
    assert planned_tokens == [1, 3, 8]
    assert launched_tokens == planned_tokens


def test_b12x_moe_warmup_deduplicates_identical_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    calls = []

    class RoutedExpertsStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = SimpleNamespace(
                moe_kernel=SimpleNamespace(fused_experts=experts)
            )

    class Holder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.routed_experts = RoutedExpertsStub()

    monkeypatch.setattr(
        B12xExperts,
        "warmup_signature",
        lambda self, layer: ("identical",),
    )

    def fake_warmup(self, layer, *, token_counts):
        calls.append(tuple(token_counts))
        return 2

    monkeypatch.setattr(B12xExperts, "warmup_launches", fake_warmup)
    model = torch.nn.Sequential(Holder(), Holder())

    warmed = b12x_moe.warmup_b12x_moe(
        model,
        max_tokens=4,
        token_counts=(3,),
    )

    assert warmed == 2
    assert calls == [(1, 2, 3, 4)]


def test_b12x_moe_warmup_distinguishes_intermediate_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class RoutedExpertsStub(torch.nn.Module):
        def __init__(self, intermediate_size: int) -> None:
            super().__init__()
            experts = B12xExperts(
                make_dummy_moe_config(
                    hidden_dim=128,
                    intermediate_size=intermediate_size,
                ),
                _quant_config("mxfp4", None),
            )
            self.quant_method = SimpleNamespace(
                moe_kernel=SimpleNamespace(fused_experts=experts)
            )
            self.w13_weight = torch.empty(
                (4, 2 * intermediate_size, 64),
                dtype=torch.uint8,
            )
            self.w2_weight = torch.empty(
                (4, 128, intermediate_size // 2),
                dtype=torch.uint8,
            )

    class Holder(torch.nn.Module):
        def __init__(self, intermediate_size: int) -> None:
            super().__init__()
            self.routed_experts = RoutedExpertsStub(intermediate_size)

    def fake_warmup(self, layer, *, token_counts):
        calls.append(self.moe_config.intermediate_size)
        return 1

    monkeypatch.setattr(B12xExperts, "warmup_launches", fake_warmup)
    model = torch.nn.Sequential(Holder(64), Holder(128))

    warmed = b12x_moe.warmup_b12x_moe(
        model,
        max_tokens=4,
        token_counts=(3,),
    )

    assert warmed == 2
    assert calls == [64, 128]


def test_b12x_source_release_preserves_prepared_storage_owner() -> None:
    layer = torch.nn.Module()
    for name, shape in (
        ("w13_weight", (4, 32, 16)),
        ("w2_weight", (4, 64, 8)),
        ("w13_weight_scale", (4, 32, 2)),
        ("w2_weight_scale", (4, 64, 1)),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.empty(shape, dtype=torch.uint8),
                requires_grad=False,
            ),
        )
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    owner = SimpleNamespace(
        w1_fp4=layer.w13_weight,
        w2_fp4=layer.w2_weight,
        w1_blockscale=layer.w13_weight_scale,
        w2_blockscale=layer.w2_weight_scale,
    )
    experts._prepared_experts = owner
    owner_tensors = (
        owner.w1_fp4,
        owner.w2_fp4,
        owner.w1_blockscale,
        owner.w2_blockscale,
    )
    owner_ptrs = tuple(tensor.untyped_storage().data_ptr() for tensor in owner_tensors)

    experts._release_source_parameters(layer)
    experts._release_source_parameters(layer)

    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0
    assert layer.w13_weight_scale.numel() == 0
    assert layer.w2_weight_scale.numel() == 0
    assert (
        tuple(tensor.untyped_storage().data_ptr() for tensor in owner_tensors)
        == owner_ptrs
    )


def test_b12x_moe_reload_reuses_prepared_tensor_addresses() -> None:
    @dataclass(frozen=True)
    class Prepared:
        weight: torch.Tensor
        contract: tuple[str, ...]

    layer = torch.nn.Module()
    previous = Prepared(torch.tensor([1.0, 2.0]), ("w4a16", "bf16"))
    replacement = Prepared(torch.tensor([3.0, 4.0]), ("w4a16", "bf16"))
    layer._b12x_prepared_experts = previous
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    weight_ptr = previous.weight.data_ptr()

    reused = experts._reuse_prepared_storage(layer, replacement)

    assert reused is previous
    assert experts._prepared_experts is previous
    assert layer._b12x_prepared_experts is previous
    assert previous.weight.data_ptr() == weight_ptr
    torch.testing.assert_close(previous.weight, replacement.weight)


def test_b12x_moe_rejects_router_weight_on_input_for_w4a8() -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=256, intermediate_size=64),
        _quant_config("mxfp4", "mxfp8"),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=True,
    )

    with pytest.raises(
        ValueError,
        match="apply_router_weight_on_input only with W4A16",
    ):
        experts.process_weights_after_loading(layer)


def test_b12x_moe_workspace_uses_prepared_router_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=128, intermediate_size=64),
        _quant_config("mxfp4", None),
    )
    prepared = SimpleNamespace(
        plan=SimpleNamespace(discards_source_parameters=False),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=True,
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
    )
    monkeypatch.setattr(experts, "_prepare_experts", lambda **kwargs: prepared)
    planned = []

    def fake_plan(**kwargs):
        planned.append(kwargs)
        return SimpleNamespace(
            scratch_specs=lambda: [SimpleNamespace(dtype=torch.uint8, shape=(64,))]
        )

    monkeypatch.setattr(experts, "_plan", fake_plan)

    experts.process_weights_after_loading(layer)
    experts.workspace_shapes(
        8,
        128,
        128,
        2,
        4,
        4,
        None,
        MoEActivation.SILU,
    )

    assert planned == [
        {
            "tokens": 8,
            "topk": 2,
            "activation": MoEActivation.SILU,
            "apply_router_weight_on_input": True,
        }
    ]


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires B12X MoE on SM120")
@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL],
)
@torch.inference_mode()
def test_b12x_nvfp4_w4a16_matches_torch(
    activation: MoEActivation,
    workspace_init,
) -> None:
    set_random_seed(7)
    tokens, intermediate_size, hidden_size = 16, 128, 512
    num_experts, topk = 4, 2
    dtype = torch.bfloat16

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = (
            torch.randn((tokens, hidden_size), device="cuda", dtype=dtype) / 10
        )
        w1_rows = 2 * intermediate_size if activation.is_gated else intermediate_size
        w1 = (
            torch.randn(
                (num_experts, w1_rows, hidden_size),
                device="cuda",
                dtype=dtype,
            )
            / 15
        )
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size),
                device="cuda",
                dtype=dtype,
            )
            / 15
        )
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)
        unit_scale = torch.ones(num_experts, device="cuda", dtype=torch.float32)

        prepared = prepare_nvfp4_moe_layer_for_b12x(
            w1_q,
            w1_scale,
            1.0 / w1_global_scale,
            unit_scale,
            w2_q,
            w2_scale,
            1.0 / w2_global_scale,
            unit_scale,
            is_act_and_mul=activation.is_gated,
            reorder_w13=activation.is_gated,
        )
        w1_b12x, w1_scale_b12x, w1_alpha = prepared[:3]
        w2_b12x, w2_scale_b12x, w2_alpha = prepared[4:7]
        assert _count_fp4_negative_zeros(w1_b12x) > 0
        assert _count_fp4_negative_zeros(w2_b12x) > 0
        quant_config = nvfp4_w4a16_moe_quant_config(
            g1_alphas=w1_alpha,
            g2_alphas=w2_alpha,
            w1_scale=w1_scale_b12x,
            w2_scale=w2_scale_b12x,
        )
        score = torch.randn((tokens, num_experts), device="cuda", dtype=dtype)
        output = _run_b12x_moe(
            hidden_states,
            w1_b12x,
            w2_b12x,
            score,
            topk,
            activation,
            quant_config,
        )
        assert _count_fp4_negative_zeros(w1_b12x) == 0
        assert _count_fp4_negative_zeros(w2_b12x) == 0

        w1_ref = torch.empty_like(w1)
        w2_ref = torch.empty_like(w2)
        for expert in range(num_experts):
            w1_ref[expert] = _dequantize_nvfp4_linear(
                w1_q[expert],
                w1_scale[expert],
                w1_global_scale[expert],
                dtype,
            )
            w2_ref[expert] = _dequantize_nvfp4_linear(
                w2_q[expert],
                w2_scale[expert],
                w2_global_scale[expert],
                dtype,
            )
        reference = torch_moe(
            hidden_states,
            w1_ref,
            w2_ref,
            score,
            topk,
            activation=activation,
        )

        torch.testing.assert_close(output, reference, atol=2e-1, rtol=2e-1)
        cosine = torch.nn.functional.cosine_similarity(
            output.flatten().float(), reference.flatten().float(), dim=0
        )
        assert cosine > 0.99, (
            f"cosine={cosine.item():.4f}, "
            f"output_norm={output.float().norm().item():.4f}, "
            f"reference_norm={reference.float().norm().item():.4f}"
        )


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires B12X MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype",
    [
        ("mxfp4", "mxfp8"),
        ("nvfp4", "nvfp4"),
        ("nvfp4", "mxfp8"),
    ],
)
@torch.inference_mode()
def test_b12x_dynamic_fp4_modes_match_torch(
    weight_dtype: str,
    activation_dtype: str,
    workspace_init,
) -> None:
    set_random_seed(19)
    tokens, intermediate_size, hidden_size = 16, 128, 512
    num_experts, topk = 4, 2
    dtype = torch.bfloat16

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = (
            torch.randn((tokens, hidden_size), device="cuda", dtype=dtype) / 10
        )
        w1 = (
            torch.randn(
                (num_experts, 2 * intermediate_size, hidden_size),
                device="cuda",
                dtype=dtype,
            )
            / 15
        )
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size),
                device="cuda",
                dtype=dtype,
            )
            / 15
        )
        nvfp4_input_scale = torch.full(
            (num_experts,),
            1.0 / 1024.0,
            device="cuda",
            dtype=torch.float32,
        )
        if weight_dtype == "mxfp4":
            w1_q, w1_scale = _quantize_mxfp4(w1)
            w2_q, w2_scale = _quantize_mxfp4(w2)
            w1_ref = torch.stack(
                [
                    _dequantize_mxfp4(w1_q[e], w1_scale[e], dtype)
                    for e in range(num_experts)
                ]
            )
            w2_ref = torch.stack(
                [
                    _dequantize_mxfp4(w2_q[e], w2_scale[e], dtype)
                    for e in range(num_experts)
                ]
            )
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
            w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)
            w1_ref = torch.stack(
                [
                    _dequantize_nvfp4_linear(
                        w1_q[e],
                        w1_scale[e],
                        w1_global_scale[e],
                        dtype,
                    )
                    for e in range(num_experts)
                ]
            )
            w2_ref = torch.stack(
                [
                    _dequantize_nvfp4_linear(
                        w2_q[e],
                        w2_scale[e],
                        w2_global_scale[e],
                        dtype,
                    )
                    for e in range(num_experts)
                ]
            )
            prepared = prepare_nvfp4_moe_layer_for_b12x(
                w1_q,
                w1_scale,
                1.0 / w1_global_scale,
                nvfp4_input_scale,
                w2_q,
                w2_scale,
                1.0 / w2_global_scale,
                nvfp4_input_scale,
                is_act_and_mul=True,
            )
            w1_q, w1_scale, w1_alpha, a1_scale = prepared[:4]
            w2_q, w2_scale, w2_alpha, a2_scale = prepared[4:]
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                a1_gscale=1.0 / a1_scale,
                a2_gscale=1.0 / a2_scale,
            )

        score = torch.randn((tokens, num_experts), device="cuda", dtype=dtype)
        reference = torch_moe(hidden_states, w1_ref, w2_ref, score, topk)
        output = _run_b12x_moe(
            hidden_states,
            w1_q,
            w2_q,
            score,
            topk,
            MoEActivation.SILU,
            quant_config,
        )

        if activation_dtype == "nvfp4":
            topk_weights, topk_ids, _ = fused_topk(
                hidden_states, score, topk, renormalize=False
            )
            reference = _nvfp4_activation_reference(
                hidden_states,
                w1_ref,
                w2_ref,
                topk_weights,
                topk_ids,
                quant_config.a1_gscale,
                quant_config.a2_gscale,
            )

        torch.testing.assert_close(output, reference, atol=2e-1, rtol=2e-1)
        cosine = torch.nn.functional.cosine_similarity(
            output.flatten().float(),
            reference.flatten().float(),
            dim=0,
        )
        assert cosine > 0.99


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires B12X MoE on SM120")
@torch.inference_mode()
def test_b12x_mxfp4_w4a16_matches_torch(workspace_init) -> None:
    set_random_seed(11)
    tokens, intermediate_size, hidden_size = 16, 128, 512
    num_experts, topk = 4, 2
    dtype = torch.bfloat16

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = (
            torch.randn((tokens, hidden_size), device="cuda", dtype=dtype) / 10
        )
        w1 = (
            torch.randn(
                (num_experts, 2 * intermediate_size, hidden_size),
                device="cuda",
                dtype=dtype,
            )
            / 15
        )
        w2 = (
            torch.randn(
                (num_experts, hidden_size, intermediate_size),
                device="cuda",
                dtype=dtype,
            )
            / 15
        )
        w1_q, w1_scale = _quantize_mxfp4(w1)
        w2_q, w2_scale = _quantize_mxfp4(w2)
        w1_ref = torch.empty_like(w1)
        w2_ref = torch.empty_like(w2)
        for expert in range(num_experts):
            w1_ref[expert] = _dequantize_mxfp4(w1_q[expert], w1_scale[expert], dtype)
            w2_ref[expert] = _dequantize_mxfp4(w2_q[expert], w2_scale[expert], dtype)
        quant_config = mxfp4_w4a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        score = torch.randn((tokens, num_experts), device="cuda", dtype=dtype)
        reference = torch_moe(hidden_states, w1_ref, w2_ref, score, topk)
        output = _run_b12x_moe(
            hidden_states,
            w1_q,
            w2_q,
            score,
            topk,
            MoEActivation.SILU,
            quant_config,
        )

        torch.testing.assert_close(output, reference, atol=2e-1, rtol=2e-1)
        cosine = torch.nn.functional.cosine_similarity(
            output.flatten().float(), reference.flatten().float(), dim=0
        )
        assert cosine > 0.99, (
            f"cosine={cosine.item():.4f}, "
            f"output_norm={output.float().norm().item():.4f}, "
            f"reference_norm={reference.float().norm().item():.4f}"
        )


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires B12X MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype",
    [
        ("mxfp4", None),
        ("mxfp4", "mxfp8"),
        ("nvfp4", None),
        ("nvfp4", "nvfp4"),
        ("nvfp4", "mxfp8"),
    ],
)
@torch.inference_mode()
def test_b12x_moe_cuda_graph_replay(
    weight_dtype: str,
    activation_dtype: str | None,
    workspace_init,
) -> None:
    from vllm.v1.worker.workspace import lock_workspace

    set_random_seed(23)
    tokens = 4
    if weight_dtype == "nvfp4" and activation_dtype is not None:
        intermediate_size, hidden_size = 1024, 4096
    else:
        intermediate_size, hidden_size = 128, 512
    num_experts, topk = 4, 2
    hidden_states = (
        torch.randn(
            (tokens, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 10
    )
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 15
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 15
    )
    if weight_dtype == "mxfp4":
        w1_q, w1_scale = _quantize_mxfp4(w1)
        w2_q, w2_scale = _quantize_mxfp4(w2)
        if activation_dtype is None:
            quant_config = mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
    else:
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)
        input_scale = torch.full(
            (num_experts,),
            1.0 if activation_dtype is None else 1.0 / 1024.0,
            device="cuda",
            dtype=torch.float32,
        )
        prepared = prepare_nvfp4_moe_layer_for_b12x(
            w1_q,
            w1_scale,
            1.0 / w1_global_scale,
            input_scale,
            w2_q,
            w2_scale,
            1.0 / w2_global_scale,
            input_scale,
            is_act_and_mul=True,
            reorder_w13=activation_dtype is None,
        )
        w1_q, w1_scale, w1_alpha, a1_scale = prepared[:4]
        w2_q, w2_scale, w2_alpha, a2_scale = prepared[4:]
        if activation_dtype is None:
            quant_config = nvfp4_w4a16_moe_quant_config(
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                a1_gscale=1.0 / a1_scale,
                a2_gscale=1.0 / a2_scale,
            )

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        kernel = _make_b12x_moe_kernel(
            hidden_states,
            w1_q,
            w2_q,
            topk,
            MoEActivation.SILU,
            quant_config,
        )
        score = torch.randn(
            (tokens, num_experts),
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()

        def apply() -> torch.Tensor:
            return kernel.apply(
                hidden_states=hidden_states,
                w1=w1_q,
                w2=w2_q,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        expected = apply().clone()
        lock_workspace()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            actual = apply()
        graph.replay()
        torch.accelerator.synchronize()

        assert torch.isfinite(expected).all()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
