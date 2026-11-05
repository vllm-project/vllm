# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online block Hadamard rotation of Quark MXFP4 MoE experts."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    InputHadamardOCPMXEmulationExperts,
    OCP_MXQuantizationEmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_kernel,
)
from vllm.model_executor.layers.quantization.input_hadamard import (
    block_hadamard_rotate,
    hadamard_matrix,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize


def _quantize_like_quark(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack with Quark's MXFP4 quantizer, the format emulation dequantizes."""
    try:
        from quark.torch.export.nn.modules.realquantizer import (
            StaticScaledRealQuantizer,
        )
        from quark.torch.quantization.config.config import FP4PerGroupSpec
    except ImportError:
        return mxfp4_quantize(weight)
    spec = FP4PerGroupSpec(
        ch_axis=-1,
        group_size=32,
        scale_format="e8m0",
        scale_calculation_mode="even",
        is_dynamic=False,
    ).to_quantization_spec()
    quantizer = StaticScaledRealQuantizer(
        qspec=spec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=weight.dtype,
        device=weight.device,
    )
    flat = weight.reshape(-1, weight.shape[-1])
    observer = spec.observer_cls(spec, device=weight.device)
    observer(flat)
    scale, _ = observer._calculate_qparams()
    quantizer.scale = scale
    packed = quantizer.to_real_quantize_params(flat)
    quantizer.maybe_convert_and_transpose_scale()
    return (
        packed.view(*weight.shape[:-1], weight.shape[-1] // 2),
        quantizer.scale.view(*weight.shape[:-1], weight.shape[-1] // 32),
    )
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

_PREFIX = "language_model.model.layers.3.block_sparse_moe.experts"
_SHARED = "language_model.model.layers.3.block_sparse_moe.shared_experts"
_DENSE = "language_model.model.layers.0.mlp"


def _config() -> QuarkConfig:
    return QuarkConfig(
        {
            "global_quant_config": {},
            "layer_type_quant_config": {},
            "exclude": [],
            "algo_config": [{"name": "rotation"}],
        }
    )


def _expert_keys(experts: range, projs: tuple[str, ...] = ("w1", "w2", "w3")) -> dict:
    return {
        f"{_PREFIX}.{idx}.{proj}": 128
        for idx in experts
        for proj in projs
    }


def _moe_config(intermediate: int = 256, hidden: int = 256) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=4,
        experts_per_token=1,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        device=current_platform.device_type,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )


def test_moe_rotation_groups_w1_w2_w3():
    config = _config()
    config.input_rotation_sizes = _expert_keys(range(4))
    # A dense layer recorded by the linear scan must stay there.
    config.input_rotation_sizes[f"{_DENSE}.gate_proj"] = 128
    config._collect_moe_input_rotations()

    assert config.moe_input_rotation_sizes == {_PREFIX: 128}
    assert config._moe_input_rotation_size(_PREFIX) == 128
    assert config._moe_input_rotation_size(_SHARED) is None
    assert config._moe_input_rotation_size(_DENSE) is None
    assert config.input_rotation_sizes == {f"{_DENSE}.gate_proj": 128}


def test_moe_rotation_rejects_missing_w2_and_non_sylvester(default_vllm_config):
    default_vllm_config.model_config = ModelConfig()
    config = _config()
    config.input_rotation_sizes = _expert_keys(range(2), ("w1", "w3"))
    with pytest.raises(NotImplementedError, match="w1, w2 and w3"):
        config._collect_moe_input_rotations()

    method = QuarkOCP_MX_MoEMethod(
        _moe_config(), kMxfp4Static, kMxfp4Dynamic, rotation_size=128
    )
    param = torch.nn.Parameter(
        torch.zeros(128, 128, dtype=torch.int8), requires_grad=False
    )
    signs = (hadamard_matrix(128, torch.device("cpu")) * 128**0.5).round()
    method._load_input_rotation(param, signs.to(torch.int8))
    flipped = signs.clone()
    flipped[0] = -flipped[0]
    with pytest.raises(NotImplementedError, match="Sylvester"):
        method._load_input_rotation(param, flipped.to(torch.int8))


def test_moe_rotation_rejects_indivisible_shard_and_fused_shared(default_vllm_config):
    default_vllm_config.model_config = ModelConfig()
    with pytest.raises(ValueError, match="intermediate"):
        QuarkOCP_MX_MoEMethod(
            _moe_config(intermediate=192),
            kMxfp4Static,
            kMxfp4Dynamic,
            rotation_size=128,
        )

    method = QuarkOCP_MX_MoEMethod(
        _moe_config(), kMxfp4Static, kMxfp4Dynamic, rotation_size=128
    )
    layer = torch.nn.Module()
    layer.expert_map_manager = SimpleNamespace(num_fused_shared_experts=1)
    with pytest.raises(NotImplementedError, match="fused shared"):
        method.create_weights(
            layer, 4, 256, 256, torch.bfloat16, weight_loader=None
        )


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        gate_up.shape[0],
        gate_up.shape[-1] // 2,
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    torch.ops._C.silu_and_mul_with_clamp(out, gate_up.contiguous(), 7.0, 1.702, 1.0)
    return out


def _bf16_reference(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    rotate: bool,
) -> torch.Tensor:
    hadamard = hadamard_matrix(128, x.device)
    if rotate:
        x = block_hadamard_rotate(x, hadamard)
    rows = []
    for token in range(x.shape[0]):
        expert = int(topk_ids[token, 0])
        gate_up = (x[token].float() @ w13[expert].float().T).to(x.dtype)[None, :]
        mid = _swiglu(gate_up)
        if rotate:
            mid = block_hadamard_rotate(mid, hadamard)
        rows.append((mid.float() @ w2[expert].float().T).squeeze(0))
    return torch.stack(rows)


def _emulated(
    x: torch.Tensor,
    w13_q: torch.Tensor,
    w13_s: torch.Tensor,
    w2_q: torch.Tensor,
    w2_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_cls: type,
    rotation_size: int | None,
) -> torch.Tensor:
    quant = ocp_mx_moe_quant_config(
        quant_dtype="mxfp4",
        weight_dtype="mxfp4",
        w1_scale=w13_s,
        w2_scale=w2_s,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    if rotation_size is not None:
        quant.input_rotation_size = rotation_size
    init_workspace_manager(x.device)
    num_experts = w13_q.shape[0]
    kernel = make_mxfp4_moe_kernel(
        quant,
        FusedMoEConfig(
            num_experts=num_experts,
            experts_per_token=1,
            hidden_dim=x.shape[-1],
            intermediate_size=w13_q.shape[1],
            num_local_experts=num_experts,
            num_logical_experts=num_experts,
            activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            device=x.device,
            routing_method=RoutingMethodType.Renormalize,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            swiglu_limit=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        ),
        experts_cls,
        Mxfp4MoeBackend.EMULATION,
    )
    return kernel.apply(
        hidden_states=x,
        w1=w13_q,
        w2=w2_q,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want).norm() / want.norm()).item()


@pytest.mark.skipif(
    not current_platform.supports_mx(), reason="MXFP4 kernels need MX support."
)
def test_rotated_mxfp4_moe_matches_rotated_bf16():
    torch.manual_seed(0)
    device = current_platform.device_type
    num_experts, hidden, intermediate, tokens = 4, 256, 256, 8
    hadamard = hadamard_matrix(128, torch.device(device))
    with torch.device(device):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        w13 = torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
        w2 = torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16)
        w13 *= 0.02
        w2 *= 0.02
        w13_rot = block_hadamard_rotate(w13, hadamard)
        w2_rot = block_hadamard_rotate(w2, hadamard)
        w13_q, w13_s = _quantize_like_quark(w13_rot)
        w2_q, w2_s = _quantize_like_quark(w2_rot)
        topk_ids = torch.arange(tokens, dtype=torch.int32).remainder(num_experts)
        topk_ids = topk_ids[:, None]
        topk_weights = torch.ones(tokens, 1, dtype=torch.float32)

    rotated_ref = _bf16_reference(x, w13_rot, w2_rot, topk_ids, rotate=True)
    # Rotated weights without rotating the activation is a different product.
    skipped = _bf16_reference(x, w13_rot, w2_rot, topk_ids, rotate=False)
    got = _emulated(
        x,
        w13_q,
        w13_s,
        w2_q,
        w2_s,
        topk_ids,
        topk_weights,
        InputHadamardOCPMXEmulationExperts,
        128,
    )

    # Two MXFP4 GEMMs, so this is quantization error against the unquantized
    # rotated weights, not a second rotation.
    assert _rel(got, rotated_ref) < 0.35
    assert _rel(got, rotated_ref) < 0.5 * _rel(got, skipped)


@pytest.mark.skipif(
    not current_platform.supports_mx(), reason="MXFP4 kernels need MX support."
)
def test_moe_without_rotation_matches_unrotated_bf16():
    """Shared experts and the first dense layers have no input_rotation, so the
    expert path must reproduce the unrotated product."""
    torch.manual_seed(1)
    device = current_platform.device_type
    num_experts, hidden, intermediate, tokens = 4, 256, 256, 8
    with torch.device(device):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        w13 = torch.randn(num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
        w2 = torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16)
        w13 *= 0.02
        w2 *= 0.02
        w13_q, w13_s = _quantize_like_quark(w13)
        w2_q, w2_s = _quantize_like_quark(w2)
        topk_ids = torch.arange(tokens, dtype=torch.int32).remainder(num_experts)
        topk_ids = topk_ids[:, None]
        topk_weights = torch.ones(tokens, 1, dtype=torch.float32)

    plain_ref = _bf16_reference(x, w13, w2, topk_ids, rotate=False)
    got = _emulated(
        x,
        w13_q,
        w13_s,
        w2_q,
        w2_s,
        topk_ids,
        topk_weights,
        OCP_MXQuantizationEmulationTritonExperts,
        None,
    )
    assert _rel(got, plain_ref) < 0.35
