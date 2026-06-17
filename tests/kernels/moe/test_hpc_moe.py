# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import input_to_float8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(90):
    pytest.skip(
        reason="HPC attention requires compute capability >= SM90.",
        allow_module_level=True,
    )

from vllm.utils.hpc import has_hpc

if not has_hpc():
    pytest.skip(
        reason="HPC attention requires hpc module.",
        allow_module_level=True,
    )

from vllm.model_executor.layers.fused_moe.hpc_moe import HPCExperts

logger = init_logger(__name__)

NUM_EXPERTS = [128]
TOP_KS = [8]

MNK_FACTORS = [(128, 4096, 512)]

BLOCK_SIZE = [[128, 128]]

vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))


def calculate_errors(ref_tensor, real_tensor, eps=1e-6, top_k=10):
    """
    Calculate various error metrics between reference and real tensors

    Args:
        ref_tensor: Reference tensor (PyTorch tensor)
        real_tensor: Real tensor (PyTorch tensor with the same shape as ref_tensor)
        eps: Small value to prevent division by zero, default 1e-6
        top_k: Number of top largest errors to return, default 10

    Returns:
        dict: Dictionary containing the following metrics:
            - mean_abs_error: Mean Absolute Error
            - max_abs_error: Maximum Absolute Error
            - max_abs_error_ref: Reference value at the position of maximum absolute
                                error
            - max_abs_error_real: Real value at the position of maximum absolute error
            - max_abs_error_pos: Position coordinates of maximum absolute error
                                (as tuple)
            - mean_rel_error: Mean Relative Error
            - max_rel_error: Maximum Relative Error
            - max_rel_error_ref: Reference value at the position of maximum relative
                                error
            - max_rel_error_real: Real value at the position of maximum relative error
            - max_rel_error_pos: Position coordinates of maximum relative error
                                (as tuple)
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(ref_tensor, torch.Tensor) or not isinstance(
        real_tensor, torch.Tensor
    ):
        raise TypeError("Inputs must be PyTorch tensors")

    # Check if tensor shapes match
    if ref_tensor.shape != real_tensor.shape:
        raise ValueError("Reference and real tensors must have the same shape")

    # Calculate absolute errors
    abs_error = torch.abs(ref_tensor - real_tensor)

    # Mean Absolute Error
    mae = torch.mean(abs_error).item()

    # Get top K absolute errors and their positions
    num_elements = abs_error.numel()
    k = min(top_k, num_elements)

    # Flatten the error tensor and obtain the indices of the top k largest values
    abs_error_flat = abs_error.flatten()
    top_abs_values, top_abs_flat_indices = torch.topk(abs_error_flat, k, largest=True)

    # Convert to multidimensional coordinates and collect corresponding values
    top_abs_errors = []
    for val, idx in zip(top_abs_values, top_abs_flat_indices):
        pos = tuple(torch.unravel_index(idx, abs_error.shape))
        top_abs_errors.append(
            {
                "error_value": val.item(),
                "ref_value": ref_tensor[pos].item(),
                "real_value": real_tensor[pos].item(),
                "position": pos,
            }
        )

    # Calculate relative errors (with protection against division by zero)
    rel_error = abs_error / (
        torch.max(torch.abs(ref_tensor), torch.abs(real_tensor)) + eps
    )

    # Mean Relative Error
    mre = torch.mean(rel_error).item()

    # Get top K relative errors and their positions
    rel_error_flat = rel_error.flatten()
    top_rel_values, top_rel_flat_indices = torch.topk(rel_error_flat, k, largest=True)

    # Convert to multidimensional coordinates and collect corresponding values
    top_rel_errors = []
    for val, idx in zip(top_rel_values, top_rel_flat_indices):
        pos = tuple(torch.unravel_index(idx, rel_error.shape))
        top_rel_errors.append(
            {
                "error_value": val.item(),
                "ref_value": ref_tensor[pos].item(),
                "real_value": real_tensor[pos].item(),
                "position": pos,
            }
        )

    return {
        "mean_abs_error": mae,
        "top_abs_errors": top_abs_errors,
        "mean_rel_error": mre,
        "top_rel_errors": top_rel_errors,
    }


def errors_to_string(error_results, precision=6):
    """
    Convert error calculation results to a human-readable string

    Args:
        error_results: Dictionary returned by calculate_errors function
        precision: Number of decimal places to display, default 6

    Returns:
        str: Formatted string with error information
    """
    # Create the header section
    lines = [""]
    lines.append("=" * 80)
    lines.append("Error Analysis Results".center(80))
    lines.append("=" * 80)
    lines.append("")

    # Add mean error metrics
    lines.append("Mean Error Metrics:")
    lines.append("-" * 40)
    lines.append(
        f"Mean Absolute Error: {error_results['mean_abs_error']:.{precision}f}"
    )
    lines.append(
        f"Mean Relative Error: {error_results['mean_rel_error']:.{precision}f}"
    )
    lines.append("")

    # Add top absolute errors section
    lines.append(f"Top {len(error_results['top_abs_errors'])} Absolute Errors:")
    lines.append("-" * 80)
    # Header for the table
    abs_header = (
        "Rank".ljust(6)
        + "Error Value".ljust(16)
        + "Ref Value".ljust(16)
        + "Real Value".ljust(16)
        + "Position"
    )
    lines.append(abs_header)
    lines.append("-" * 80)

    # Add each top absolute error
    for i, err in enumerate(error_results["top_abs_errors"], 1):
        line = (
            f"{i:^6}"
            + f"{err['error_value']:.{precision}f}".ljust(16)
            + f"{err['ref_value']:.{precision}f}".ljust(16)
            + f"{err['real_value']:.{precision}f}".ljust(16)
            + f"{err['position']}"
        )
        lines.append(line)
    lines.append("")

    # Add top relative errors section
    lines.append(f"Top {len(error_results['top_rel_errors'])} Relative Errors:")
    lines.append("-" * 80)
    # Header for the table
    rel_header = (
        "Rank".ljust(6)
        + "Error Value".ljust(16)
        + "Ref Value".ljust(16)
        + "Real Value".ljust(16)
        + "Position"
    )
    lines.append(rel_header)
    lines.append("-" * 80)

    # Add each top relative error
    for i, err in enumerate(error_results["top_rel_errors"], 1):
        line = (
            f"{i:^6}"
            + f"{err['error_value']:.{precision}f}".ljust(16)
            + f"{err['ref_value']:.{precision}f}".ljust(16)
            + f"{err['real_value']:.{precision}f}".ljust(16)
            + f"{err['position']}"
        )
        lines.append(line)
    lines.append("")

    lines.append("=" * 80)

    # Join all lines into a single string
    return "\n".join(lines)


def allclose(ref_tensor, real_tensor, atol=1e-8, rtol=1e-5):
    assert ref_tensor.dtype == real_tensor.dtype
    assert ref_tensor.device == real_tensor.device
    assert ref_tensor.shape == real_tensor.shape
    is_true = torch.allclose(
        ref_tensor.to(torch.float32),
        real_tensor.to(torch.float32),
        atol=atol,
        rtol=rtol,
    )
    if not is_true:
        print(
            errors_to_string(
                calculate_errors(
                    ref_tensor.to(torch.float32), real_tensor.to(torch.float32)
                )
            )
        )
    return is_true


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = input_to_float8(a[i])
        if a_global_sf.numel() == 1:
            a_global_sf = a_global_sf.view(1, 1)
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int], use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()

    def _align(x: int, y: int) -> int:
        return cdiv(x, y) * y

    def _ceil_to_ue8m0(x: torch.Tensor):
        return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

    if x.dim() == 2:
        x = x.unsqueeze(0)

    xs = [None for _ in range(x.size(0))]
    sfs = [None for _ in range(x.size(0))]
    for e in range(x.size(0)):
        m, n = x[e].shape
        block_m, block_n = block_size
        x_padded = torch.zeros(
            (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
        )
        x_padded[:m, :n] = x[e]
        x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        _, fp8_max = get_fp8_min_max()
        sf = x_amax / fp8_max
        sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
        x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)

        x_scaled = x_scaled.view_as(x_padded)[:m, :n].contiguous()
        sf = sf.view(x_view.size(0), x_view.size(2))

        xs[e] = x_scaled
        sfs[e] = sf

    return torch.stack(xs), torch.stack(sfs)


@dataclass
class TestData:
    hidden_states: torch.Tensor
    w13_quantized: torch.Tensor
    w2_quantized: torch.Tensor
    a1_scale: torch.Tensor
    a2_scale: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    layer: torch.nn.Module

    @staticmethod
    def make_moe_tensors_8bit(
        m: int,
        k: int,
        n: int,
        e: int,
        is_trtllm: bool,
        block_shape: list[int] | None = None,
        activation: MoEActivation = MoEActivation.SILU,
    ) -> "TestData":
        is_gated = activation.is_gated

        hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 15
        w13 = (
            torch.randn(
                (e, (2 * n) if is_gated else n, k), device="cuda", dtype=torch.bfloat16
            )
            / 15
        )
        w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 15

        # Scale to fp8
        _, a1_scale = input_to_float8(hidden_states)
        a2_scale = torch.randn([], device="cuda", dtype=torch.float32)
        if block_shape:
            w13_quantized, w13_weight_scale = per_block_cast_to_fp8(
                w13, block_shape, False
            )
            w2_quantized, w2_weight_scale = per_block_cast_to_fp8(
                w2, block_shape, False
            )
        else:
            w13_quantized, w13_weight_scale = quant_fp8_per_tensor_batches(w13)
            w2_quantized, w2_weight_scale = quant_fp8_per_tensor_batches(w2)

        layer = torch.nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.w13_weight = w13_quantized.clone()
        layer.w2_weight = w2_quantized.clone()
        layer.w13_input_scale = a1_scale
        layer.w2_input_scale = a2_scale
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight_scale = w2_weight_scale
        layer.activation = activation
        # Setup dummy config.
        layer.moe_parallel_config = mk.FusedMoEParallelConfig.make_no_parallel()

        # flashinfer expects swapped rows for w13
        if is_gated:
            layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        if is_trtllm:
            rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
                layer.w13_weight, layer.w2_weight, is_gated
            )
        layer.custom_routing_function = Llama4MoE.custom_routing_function
        layer.routing_method_type = RoutingMethodType.Llama4
        layer.renormalize = False
        layer.intermediate_size_per_partition = n
        layer.ep_rank = 0
        layer.local_num_experts = e

        return TestData(
            hidden_states=hidden_states,
            w13_quantized=w13_quantized,
            w2_quantized=w2_quantized,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            layer=layer,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU])
def test_hpc_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    # init_workspace_manager(device="cuda:0")
    # monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")

    block_size = None
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=False, block_shape=block_size, activation=activation
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            g1_alphas=(td.w13_weight_scale * td.a1_scale).squeeze(),
            w2_scale=td.w2_weight_scale,
            g2_alphas=(td.w2_weight_scale * td.a2_scale).squeeze(),
            a1_scale=td.a1_scale,
            a1_gscale=td.a1_scale,
            a2_scale=td.a2_scale,
            a2_gscale=1.0 / td.a2_scale,
            per_act_token_quant=False,
            block_shape=block_size,
        )

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=quant_config,
        )
        torch.accelerator.synchronize()

        td.layer.dp_size = 1

        def get_fused_moe_quant_config(n: torch.nn.Module) -> FusedMoEQuantConfig:
            return quant_config

        td.layer.get_fused_moe_quant_config = get_fused_moe_quant_config
        td.layer.quant_method = td.layer

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
        )

        hpc_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            HPCExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        hidden_states = td.hidden_states
        hpc_output = hpc_kernel.apply(
            hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        torch.accelerator.synchronize()

        assert allclose(
            output.to(torch.float32), hpc_output.to(torch.float32), rtol=0.08, atol=0.1
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU])
def test_flashinfer_vs_hpc_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    # init_workspace_manager(device="cuda:0")
    # monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    block_size = None
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=False, block_shape=block_size, activation=activation
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            g1_alphas=(td.w13_weight_scale * td.a1_scale).squeeze(),
            w2_scale=td.w2_weight_scale,
            g2_alphas=(td.w2_weight_scale * td.a2_scale).squeeze(),
            a1_scale=td.a1_scale,
            a1_gscale=td.a1_scale,
            a2_scale=td.a2_scale,
            a2_gscale=1.0 / td.a2_scale,
            per_act_token_quant=False,
            block_shape=block_size,
        )

        td.layer.dp_size = 1

        def get_fused_moe_quant_config(n: torch.nn.Module) -> FusedMoEQuantConfig:
            return quant_config

        td.layer.get_fused_moe_quant_config = get_fused_moe_quant_config
        td.layer.quant_method = td.layer

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
        )

        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        flashinfer_cutlass_output = kernel.apply(
            td.hidden_states,
            td.layer.w13_weight,
            td.layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        torch.accelerator.synchronize()

        hpc_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            HPCExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        hidden_states = td.hidden_states
        hpc_output = hpc_kernel.apply(
            hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        torch.accelerator.synchronize()

        assert allclose(
            flashinfer_cutlass_output.to(torch.float32),
            hpc_output.to(torch.float32),
            rtol=0.08,
            atol=0.1,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU])
def test_flashinfer_vs_hpc_moe_fp8_blockwise_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    # init_workspace_manager(device="cuda:0")
    # monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    block_size = [128, 128]
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=False, block_shape=block_size, activation=activation
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            g1_alphas=None,
            w2_scale=td.w2_weight_scale,
            g2_alphas=None,
            a1_scale=None,
            a1_gscale=None,
            a2_scale=None,
            a2_gscale=None,
            per_act_token_quant=False,
            block_shape=block_size,
        )

        td.layer.dp_size = 1

        def get_fused_moe_quant_config(n: torch.nn.Module) -> FusedMoEQuantConfig:
            return quant_config

        td.layer.get_fused_moe_quant_config = get_fused_moe_quant_config
        td.layer.quant_method = td.layer

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
        )

        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        flashinfer_cutlass_output = kernel.apply(
            td.hidden_states,
            td.layer.w13_weight,
            td.layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        torch.accelerator.synchronize()

        hpc_kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            HPCExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        hidden_states = td.hidden_states
        hpc_output = hpc_kernel.apply(
            hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )
        torch.accelerator.synchronize()

        assert allclose(
            flashinfer_cutlass_output.to(torch.float32),
            hpc_output.to(torch.float32),
            rtol=0.08,
            atol=0.1,
        )
