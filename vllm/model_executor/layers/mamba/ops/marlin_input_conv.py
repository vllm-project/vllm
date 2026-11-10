# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB10 Mamba projection ownership and specialized projection/conv launches."""

import torch
from vllm.model_executor.kernels.linear import is_supported_and_can_implement_kernel
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.fusion.quant_activation import expose_input_quant_key
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.modelopt import (
    FormatScheme,
    ModelOptLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    get_marlin_workspace,
    should_use_atomic_add_reduce,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm import _custom_ops as ops
from vllm import envs


def select_mamba_marlin_ownership(layer, config) -> None:
    """Choose only Mamba2's projection kernels before normal weight packing.

    The current quantization API stores the per-layer kernel on its method.
    Canonical checkpoint parameters already exist here; post-load packing has
    not run. Other linears, MoE kernels and the shared configuration are untouched.
    """
    if (
        not current_platform.is_cuda()
        or envs.VLLM_BATCH_INVARIANT
        or config.lora_config is not None
        or is_weights_pre_processed()
    ):
        return
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() != 121:
        return
    for projection in (layer.in_proj, layer.out_proj):
        method = projection.quant_method
        if type(getattr(method, "kernel", None)) is MarlinNvFp4LinearKernel:
            continue
        if not (
            type(method) is ModelOptLinearMethod
            and method.spec.weight == kNvfp4Static
            and method.spec.activation is None
            and method.akey is None
            and projection.params_dtype in (torch.float16, torch.bfloat16)
            and isinstance(
                getattr(method.kernel, "config", None), NvFp4LinearLayerConfig
            )
        ):
            continue
        supported, _ = is_supported_and_can_implement_kernel(
            MarlinNvFp4LinearKernel, method.kernel.config, 121
        )
        if not supported:
            continue
        method.kernel = MarlinNvFp4LinearKernel(method.kernel.config)
        if hasattr(projection, "_input_quant_key"):
            del projection._input_quant_key
        expose_input_quant_key(projection, method.kernel)


# Qualified GB10 component measurements cover all B1–4/T1–8 shapes with M<=16.
# Larger M keeps ordinary Marlin; structural and padding admission stays below.
def _batched_shape_is_profitable(batches: int, query_tokens: int) -> bool:
    return (
        1 <= batches <= 4
        and 1 <= query_tokens <= 8
        and batches * query_tokens <= 16  # Total input-projection GEMM rows.
    )


def _unmodified(module, methods: tuple[str, ...]) -> bool:
    return (
        all(name not in module.__dict__ for name in methods)
        and not getattr(module, "_forward_hooks", ())
        and not getattr(module, "_forward_pre_hooks", ())
        and not getattr(module, "_backward_hooks", ())
        and not getattr(module, "_backward_pre_hooks", ())
        and getattr(module, "_compiled_call_impl", None) is None
    )


def _global_hooks_unmodified() -> bool:
    for name in (
        "_global_forward_hooks",
        "_global_forward_pre_hooks",
        "_global_backward_hooks",
        "_global_backward_pre_hooks",
    ):
        hooks = getattr(torch.nn.modules.module, name, None)
        if hooks is None or hooks:
            return False
    return True


def can_use_marlin_input_conv(layer, hidden_states, mup_vector) -> bool:
    """Keep hardware/LoRA admission cached and preserve ordinary module hooks."""
    if not _global_hooks_unmodified():
        return False
    linear = layer.in_proj
    if not (
        layer._marlin_input_conv_allowed
        and mup_vector is None
        and 0 <= layer.num_spec <= 7
        and layer.num_heads == 128
        and layer.head_dim == 64
        and layer.ssm_state_size == 128
        and layer.n_groups == 8
        and layer.use_replayssm
        and layer.use_flashinfer_replayssm
        and type(linear) is MergedColumnParallelLinear
        and _unmodified(linear, ("forward", "__call__", "_call_impl"))
        and linear.bias is None
        and linear.tp_size == 1
        and linear.return_bias
        and linear.input_size_per_partition == 4096
        and linear.output_size_per_partition == 18560
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == 4096
        and hidden_states.is_contiguous()
    ):
        return False
    method = linear.quant_method
    if not (
        type(method) is ModelOptLinearMethod
        and _unmodified(method, ("apply",))
        and method.spec.activation is None
        and method.akey is None
        and getattr(linear, "is_w4a16_nvfp4", False)
        and type(method.kernel) is MarlinNvFp4LinearKernel
        and _unmodified(method.kernel, ("apply_weights",))
        and "apply" not in method.fmt.__dict__
        and type(method.fmt).apply is FormatScheme.apply
    ):
        return False
    return (
        linear.weight.dtype == torch.int32
        and linear.weight.shape == (256, 37120)
        and linear.weight.is_contiguous()
        and linear.weight_scale.dtype == torch.float8_e4m3fn
        and linear.weight_scale.shape == (256, 18560)
        and linear.weight_scale.is_contiguous()
        and linear.weight_global_scale.dtype == torch.float32
        and linear.weight_global_scale.numel() == 1
        and linear.weight_global_scale.is_contiguous()
    )


def can_use_marlin_output(layer, hidden_states, *, check_rows: bool = True) -> bool:
    """Select only the measured Mamba output launch, without changing ownership."""
    if not layer._marlin_output_profile_allowed or not _global_hooks_unmodified():
        return False
    linear = layer.out_proj
    if not (
        type(linear) is RowParallelLinear
        and _unmodified(linear, ("forward", "__call__", "_call_impl"))
        and linear.bias is None
        and linear.tp_size == 1
        and linear.return_bias
        and linear.input_size_per_partition == 8192
        and linear.output_size_per_partition == 4096
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == 8192
        and (not check_rows or hidden_states.shape[0] == 8)
        and hidden_states.is_contiguous()
    ):
        return False
    method = linear.quant_method
    if not (
        type(method) is ModelOptLinearMethod
        and _unmodified(method, ("apply",))
        and method.spec.activation is None
        and method.akey is None
        and getattr(linear, "is_w4a16_nvfp4", False)
        and type(method.kernel) is MarlinNvFp4LinearKernel
        and _unmodified(method.kernel, ("apply_weights",))
        and "apply" not in method.fmt.__dict__
        and type(method.fmt).apply is FormatScheme.apply
    ):
        return False
    return (
        linear.weight.dtype == torch.int32
        and linear.weight.shape == (512, 8192)
        and linear.weight.is_contiguous()
        and linear.weight.device == hidden_states.device
        and linear.weight_scale.dtype == torch.float8_e4m3fn
        and linear.weight_scale.shape == (512, 4096)
        and linear.weight_scale.is_contiguous()
        and linear.weight_scale.device == hidden_states.device
        and linear.weight_global_scale.dtype == torch.float32
        and linear.weight_global_scale.numel() == 1
        and linear.weight_global_scale.is_contiguous()
        and linear.weight_global_scale.device == hidden_states.device
    )


def run_marlin_output(layer, hidden_states):
    linear = layer.out_proj
    return ops.mamba_marlin_output(
        hidden_states,
        linear.weight,
        linear.weight_scale,
        linear.weight_global_scale,
        get_marlin_workspace(hidden_states.device),
    )


def _first_cache_indices(indices, batches=1):
    """Match public speculative conv/FI first-column ownership, without a copy."""
    if indices is None:
        return None
    if indices.ndim == 1 and indices.shape == (batches,):
        return indices
    if indices.ndim == 2 and indices.shape[0] == batches and 1 <= indices.shape[1] <= 8:
        return indices[:, 0]
    return None


def _supported_checkpoint_cache(cache):
    if len(cache) == 5:
        return cache[1].dtype in (torch.float16, torch.bfloat16)
    if len(cache) != 6 or cache[1].dtype not in (
        torch.float8_e4m3fn,
        torch.int8,
    ):
        return False
    state, scales = cache[1], cache[5]
    return (
        scales.dtype == torch.float32
        and scales.device == state.device
        and scales.shape == state.shape[:3]
        and scales.stride()[1:] == (64, 1)
        and scales.stride(0) >= 128 * 64
    )


def _runtime_metadata(layer, metadata):
    if type(metadata) is not Mamba2AttentionMetadata:
        return None
    batches = metadata.num_decodes
    batched = batches != 1
    # At full configured capacity every bounded query must have the same length.
    # A smaller aggregate M cannot prove uniformity without reading GPU lengths.
    batch_geometry = (
        2 <= batches <= 4
        and metadata.num_decode_tokens == batches * (1 + layer.num_spec)
        and metadata.num_decode_tokens <= 32
    )
    if not (
        type(metadata) is Mamba2AttentionMetadata
        and metadata.num_prefills == 0
        and metadata.num_prefill_tokens == 0
        and (
            batch_geometry
            if batched
            else (1 <= metadata.num_decode_tokens <= 1 + layer.num_spec)
        )
        and layer.tp_size == 1
        and layer.num_heads == 128
        and layer.head_dim == 64
        and layer.ssm_state_size == 128
        and layer.n_groups == 8
        and 0 <= layer.num_spec <= 7
        and layer.activation in ("silu", "swish")
        and layer.use_replayssm
        and layer.use_flashinfer_replayssm
        and _supported_checkpoint_cache(layer.kv_cache)
    ):
        return None
    history = layer.kv_cache[0]
    history_len = 3 + layer.num_spec
    indices = _first_cache_indices(metadata.state_indices_tensor_d, batches)
    accepted = metadata.num_accepted_tokens
    if not (
        history.dtype == torch.bfloat16
        and history.ndim == 3
        and history.shape[0] > 1
        and history.shape[1:] == (history_len, 10240)
        and history.stride(0) >= history_len * 10240
        and history.stride(0) % 8 == 0
        and history.stride(1) == 10240
        and history.stride(2) == 1
        and indices is not None
        and indices.dtype == torch.int32
        and indices.numel() == batches
        and (
            1 <= indices.stride(0) <= 134217727 if batched else indices.is_contiguous()
        )
        and layer.conv_weights.dtype == torch.bfloat16
        and layer.conv_weights.shape == (10240, 4)
        and layer.conv_weights.is_contiguous()
        and layer.conv1d.bias is not None
        and layer.conv1d.bias.dtype == torch.bfloat16
        and layer.conv1d.bias.numel() == 10240
        and layer.conv1d.bias.is_contiguous()
    ):
        return None
    if accepted is None:
        # Public regular decode has no acceptance tensor and carries rows0..2.
        if layer.num_spec != 0:
            return None
    elif not (
        accepted.dtype == torch.int32
        and accepted.numel() == batches
        and accepted.is_contiguous()
    ):
        return None
    return history, indices, accepted


def project_and_run(layer, hidden_states, projected, output, metadata):
    """Run inside the new stable-buffer opaque boundary, after static admission."""
    tokens = hidden_states.shape[0]
    if not (
        projected.shape == (tokens, 18560)
        and output.shape == (tokens, 8192)
        and projected.dtype == output.dtype == hidden_states.dtype == torch.bfloat16
        and projected.device == output.device == hidden_states.device
        and projected.is_contiguous()
        and output.is_contiguous()
    ):
        raise ValueError("Mamba fused caller-buffer geometry mismatch")
    projected_start, output_start = projected.data_ptr(), output.data_ptr()
    if (
        projected_start < output_start + tokens * 8192 * 2
        and output_start < projected_start + tokens * 18560 * 2
    ):
        raise ValueError("Mamba projected and SSM outputs must be disjoint")
    linear = layer.in_proj
    admitted = _runtime_metadata(layer, metadata)
    if admitted is not None and (
        metadata.num_decode_tokens != tokens
        or (
            metadata.num_decodes > 1
            and not _batched_shape_is_profitable(
                metadata.num_decodes, 1 + layer.num_spec
            )
        )
    ):
        # Preserve the original path for padding and unqualified batch shapes.
        admitted = None
    if admitted is None:
        # This exact owner has no activation/format/adapter transform or padding.
        # The existing public output argument fills the caller buffer directly.
        ops.marlin_gemm(
            a=hidden_states,
            c=projected,
            b_q_weight=linear.weight,
            b_bias=None,
            b_scales=linear.weight_scale,
            global_scale=linear.weight_global_scale,
            b_zeros=None,
            workspace=get_marlin_workspace(hidden_states.device),
            b_q_type=scalar_types.float4_e2m1f,
            size_m=tokens,
            size_n=18560,
            size_k=4096,
            use_atomic_add=should_use_atomic_add_reduce(
                m=tokens,
                n=18560,
                k=4096,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
        )
        layer.conv_ssm_forward(projected, output)
        return
    history, indices, accepted = admitted
    ops.mamba_marlin_input_conv(
        hidden_states,
        projected,
        linear.weight,
        linear.weight_scale,
        linear.weight_global_scale,
        layer.conv_weights,
        layer.conv1d.bias,
        history,
        indices,
        accepted,
        PAD_SLOT_ID,
        NULL_BLOCK_ID,
    )
    layer.conv_ssm_forward(projected, output, _conv_precomputed=True)
