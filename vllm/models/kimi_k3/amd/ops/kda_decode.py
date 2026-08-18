# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry points for the fused Kimi-K3 KDA decode kernel.

The kernel in ``csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu``
replaces, for a pure non-speculative decode batch, the three Triton launches
and two copies the AMD KDA layer otherwise runs per layer: the packed causal
conv1d update, the recurrent delta-rule step, and the gated output RMSNorm.

The kernel wants a width-major conv weight and an fp32 norm weight, so both are
staged once at load time by the weight loaders below.
"""

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

# Head counts the kernel is instantiated for (Kimi-K3 has 96 KDA heads, so this
# covers TP1/2/4/8).
SUPPORTED_NUM_HEADS = (12, 24, 48, 96)


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    """Whether the fused decode kernel can serve this layer on this device."""
    from vllm.platforms.rocm import on_gfx950

    if (
        num_heads not in SUPPORTED_NUM_HEADS
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
        or is_conv_state_dim_first()
        or not hasattr(torch.ops._C, "fused_kda_decode")
    ):
        return False
    # TODO: Verify on other archs; only measured on gfx950 for now
    return on_gfx950()


def make_decode_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
    decode_conv1d_weight: torch.Tensor | None,
) -> Callable[..., None]:
    """Load the packed conv1d weight, mirroring a width-major fp32 copy.

    The fused kernel indexes the weight as ``[qkv, width, channel]`` so the
    channel dimension is the contiguous one; the Triton prefill and fallback
    decode kernels keep the ``[channel, width]`` layout.
    """
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        param.data[target_start : target_start + shard_size].copy_(loaded_shard)
        if decode_conv1d_weight is not None and not param.is_meta:
            decode_conv1d_weight[loaded_shard_id].copy_(
                loaded_shard.squeeze(1).transpose(0, 1)
            )

    return weight_loader


def make_decode_norm_weight_loader(
    decode_norm_weight: torch.Tensor,
) -> Callable[..., None]:
    """Load the gated-norm weight, mirroring an fp32 copy for the kernel."""

    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        if not param.is_meta:
            decode_norm_weight.copy_(param.data)

    return weight_loader
