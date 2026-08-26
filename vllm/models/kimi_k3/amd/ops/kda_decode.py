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
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

# Head counts the kernel is instantiated for (Kimi-K3 has 96 KDA heads, so this
# covers TP1/2/4/8).
SUPPORTED_NUM_HEADS = (12, 24, 48, 96)

_MXFP4_GROUP = 32

def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    """Whether the fused decode kernel can serve this layer on this device."""
    from vllm.platforms.rocm import on_gfx942, on_gfx950

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
    # gfx950 (MI355X) and gfx942 (MI325X): both CDNA, sharing the wave64 / DPP /
    # bf16 primitives the kernel relies on.
    return on_gfx950() or on_gfx942()


def mxfp4_layout_for_oproj(o_proj: torch.nn.Module) -> str | None:
    """Return `plain` / `shuffled` when `o_proj` can consume that ABI.

    Requires `input_quant_key is kMxfp4Dynamic`. `input_quant_layout` is `shuffled` and selects ASM.
    """
    if getattr(o_proj, "input_quant_key", None) != kMxfp4Dynamic:
        return None
    if getattr(o_proj, "input_quant_layout", None) == "shuffled":
        return "shuffled"
    return "plain"


def alloc_kda_mxfp4(
    num_tokens: int,
    hidden: int,
    layout: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the MXFP4 pair the fused decode kernel writes.
    """
    assert hidden % _MXFP4_GROUP == 0
    n_groups = hidden // _MXFP4_GROUP
    if layout == "plain":
        data = torch.zeros((num_tokens, hidden // 2), dtype=torch.uint8, device=device)
        scale = torch.zeros((n_groups, num_tokens), dtype=torch.uint8, device=device).T
        return data, scale
    if layout == "shuffled":
        pad_m32 = (num_tokens + 31) // 32 * 32
        pad_m256 = (num_tokens + 255) // 256 * 256
        pad_n8 = (n_groups + 7) // 8 * 8
        data = torch.zeros((pad_m32, hidden // 2), dtype=torch.uint8, device=device)
        scale = torch.zeros((pad_m256, pad_n8), dtype=torch.uint8, device=device)
        return data, scale
    raise ValueError(f"Unknown MXFP4 layout: {layout!r}")


def wrap_kda_mxfp4(
    data: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: torch.Size,
    orig_dtype: torch.dtype,
    o_proj: torch.nn.Module,
) -> QuantizedActivation:
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=orig_dtype,
        orig_shape=orig_shape,
        quant_key=o_proj.input_quant_key,
        layout=getattr(o_proj, "input_quant_layout", None)
    )


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
