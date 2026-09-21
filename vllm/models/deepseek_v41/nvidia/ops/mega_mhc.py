# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import Any

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8DynamicDeepGemm,
)
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import MegaMoeFp8Target
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _import_deep_gemm,
    get_tma_aligned_size,
    is_deep_gemm_supported,
    mega_mhc,
)

# Hidden elements per packed int32 scale word: 4 UE8M0 scales x 32-element groups.
_HIDDEN_PER_SF_WORD = 128


def make_deep_gemm_packed_scale(
    num_tokens: int, hidden: int, device: torch.device
) -> torch.Tensor:
    """Allocate a [T, H/128] int32 scale in DeepGEMM's MN-major TMA-aligned layout."""
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    return torch.empty_strided(
        (num_tokens, hidden // _HIDDEN_PER_SF_WORD),
        (1, aligned),
        dtype=torch.int32,
        device=device,
    )


@functools.cache
def is_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
    """Return whether DeepGEMM can run the DSV4.1 shifted mHC kernel."""
    if not (
        is_deep_gemm_supported()
        and current_platform.is_device_capability_family(100)
        and hidden_size > 0
        and hidden_size % 1024 == 0
        and hc_mult == 4
    ):
        return False
    deep_gemm = _import_deep_gemm()
    return deep_gemm is not None and callable(getattr(deep_gemm, "mega_mhc", None))


def mhc_shifted_post_pre_deep_gemm(
    x: torch.Tensor,
    residual: torch.Tensor,
    shifted_prev_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    fp8_out: str | None = None,
    moe_target: MegaMoeFp8Target | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    QuantizedActivation | None,
]:
    """Run DSV4.1 shifted post, next pre, and BF16 RMSNorm with Mega mHC.

    ``fp8_out`` additionally emits the normalized output as MXFP8 straight from
    the kernel (the BF16 output is always produced for the BF16 consumers):

    * ``"gemm"``: returns a ``QuantizedActivation`` with DeepGEMM packed scales
      for a ``fp8_gemm_nt`` consumer (the attention ``fused_wqa_wkv``).
    * ``"moe"``: writes the FP8 tokens and both scale layouts into
      ``moe_target`` (the Mega MoE symmetric buffer), so the MoE input staging
      no longer quantizes; returns ``None``.
    """
    num_tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    new_residual = torch.empty_like(residual)
    new_prev_mix = x.new_empty(num_tokens, hc_mult, 1, dtype=torch.float32)
    new_post_mix = torch.empty_like(post_mix)
    new_comb_res_mix = torch.empty_like(comb_res_mix)
    y_bf16 = x.new_empty(num_tokens, hidden_size, dtype=torch.bfloat16)
    fp8_kwargs: dict[str, Any] = {}
    y_quant: QuantizedActivation | None = None
    if fp8_out == "gemm":
        y_fp8 = x.new_empty(num_tokens, hidden_size, dtype=torch.float8_e4m3fn)
        y_gemm_sf = make_deep_gemm_packed_scale(num_tokens, hidden_size, x.device)
        fp8_kwargs = {"y_fp8": y_fp8, "y_gemm_sf": y_gemm_sf}
        y_quant = QuantizedActivation(
            y_fp8, y_gemm_sf, torch.bfloat16, y_fp8.shape, kMxfp8DynamicDeepGemm
        )
    elif fp8_out == "moe":
        assert moe_target is not None
        y_fp8 = moe_target.x[:num_tokens]
        assert y_fp8.is_contiguous() and y_fp8.shape == (num_tokens, hidden_size)
        fp8_kwargs = {
            "y_fp8": y_fp8,
            "y_routed_sf": moe_target.x_sf[:num_tokens],
            # DeepGEMM checks every SF as [num_tokens, H/128]; the strides and
            # storage of the full shared-expert view stay those of the buffer.
            "y_shared_sf": moe_target.shared_sf[:num_tokens],
            "shared_sf_block_m": moe_target.shared_block_m,
        }
    elif fp8_out is not None:
        raise ValueError(f"unknown fp8_out={fp8_out!r}")
    mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=shifted_prev_mix.unsqueeze(-1),
        post_mix=post_mix,
        comb_res_mix=comb_res_mix,
        fn=fn,
        mix_scales=mix_scales,
        mix_bases=mix_bases,
        hc_mult=hc_mult,
        hc_norm_eps=hc_norm_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_scale=hc_post_scale,
        sinkhorn_eps=sinkhorn_eps,
        num_sinkhorn_iters=num_sinkhorn_iters,
        rmsnorm_weight=rmsnorm_weight,
        rmsnorm_eps=rmsnorm_eps,
        rmsnorm_scale=1.0,
        new_residual=new_residual,
        new_prev_mix=new_prev_mix,
        new_post_mix=new_post_mix,
        new_comb_res_mix=new_comb_res_mix,
        y_bf16=y_bf16,
        **fp8_kwargs,
    )
    return (
        new_residual,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        new_prev_mix.squeeze(-1),
        y_quant,
    )
