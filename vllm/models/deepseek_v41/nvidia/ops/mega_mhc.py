# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import NamedTuple

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import MegaMoEInputs
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8DynamicPacked,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _import_deep_gemm,
    get_tma_aligned_size,
    is_deep_gemm_supported,
    mega_mhc,
)

# Hidden elements per packed int32 scale: four UE8M0 scales of 32 elements.
_HIDDEN_PER_PACKED_SF = 128


class MegaMhcFp8Out(NamedTuple):
    """Where Mega mHC also writes its RMSNorm output as MXFP8.

    The scales go to exactly one consumer layout: ``gemm_sf`` for an
    ``fp8_gemm_nt`` A operand, or ``routed_sf`` plus ``shared_sf`` for the
    MegaMoE symmetric buffer.
    """

    y: torch.Tensor
    """``[num_tokens, hidden]`` E4M3 values."""
    gemm_sf: torch.Tensor | None = None
    """Packed UE8M0, MN-major with a TMA-aligned token stride."""
    routed_sf: torch.Tensor | None = None
    """Packed UE8M0, K-major: MegaMoE ``x_sf``."""
    shared_sf: torch.Tensor | None = None
    """The fused shared expert's MN-major view of the whole buffer."""
    shared_block_m: int = 0

    @classmethod
    def for_gemm(
        cls, num_tokens: int, hidden_size: int, device: torch.device
    ) -> "MegaMhcFp8Out":
        aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
        gemm_sf = torch.empty_strided(
            (num_tokens, hidden_size // _HIDDEN_PER_PACKED_SF),
            (1, aligned),
            dtype=torch.int32,
            device=device,
        )
        y = torch.empty(
            num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=device
        )
        return cls(y, gemm_sf=gemm_sf)

    @classmethod
    def for_mega_moe(cls, inputs: MegaMoEInputs) -> "MegaMhcFp8Out":
        assert inputs.x_sf is not None, "MegaMoE format does not take MXFP8"
        return cls(
            inputs.x,
            routed_sf=inputs.x_sf,
            shared_sf=inputs.shared_x_sf,
            shared_block_m=inputs.shared_block_m or 0,
        )

    def as_gemm_input(self) -> QuantizedActivation:
        assert self.gemm_sf is not None
        return QuantizedActivation(
            self.y, self.gemm_sf, torch.bfloat16, self.y.shape, kMxfp8DynamicPacked
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


def can_use_mega_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    capture_aux: bool,
) -> bool:
    return (
        pre_mix is not None
        and norm_weight is not None
        and not capture_aux
        and x.shape[0] <= 1 << 20
        and is_mega_mhc_supported(x.shape[1], residual.shape[1])
    )


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
    fp8_out: MegaMhcFp8Out | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run DSV4.1 shifted post, next pre, and BF16 RMSNorm with Mega mHC.

    The BF16 output is always produced; ``fp8_out`` also gets it as MXFP8.
    """
    num_tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    new_residual = torch.empty_like(residual)
    new_prev_mix = x.new_empty(num_tokens, hc_mult, 1, dtype=torch.float32)
    new_post_mix = torch.empty_like(post_mix)
    new_comb_res_mix = torch.empty_like(comb_res_mix)
    y_bf16 = x.new_empty(num_tokens, hidden_size, dtype=torch.bfloat16)
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
        **_fp8_kwargs(fp8_out, num_tokens),
    )
    return (
        new_residual,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        new_prev_mix.squeeze(-1),
    )


def _fp8_kwargs(fp8_out: MegaMhcFp8Out | None, num_tokens: int) -> dict:
    if fp8_out is None:
        return {}
    kwargs = {"y_fp8": fp8_out.y[:num_tokens], "y_gemm_sf": fp8_out.gemm_sf}
    if fp8_out.routed_sf is not None:
        kwargs["y_routed_sf"] = fp8_out.routed_sf[:num_tokens]
        if fp8_out.shared_sf is not None:
            # DeepGEMM checks the shape as [num_tokens, hidden / 128]; the
            # rows it writes are permuted across the whole buffer.
            kwargs["y_shared_sf"] = fp8_out.shared_sf[:num_tokens]
            kwargs["shared_sf_block_m"] = fp8_out.shared_block_m
    return kwargs
