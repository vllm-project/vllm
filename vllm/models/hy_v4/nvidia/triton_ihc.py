# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton iHC pre/post kernels for HY V4.

Adapted from the SGLang HY V4 implementation:
https://github.com/sgl-project/sglang/pull/36805
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm import envs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_BLOCK_K = 1024
_BLOCK_D = 1024


def triton_ihc_supported(x: torch.Tensor) -> bool:
    """Return whether the in-tree Triton path can run for this input."""
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and not envs.VLLM_BATCH_INVARIANT
    )


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    x = tl.load(
        x_ptr + token_idx * K_TOTAL + k_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)

    weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0) * x[None, :],
        axis=1,
    )
    post_mix = tl.sum(
        tl.load(
            weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
            mask=weight_mask,
            other=0.0,
        )
        * x[None, :],
        axis=1,
    )

    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        post_mix += tl.load(
            partial + 1 + HC_POW2 + hc_idx,
            mask=hc_mask,
            other=0.0,
        )

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(
            post_ptr + token_idx * HC_MULT + hc_idx,
            post,
            mask=hc_mask,
        )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT
    post = tl.load(
        post_ptr + token_idx * HC_MULT + hc_idx,
        mask=hc_mask,
        other=0.0,
    )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    residual_row = residual_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    output_row = output_ptr + token_idx * HC_MULT * HIDDEN_SIZE
    for channel_idx in tl.static_range(HC_MULT):
        residual = tl.load(
            residual_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, post, 0.0), axis=0)
        output = channel_gate * x + residual
        tl.store(
            output_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


def _get_ihc_warmup_config(
    vllm_config: Any,
) -> tuple[torch.dtype, int, int, float, float, float] | None:
    dtype = vllm_config.model_config.dtype
    if (
        not HAS_TRITON
        or not current_platform.is_cuda()
        or dtype not in (torch.float16, torch.bfloat16)
        or envs.VLLM_BATCH_INVARIANT
    ):
        return None

    config = vllm_config.model_config.hf_config
    if not getattr(config, "enable_ihc", False):
        return None
    return (
        dtype,
        int(config.hidden_size),
        int(config.hc_mult),
        float(config.hc_magnitude),
        float(config.hc_eps),
        float(config.rms_norm_eps),
    )


class IHCPreStage1Kernel(
    VllmTritonJitKernel["IHCPreStage1Kernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        k_total: int
        hc_mult: int
        hc_pow2: int
        num_splits: int
        block_k: int
        partial_stride: int

    kernel = staticmethod(_ihc_pre_stage1)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        hidden_size: int,
        hc_mult: int,
    ) -> CompileKey:
        k_total = hc_mult * hidden_size
        hc_pow2 = triton.next_power_of_2(hc_mult)
        num_splits = triton.cdiv(k_total, _BLOCK_K)
        partial_stride = 1 + 2 * hc_pow2
        return self.CompileKey(
            dtype=dtype,
            k_total=k_total,
            hc_mult=hc_mult,
            hc_pow2=hc_pow2,
            num_splits=num_splits,
            block_k=_BLOCK_K,
            partial_stride=partial_stride,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        warmup_config = _get_ihc_warmup_config(vllm_config)
        if warmup_config is None:
            return []
        dtype, hidden_size, hc_mult, _, _, _ = warmup_config
        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        hidden_size = compile_key.k_total // compile_key.hc_mult
        return dict(
            x=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hc_mult, hidden_size),
            ),
            weight=TritonWarmupTensor(
                torch.float32,
                shape=(2 * compile_key.hc_mult, compile_key.k_total),
            ),
            _partial=TritonWarmupTensor(
                torch.float32,
                shape=(1, compile_key.num_splits, compile_key.partial_stride),
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        _partial: Any | None = None,
    ) -> LaunchSpec:
        num_tokens, hc_mult, hidden_size = x.shape
        k_total = hc_mult * hidden_size
        hc_pow2 = triton.next_power_of_2(hc_mult)
        num_splits = triton.cdiv(k_total, _BLOCK_K)
        partial_stride = 1 + 2 * hc_pow2
        if _partial is None:
            _partial = torch.empty(
                (num_tokens, num_splits, partial_stride),
                dtype=torch.float32,
                device=x.device,
            )
        return (
            (num_tokens, num_splits),
            dict(
                partial_ptr=_partial,
                K_TOTAL=k_total,
                HC_MULT=hc_mult,
                HC_POW2=hc_pow2,
                NUM_SPLITS=num_splits,
                BLOCK_K=_BLOCK_K,
                PARTIAL_STRIDE=partial_stride,
                num_warps=8,
                enable_fp_fusion=False,
            ),
            _partial,
        )


class IHCPreStage2Kernel(
    VllmTritonJitKernel["IHCPreStage2Kernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        hidden_size: int
        k_total: int
        hc_mult: int
        hc_pow2: int
        num_splits: int
        partial_stride: int
        block_d: int
        magnitude: float
        norm_eps: float
        hc_eps: float

    kernel = staticmethod(_ihc_pre_stage2)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        hidden_size: int,
        hc_mult: int,
        magnitude: float,
        norm_eps: float,
        hc_eps: float,
    ) -> CompileKey:
        k_total = hc_mult * hidden_size
        hc_pow2 = triton.next_power_of_2(hc_mult)
        num_splits = triton.cdiv(k_total, _BLOCK_K)
        partial_stride = 1 + 2 * hc_pow2
        return self.CompileKey(
            dtype=dtype,
            hidden_size=hidden_size,
            k_total=k_total,
            hc_mult=hc_mult,
            hc_pow2=hc_pow2,
            num_splits=num_splits,
            partial_stride=partial_stride,
            block_d=_BLOCK_D,
            magnitude=magnitude,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        warmup_config = _get_ihc_warmup_config(vllm_config)
        if warmup_config is None:
            return []
        dtype, hidden_size, hc_mult, magnitude, hc_eps, norm_eps = warmup_config
        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            magnitude=magnitude,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            x=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hc_mult, compile_key.hidden_size),
            ),
            partial=TritonWarmupTensor(
                torch.float32,
                shape=(1, compile_key.num_splits, compile_key.partial_stride),
            ),
            scale=TritonWarmupTensor(torch.float32, shape=(2,)),
            base=TritonWarmupTensor(
                torch.float32,
                shape=(2 * compile_key.hc_mult,),
            ),
            magnitude=compile_key.magnitude,
            hc_eps=compile_key.hc_eps,
            norm_eps=compile_key.norm_eps,
            _outputs=(
                TritonWarmupTensor(
                    compile_key.dtype,
                    shape=(1, compile_key.hidden_size),
                ),
                TritonWarmupTensor(
                    torch.float32,
                    shape=(1, compile_key.hc_mult),
                ),
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        x: torch.Tensor,
        partial: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        magnitude: float,
        hc_eps: float,
        norm_eps: float,
        *,
        _outputs: tuple[Any, Any] | None = None,
    ) -> LaunchSpec:
        num_tokens, hc_mult, hidden_size = x.shape
        k_total = hc_mult * hidden_size
        hc_pow2 = triton.next_power_of_2(hc_mult)
        num_splits = triton.cdiv(k_total, _BLOCK_K)
        partial_stride = 1 + 2 * hc_pow2
        if _outputs is None:
            _outputs = (
                torch.empty(
                    (num_tokens, hidden_size), dtype=x.dtype, device=x.device
                ),
                torch.empty(
                    (num_tokens, hc_mult), dtype=torch.float32, device=x.device
                ),
            )
        output, post = _outputs
        return (
            (num_tokens, triton.cdiv(hidden_size, _BLOCK_D)),
            dict(
                output_ptr=output,
                post_ptr=post,
                HIDDEN_SIZE=hidden_size,
                K_TOTAL=k_total,
                HC_MULT=hc_mult,
                HC_POW2=hc_pow2,
                NUM_SPLITS=num_splits,
                PARTIAL_STRIDE=partial_stride,
                BLOCK_D=_BLOCK_D,
                MAGNITUDE=magnitude,
                NORM_EPS=norm_eps,
                HC_EPS=hc_eps,
                num_warps=4,
                enable_fp_fusion=False,
            ),
            _outputs,
        )


class IHCPostKernel(VllmTritonJitKernel["IHCPostKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        hidden_size: int
        hc_mult: int
        hc_pow2: int
        block_d: int

    kernel = staticmethod(_ihc_post_kernel)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        hidden_size: int,
        hc_mult: int,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            hc_pow2=triton.next_power_of_2(hc_mult),
            block_d=_BLOCK_D,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        warmup_config = _get_ihc_warmup_config(vllm_config)
        if warmup_config is None:
            return []
        dtype, hidden_size, hc_mult, _, _, _ = warmup_config
        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            x=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hidden_size),
            ),
            residual=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hc_mult, compile_key.hidden_size),
            ),
            post=TritonWarmupTensor(
                torch.float32,
                shape=(1, compile_key.hc_mult),
            ),
            _output=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hc_mult, compile_key.hidden_size),
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        *,
        _output: Any | None = None,
    ) -> LaunchSpec:
        num_tokens, hidden_size = x.shape
        hc_mult = post.shape[-1]
        if _output is None:
            _output = torch.empty(
                (num_tokens, hc_mult, hidden_size), dtype=x.dtype, device=x.device
            )
        return (
            (num_tokens, triton.cdiv(hidden_size, _BLOCK_D)),
            dict(
                output_ptr=_output,
                HIDDEN_SIZE=hidden_size,
                HC_MULT=hc_mult,
                HC_POW2=triton.next_power_of_2(hc_mult),
                BLOCK_D=_BLOCK_D,
                num_warps=4,
                enable_fp_fusion=False,
            ),
            _output,
        )


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels and produce the post gates."""
    assert x.dim() == 3
    assert x.is_cuda and weight.is_cuda and scale.is_cuda and base.is_cuda
    assert weight.dtype == torch.float32
    assert scale.dtype == torch.float32 and base.dtype == torch.float32

    x = x.contiguous()
    assert weight.is_contiguous()
    scale = scale.contiguous()
    base = base.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,)
    assert base.shape == (2 * hc_mult,)

    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
        )

    partial = _IHC_PRE_STAGE1_KERNEL(
        x,
        weight,
    )
    return _IHC_PRE_STAGE2_KERNEL(
        x,
        partial,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
    )


def triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Scatter a sub-block output back over the iHC residual channels."""
    assert x.dim() == 2
    assert x.is_cuda and residual.is_cuda and post.is_cuda
    assert post.dtype == torch.float32

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    if num_tokens == 0:
        return torch.empty((0, hc_mult, hidden_size), dtype=x.dtype, device=x.device)

    return _IHC_POST_KERNEL(
        x,
        residual,
        post,
    )


_IHC_PRE_STAGE1_KERNEL = IHCPreStage1Kernel()
_IHC_PRE_STAGE2_KERNEL = IHCPreStage2Kernel()
_IHC_POST_KERNEL = IHCPostKernel()
