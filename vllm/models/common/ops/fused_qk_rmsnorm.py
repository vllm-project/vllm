# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit(do_not_specialize=["num_tokens"])
def _kv_rmsnorm_512_kernel(
    kv, out, weight, num_tokens, stride, eps, LAUNCH_PDL: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, 512)
    w = tl.load(weight + cols).to(tl.float32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(
        kv + rows[:, None] * stride + cols[None, :],
        rows[:, None] < num_tokens,
        0,
    ).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, 1) / 512 + eps)
    y = x * rrms[:, None] * w[None, :]
    tl.store(out + rows[:, None] * 512 + cols[None, :], y, rows[:, None] < num_tokens)


def _kv_rmsnorm_512_warmup_inputs(
    dtype: torch.dtype, eps: float
) -> list[dict[str, Any]]:
    return [
        dict(
            kv=TritonWarmupTensor(
                dtype, aligned=aligned, shape=(4, 512), strides=(stride, 1)
            ),
            out=TritonWarmupTensor(dtype, shape=(4, 512)),
            weight=TritonWarmupTensor(dtype, shape=(512,)),
            eps=eps,
        )
        for stride in (512, 513, 1)
        for aligned in (True, False)
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_kv_rmsnorm_512_kernel, warmup_inputs=_kv_rmsnorm_512_warmup_inputs
)
def _kv_rmsnorm_512(kv, out, weight, eps):
    launch_pdl = current_platform.is_arch_support_pdl()
    return (triton.cdiv(kv.shape[0], 4),), dict(
        num_tokens=kv.shape[0],
        stride=kv.stride(0),
        LAUNCH_PDL=launch_pdl,
        num_warps=4,
        launch_pdl=launch_pdl,
    )


class FusedQKVRMSNormKernel(VllmTritonJitKernel["FusedQKVRMSNormKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        q_size: int
        kv_size: int
        block_size: int
        q_in_stride: int
        q_out_stride: int
        kv_in_stride: int
        kv_out_stride: int
        eps: float
        launch_pdl: bool
        kv_only: bool = False

    @staticmethod
    @triton.jit
    def kernel(
        q_ptr,
        q_out_ptr,
        q_weight_ptr,
        q_in_stride,
        q_out_stride,
        kv_ptr,
        kv_out_ptr,
        kv_weight_ptr,
        kv_in_stride,
        kv_out_stride,
        eps,
        Q_SIZE: tl.constexpr,
        KV_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        launch_pdl: tl.constexpr,
        KV_ONLY: tl.constexpr = False,
    ):
        # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
        # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
        # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
        # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
        # past num_tokens ~87K under large chunked prefill.
        token_idx = tl.program_id(0).to(tl.int64)
        pid_task = tl.program_id(1)

        if not KV_ONLY and pid_task == 0:
            SIZE = Q_SIZE
            row_in = q_ptr + token_idx * q_in_stride
            weight_ptr = q_weight_ptr
            row_out = q_out_ptr + token_idx * q_out_stride
        else:
            SIZE = KV_SIZE
            row_in = kv_ptr + token_idx * kv_in_stride
            weight_ptr = kv_weight_ptr
            row_out = kv_out_ptr + token_idx * kv_out_stride

        # RMSNorm in fp32 throughout — matches
        # csrc/layernorm_kernels.cu's
        # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
        # keep x, rrms, and w all in fp32 and perform a single cast at store.
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < SIZE
        # The weight load does not depend on the producer's output, so issue it
        # before the PDL wait: gamma streams in while the producer finishes,
        # leaving a single dependent global round trip (x) after the wait.
        w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)

        if launch_pdl:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()

        x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / SIZE
        rrms = tl.rsqrt(variance + eps)
        y = x * rrms * w
        tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        q_size: int,
        kv_size: int,
        eps: float,
        launch_pdl: bool,
        kv_only: bool = False,
        **compile_key_fields: int,
    ) -> CompileKey:
        max_size = q_size if q_size >= kv_size else kv_size
        return self.CompileKey(
            **compile_key_fields,
            dtype=dtype,
            q_size=q_size,
            kv_size=kv_size,
            block_size=next_power_of_2(max_size),
            eps=eps,
            launch_pdl=launch_pdl,
            kv_only=kv_only,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        q_size = int(getattr(hf_config, "q_lora_rank", 0) or 0)
        kv_size = int(getattr(hf_config, "head_dim", 0) or 0)
        if q_size <= 0 or kv_size <= 0:
            return []

        input_stride = q_size + kv_size
        keys = self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            q_size=q_size,
            kv_size=kv_size,
            q_in_stride=input_stride,
            q_out_stride=(input_stride, q_size),
            kv_in_stride=input_stride,
            kv_out_stride=(input_stride, kv_size),
            eps=float(hf_config.rms_norm_eps),
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
        keys += self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            q_size=q_size,
            kv_size=kv_size,
            q_in_stride=0,
            q_out_stride=0,
            kv_in_stride=(input_stride, kv_size),
            kv_out_stride=kv_size,
            eps=float(hf_config.rms_norm_eps),
            launch_pdl=current_platform.is_arch_support_pdl(),
            kv_only=True,
        )
        return keys

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        inputs: dict[str, Any] = dict(
            q=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.q_size),
                strides=(compile_key.q_in_stride, 1),
            ),
            kv=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.kv_size),
                strides=(compile_key.kv_in_stride, 1),
            ),
            q_out=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.q_size),
                strides=(compile_key.q_out_stride, 1),
            ),
            kv_out=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.kv_size),
                strides=(compile_key.kv_out_stride, 1),
            ),
            q_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.q_size,),
            ),
            kv_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.kv_size,),
            ),
            eps=compile_key.eps,
            q_size=compile_key.q_size,
        )
        if compile_key.kv_only:
            inputs.update(q=None, q_out=None, q_weight=None)
        return inputs

    @kernel_launcher
    def __call__(
        self,
        q: torch.Tensor | None,
        kv: torch.Tensor,
        q_out: torch.Tensor | None,
        kv_out: torch.Tensor,
        q_weight: torch.Tensor | None,
        kv_weight: torch.Tensor,
        eps: float,
        q_size: int = 0,
    ) -> LaunchSpec:
        kv_only = q is None
        q_size = q_size if q is None else q.shape[1]
        kv_size = kv.shape[1]
        block_size = next_power_of_2(max(q_size, kv_size))
        return (kv.shape[0], 1 if kv_only else 2), dict(
            q_in_stride=0 if q is None else q.stride(0),
            q_out_stride=0 if q_out is None else q_out.stride(0),
            kv_in_stride=kv.stride(0),
            kv_out_stride=kv_out.stride(0),
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            BLOCK_SIZE=block_size,
            launch_pdl=current_platform.is_arch_support_pdl(),
            KV_ONLY=kv_only,
            # One vectorized 128-bit access per thread for K3/DSv4-sized rows,
            # instead of a serial per-thread chain at the default 4 warps.
            num_warps=8 if block_size >= 2048 else 4,
        )


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    qr_out = torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    if qr.shape[0] > 0:
        _FUSED_Q_KV_RMSNORM_KERNEL(
            qr,
            kv,
            qr_out,
            kv_out,
            q_weight,
            kv_weight,
            eps,
        )
    return qr_out, kv_out


def fused_kv_rmsnorm(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    *,
    q_size: int,
) -> torch.Tensor:
    """Run only KV normalization, with a tiled CUDA path for 512 columns.

    Args:
        kv: KV projection with contiguous columns.
        kv_weight: RMSNorm weight.
        eps: RMSNorm epsilon.
        q_size: Original Q width for the generic fused-kernel fallback.

    """
    assert kv.ndim == 2 and kv.stride(-1) == 1
    assert kv_weight.is_contiguous()
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    if kv.shape[1] == 512 and current_platform.is_cuda():
        if kv.shape[0] > 0:
            _kv_rmsnorm_512(kv, kv_out, kv_weight, eps)
        return kv_out
    if kv.shape[0] > 0:
        _FUSED_Q_KV_RMSNORM_KERNEL(
            None, kv, None, kv_out, None, kv_weight, eps, q_size=q_size
        )
    return kv_out


_FUSED_Q_KV_RMSNORM_KERNEL = FusedQKVRMSNormKernel()
