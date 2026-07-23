# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2




class FusedQKVRMSNormKernel(VllmJitKernel["FusedQKVRMSNormKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        Q_SIZE: int
        KV_SIZE: int
        BLOCK_SIZE: int
        q_in_stride: int
        q_out_stride: int
        kv_in_stride: int
        kv_out_stride: int
        eps: float

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
    ):
        # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
        # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
        # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
        # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
        # past num_tokens ~87K under large chunked prefill.
        token_idx = tl.program_id(0).to(tl.int64)
        pid_task = tl.program_id(1)

        if pid_task == 0:
            SIZE = Q_SIZE
            row_in = q_ptr + token_idx * q_in_stride
            weight_ptr = q_weight_ptr
            row_out = q_out_ptr + token_idx * q_out_stride
        else:
            SIZE = KV_SIZE
            row_in = kv_ptr + token_idx * kv_in_stride
            weight_ptr = kv_weight_ptr
            row_out = kv_out_ptr + token_idx * kv_out_stride

        # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
        # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
        # keep x, rrms, and w all in fp32 and perform a single cast at store.
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < SIZE
        x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / SIZE
        rrms = tl.rsqrt(variance + eps)
        w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        q_size: int,
        kv_size: int,
        q_in_stride: int,
        q_out_stride: int,
        kv_in_stride: int,
        kv_out_stride: int,
        eps: float,
    ) -> CompileKey:
        max_size = q_size if q_size >= kv_size else kv_size
        return self.CompileKey(
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            BLOCK_SIZE=next_power_of_2(max_size),
            q_in_stride=q_in_stride,
            q_out_stride=q_out_stride,
            kv_in_stride=kv_in_stride,
            kv_out_stride=kv_out_stride,
            eps=eps,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        q_size = int(getattr(hf_config, "q_lora_rank", 0) or 0)
        kv_size = int(getattr(hf_config, "head_dim", 0) or 0)
        if q_size <= 0 or kv_size <= 0:
            return []

        input_stride = q_size + kv_size
        return self._trace_dispatch(self.dispatch)(
            q_size=q_size,
            kv_size=kv_size,
            q_in_stride=input_stride,
            q_out_stride=(input_stride, q_size),
            kv_in_stride=input_stride,
            kv_out_stride=(input_stride, kv_size),
            eps=float(getattr(hf_config, "rms_norm_eps")),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        bf16_ptr = TritonWarmupTensor(torch.bfloat16)
        fp32_ptr = TritonWarmupTensor(torch.float32)
        warmup(
            bf16_ptr,
            bf16_ptr,
            fp32_ptr,
            compile_key.q_in_stride,
            compile_key.q_out_stride,
            bf16_ptr,
            bf16_ptr,
            fp32_ptr,
            compile_key.kv_in_stride,
            compile_key.kv_out_stride,
            compile_key.eps,
            Q_SIZE=compile_key.Q_SIZE,
            KV_SIZE=compile_key.KV_SIZE,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            grid=(1, 2),
        )

    def __call__(
        self,
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

        q_size = qr.shape[1]
        kv_size = kv.shape[1]
        num_tokens = qr.shape[0]
        qr_out = torch.empty_like(qr)
        kv_out = torch.empty_like(kv)
        if num_tokens == 0:
            return qr_out, kv_out

        compile_key = self.dispatch(
            q_size=q_size,
            kv_size=kv_size,
            q_in_stride=qr.stride(0),
            q_out_stride=qr_out.stride(0),
            kv_in_stride=kv.stride(0),
            kv_out_stride=kv_out.stride(0),
            eps=eps,
        )
        self.kernel[(num_tokens, 2)](
            qr,
            qr_out,
            q_weight,
            compile_key.q_in_stride,
            compile_key.q_out_stride,
            kv,
            kv_out,
            kv_weight,
            compile_key.kv_in_stride,
            compile_key.kv_out_stride,
            compile_key.eps,
            Q_SIZE=compile_key.Q_SIZE,
            KV_SIZE=compile_key.KV_SIZE,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
        )
        return qr_out, kv_out


_FUSED_Q_KV_RMSNORM_KERNEL = FusedQKVRMSNormKernel()


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FUSED_Q_KV_RMSNORM_KERNEL(qr, kv, q_weight, kv_weight, eps)
