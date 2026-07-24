# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MTP-input RMSNorm: enorm (with mask-zero at position 0) + hnorm.

Replaces the eager sequence at the top of the MTP draft forward:
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = previous_hidden_states.view(-1, hc_mult, H)
    previous_hidden_states = self.hnorm(previous_hidden_states)

which lowers to ~6 small kernels (CompareEq, where, Fill, enorm rms_norm,
hnorm rms_norm, plus aten elementwise helpers) on the breakable-cudagraph
path. Math is preserved: positions==0 → masked row → zero RMS output
regardless of weight.

A single grid (T, hc_mult+1) drives both norms: task 0 is enorm on
inputs_embeds[token, :], task k+1 is hnorm on previous_hidden_states[token, k, :].
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit
def _rmsnorm_row(
    x,
    w_ptr,
    out_row_ptr,
    block,
    mask,
    eps,
    HIDDEN: tl.constexpr,
):
    x = x.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HIDDEN
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(w_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(out_row_ptr + block, y.to(out_row_ptr.dtype.element_ty), mask=mask)






class FusedMTPInputRMSNormKernel(
    VllmJitKernel["FusedMTPInputRMSNormKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        HIDDEN: int
        HC_MULT: int
        BLOCK_SIZE: int
        eps: float

    @staticmethod
    @triton.jit
    def kernel(
        inputs_embeds_ptr,
        positions_ptr,
        prev_hidden_ptr,
        enorm_weight_ptr,
        hnorm_weight_ptr,
        enorm_out_ptr,
        hnorm_out_ptr,
        eps,
        HIDDEN: tl.constexpr,
        HC_MULT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # int64 token index so per-token offsets don't overflow int32 at
        # large num_tokens (matches the convention in fused_q_kv_rmsnorm).
        token_idx = tl.program_id(0).to(tl.int64)
        pid_task = tl.program_id(1)

        block = tl.arange(0, BLOCK_SIZE)
        mask = block < HIDDEN

        if pid_task == 0:
            # enorm path: load inputs_embeds[token, :] then zero-mask at pos==0.
            # Math is preserved: pos==0 → x=0 → variance=0 → RMSNorm output is 0
            # regardless of weight, matching torch.where(pos==0, 0, x) + RMSNorm.
            pos = tl.load(positions_ptr + token_idx)
            keep = pos != 0
            x = tl.load(
                inputs_embeds_ptr + token_idx * HIDDEN + block, mask=mask, other=0.0
            )
            x = tl.where(keep, x, 0.0)
            _rmsnorm_row(
                x,
                enorm_weight_ptr,
                enorm_out_ptr + token_idx * HIDDEN,
                block,
                mask,
                eps,
                HIDDEN,
            )
        else:
            # hnorm path: load prev_hidden[token, slot, :].
            slot = pid_task - 1
            row_offset = (token_idx * HC_MULT + slot) * HIDDEN
            x = tl.load(prev_hidden_ptr + row_offset + block, mask=mask, other=0.0)
            _rmsnorm_row(
                x,
                hnorm_weight_ptr,
                hnorm_out_ptr + row_offset,
                block,
                mask,
                eps,
                HIDDEN,
            )

    def dispatch(  # type: ignore[override]
        self,
        *,
        hidden: int,
        hc_mult: int,
        eps: float,
    ) -> CompileKey:
        return self.CompileKey(
            HIDDEN=hidden,
            HC_MULT=hc_mult,
            BLOCK_SIZE=next_power_of_2(hidden),
            eps=eps,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0) <= 0:
            return []

        return self._trace_dispatch(self.dispatch)(
            hidden=int(getattr(hf_config, "hidden_size")),
            hc_mult=int(getattr(hf_config, "hc_mult")),
            eps=float(getattr(hf_config, "rms_norm_eps")),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        bf16_ptr = TritonWarmupTensor(torch.bfloat16)
        fp32_ptr = TritonWarmupTensor(torch.float32)
        warmup(
            bf16_ptr,
            TritonWarmupTensor(torch.int64),
            bf16_ptr,
            fp32_ptr,
            fp32_ptr,
            bf16_ptr,
            bf16_ptr,
            compile_key.eps,
            HIDDEN=compile_key.HIDDEN,
            HC_MULT=compile_key.HC_MULT,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            grid=(1, compile_key.HC_MULT + 1),
        )

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        eps: float,
        hc_mult: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds.ndim == 2
        assert previous_hidden_states.ndim == 3
        assert previous_hidden_states.shape[1] == hc_mult
        assert inputs_embeds.shape[0] == previous_hidden_states.shape[0], (
            "token dim mismatch"
        )
        assert (
            inputs_embeds.shape[1]
            == previous_hidden_states.shape[2]
            == enorm_weight.shape[0]
            == hnorm_weight.shape[0]
        )
        assert inputs_embeds.is_contiguous() and previous_hidden_states.is_contiguous()
        assert enorm_weight.is_contiguous() and hnorm_weight.is_contiguous()

        num_tokens, hidden = inputs_embeds.shape
        enorm_out = torch.empty_like(inputs_embeds)
        hnorm_out = torch.empty_like(previous_hidden_states)
        if num_tokens == 0:
            return enorm_out, hnorm_out

        compile_key = self.dispatch(hidden=hidden, hc_mult=hc_mult, eps=eps)
        self.kernel[(num_tokens, hc_mult + 1)](
            inputs_embeds,
            positions,
            previous_hidden_states,
            enorm_weight,
            hnorm_weight,
            enorm_out,
            hnorm_out,
            eps,
            HIDDEN=compile_key.HIDDEN,
            HC_MULT=compile_key.HC_MULT,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
        )
        return enorm_out, hnorm_out


class MTPSharedHeadRMSNormKernel(
    VllmJitKernel["MTPSharedHeadRMSNormKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        HIDDEN: int
        BLOCK_SIZE: int
        eps: float

    @staticmethod
    @triton.jit
    def kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        eps,
        HIDDEN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0).to(tl.int64)
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < HIDDEN
        x = tl.load(x_ptr + token_idx * HIDDEN + block, mask=mask, other=0.0)
        _rmsnorm_row(
            x,
            weight_ptr,
            out_ptr + token_idx * HIDDEN,
            block,
            mask,
            eps,
            HIDDEN,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        hidden: int,
        eps: float,
    ) -> CompileKey:
        return self.CompileKey(
            HIDDEN=hidden,
            BLOCK_SIZE=next_power_of_2(hidden),
            eps=eps,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0) <= 0:
            return []

        return self._trace_dispatch(self.dispatch)(
            hidden=int(getattr(hf_config, "hidden_size")),
            eps=float(getattr(hf_config, "rms_norm_eps")),
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        bf16_ptr = TritonWarmupTensor(torch.bfloat16)
        warmup(
            bf16_ptr,
            TritonWarmupTensor(torch.float32),
            bf16_ptr,
            compile_key.eps,
            HIDDEN=compile_key.HIDDEN,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            grid=(1,),
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        assert hidden_states.ndim == 2
        assert hidden_states.is_contiguous()
        assert weight.is_contiguous()
        num_tokens, hidden = hidden_states.shape
        out = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return out

        compile_key = self.dispatch(hidden=hidden, eps=eps)
        self.kernel[(num_tokens,)](
            hidden_states,
            weight,
            out,
            eps,
            HIDDEN=compile_key.HIDDEN,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
        )
        return out

def mtp_shared_head_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm for MTP's SharedHead.norm, on (T, H) bf16 input.

    Uses the same ``_rmsnorm_row`` body as ``fused_mtp_input_rmsnorm`` so the
    MTP draft path runs one consistent RMSNorm implementation end to end.
    """
    return _MTP_SHARED_HEAD_RMSNORM_KERNEL(hidden_states, weight, eps)


def fused_mtp_input_rmsnorm(
    inputs_embeds: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
    hc_mult: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (enorm_out, hnorm_out).

    enorm_out has the same shape as inputs_embeds (2D, [T, H]).
    hnorm_out has the same shape as previous_hidden_states (3D, [T, hc_mult, H]).
    previous_hidden_states must already be reshaped to 3D.
    """
    return _FUSED_MTP_INPUT_RMSNORM_KERNEL(
        inputs_embeds,
        positions,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        eps,
        hc_mult,
    )


_FUSED_MTP_INPUT_RMSNORM_KERNEL = FusedMTPInputRMSNormKernel()
_MTP_SHARED_HEAD_RMSNORM_KERNEL = MTPSharedHeadRMSNormKernel()
