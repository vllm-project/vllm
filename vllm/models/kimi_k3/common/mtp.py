# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Kimi-K3 MTP input preparation."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, weight, eps, hidden_size: tl.constexpr):
    x = x.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    return x * tl.rsqrt(variance + eps) * weight.to(tl.float32)


class FusedMTPInputKernel(VllmTritonJitKernel["FusedMTPInputKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        positions_dtype: torch.dtype
        dtype: torch.dtype
        inputs_embeds_stride: int
        previous_hidden_states_stride: int
        output_stride: int
        hidden_size: int
        block_size: int

    @staticmethod
    @triton.jit
    def kernel(
        positions_ptr,
        inputs_embeds_ptr,
        previous_hidden_states_ptr,
        enorm_weight_ptr,
        hnorm_weight_ptr,
        output_ptr,
        eps,
        inputs_embeds_stride,
        previous_hidden_states_stride,
        output_stride,
        hidden_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token_idx = tl.program_id(0).to(tl.int64)
        input_idx = tl.program_id(1)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size

        if input_idx == 0:
            position = tl.load(positions_ptr + token_idx)
            values = tl.load(
                inputs_embeds_ptr + token_idx * inputs_embeds_stride + offsets,
                mask=mask & (position != 0),
                other=0.0,
            )
            weight = tl.load(enorm_weight_ptr + offsets, mask=mask, other=0.0)
        else:
            values = tl.load(
                previous_hidden_states_ptr
                + token_idx * previous_hidden_states_stride
                + offsets,
                mask=mask,
                other=0.0,
            )
            weight = tl.load(hnorm_weight_ptr + offsets, mask=mask, other=0.0)

        output = _rms_norm(values, weight, eps, hidden_size)
        tl.store(
            output_ptr + token_idx * output_stride + input_idx * hidden_size + offsets,
            output,
            mask=mask,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        positions_dtype: torch.dtype,
        dtype: torch.dtype,
        inputs_embeds_stride: int,
        previous_hidden_states_stride: int,
        output_stride: int,
        hidden_size: int,
    ) -> CompileKey:
        return self.CompileKey(
            positions_dtype=positions_dtype,
            dtype=dtype,
            inputs_embeds_stride=triton_scalar_specialization_rep(
                inputs_embeds_stride
            ),
            previous_hidden_states_stride=triton_scalar_specialization_rep(
                previous_hidden_states_stride
            ),
            output_stride=triton_scalar_specialization_rep(output_stride),
            hidden_size=hidden_size,
            block_size=triton.next_power_of_2(hidden_size),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        config = vllm_config.model_config.hf_text_config
        hidden_size = config.hidden_size
        return self._trace_dispatch(self.dispatch)(
            positions_dtype=torch.int64,
            dtype=vllm_config.model_config.dtype,
            inputs_embeds_stride=hidden_size,
            previous_hidden_states_stride=hidden_size,
            output_stride=2 * hidden_size,
            hidden_size=hidden_size,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            positions=TritonWarmupTensor(compile_key.positions_dtype),
            inputs_embeds=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hidden_size),
            ),
            previous_hidden_states=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.hidden_size),
            ),
            enorm_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.hidden_size,),
            ),
            hnorm_weight=TritonWarmupTensor(
                compile_key.dtype,
                shape=(compile_key.hidden_size,),
            ),
            output=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, 2 * compile_key.hidden_size),
            ),
            eps=0.0,
        )

    @kernel_launcher
    def __call__(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        output: torch.Tensor,
        eps: float,
    ) -> LaunchSpec:
        num_tokens, hidden_size = inputs_embeds.shape
        return (num_tokens, 2), dict(
            inputs_embeds_stride=inputs_embeds.stride(0),
            previous_hidden_states_stride=previous_hidden_states.stride(0),
            output_stride=output.stride(0),
            hidden_size=hidden_size,
            block_size=triton.next_power_of_2(hidden_size),
        )



def fused_mtp_input(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mask and normalize both MTP inputs into the projection layout."""
    num_tokens, hidden_size = inputs_embeds.shape
    output = torch.empty(
        num_tokens,
        2 * hidden_size,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    if num_tokens == 0:
        return output

    _FUSED_MTP_INPUT_KERNEL(
        positions,
        inputs_embeds,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        output,
        eps,
    )
    return output


_FUSED_MTP_INPUT_KERNEL = FusedMTPInputKernel()
