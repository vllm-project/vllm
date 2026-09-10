# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


class SavePartialStatesKernel(
    VllmTritonJitKernel["SavePartialStatesKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        head_size: int
        triton_block_size: int
        state_width: int
        compress_ratio: int
        kv_stride: int
        score_stride: int
        ape_stride: int
        state_cache_stride0: int
        state_cache_stride1: int
        block_size: int

    @staticmethod
    @triton.jit
    def kernel(
        kv_ptr,
        kv_stride,
        score_ptr,
        score_stride,
        ape_ptr,
        ape_stride,
        positions_ptr,
        state_cache_ptr,
        state_cache_stride0,
        state_cache_stride1,
        slot_mapping_ptr,
        block_size,
        HEAD_SIZE: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
        # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
        STATE_WIDTH: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        slot_id = tl.load(slot_mapping_ptr + token_idx)

        # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
        # by vLLM).  During CUDA graph replay the batch may contain padding
        # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
        # illegal memory access.
        if slot_id < 0:
            return

        block_idx = slot_id // block_size
        pos_in_block = slot_id % block_size
        base_ptr = (
            state_cache_ptr
            + block_idx * state_cache_stride0
            + pos_in_block * state_cache_stride1
        )

        block = tl.arange(0, TRITON_BLOCK_SIZE)
        mask = block < HEAD_SIZE

        kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
        tl.store(base_ptr + block, kv, mask=mask)

        # Fused: score += ape[position % compress_ratio]
        position = tl.load(positions_ptr + token_idx)
        ape_row = position % COMPRESS_RATIO
        ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
        score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
        tl.store(
            base_ptr + STATE_WIDTH + block,
            score + ape,
            mask=mask,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        head_size: int,
        **compile_key_fields: int,
    ) -> CompileKey:
        return self.CompileKey(
            **compile_key_fields,
            head_size=head_size,
            triton_block_size=next_power_of_2(head_size),
        )

    def get_warmup_keys(
        self,
        *,
        head_dim: int,
        compress_ratio: int,
    ) -> list[CompileKey]:
        if head_dim <= 0 or compress_ratio not in (4, 128):
            return []

        coefficient = 2 if compress_ratio == 4 else 1
        return self._trace_dispatch(self.dispatch)(
            head_size=coefficient * head_dim,
            state_width=coefficient * head_dim,
            compress_ratio=compress_ratio,
            kv_stride=2 * coefficient * head_dim,
            score_stride=2 * coefficient * head_dim,
            ape_stride=coefficient * head_dim,
            state_cache_stride0=8 * coefficient * head_dim,
            state_cache_stride1=2 * coefficient * head_dim,
            block_size=4 if compress_ratio == 4 else 8,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            kv=TritonWarmupTensor(
                torch.float32,
                shape=(1, compile_key.head_size),
                strides=(compile_key.kv_stride, 1),
            ),
            score=TritonWarmupTensor(
                torch.float32,
                shape=(1, 1),
                strides=(compile_key.score_stride, 1),
            ),
            ape=TritonWarmupTensor(
                torch.float32,
                shape=(1, 1),
                strides=(compile_key.ape_stride, 1),
            ),
            positions=TritonWarmupTensor(torch.int64),
            state_cache=TritonWarmupTensor(
                torch.float32,
                shape=(1, 1, compile_key.state_width),
                strides=(
                    compile_key.state_cache_stride0,
                    compile_key.state_cache_stride1,
                    1,
                ),
            ),
            slot_mapping=TritonWarmupTensor(torch.int64),
            block_size=compile_key.block_size,
            state_width=compile_key.state_width,
            compress_ratio=compile_key.compress_ratio,
        )

    @kernel_launcher
    def __call__(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        state_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        state_width: int,
        compress_ratio: int,
        pdl_kwargs: dict | None = None,
    ) -> LaunchSpec:
        """Write packed [kv, score+ape] partial states into the compressor cache.

        One program per token; pads (slot_id == -1) are skipped.
        """
        num_actual = slot_mapping.shape[0]
        head_size = kv.shape[-1]
        return (num_actual,), dict(
            kv_stride=kv.stride(0),
            score_stride=score.stride(0),
            ape_stride=ape.stride(0),
            state_cache_stride0=state_cache.stride(0),
            state_cache_stride1=state_cache.stride(1),
            block_size=block_size,
            HEAD_SIZE=head_size,
            TRITON_BLOCK_SIZE=next_power_of_2(head_size),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            **(pdl_kwargs or {}),
        )


_SAVE_PARTIAL_STATES_KERNEL = SavePartialStatesKernel()
