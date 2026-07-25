# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


def save_partial_states(
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
) -> None:
    _SAVE_PARTIAL_STATES_KERNEL(
        kv,
        score,
        ape,
        positions,
        state_cache,
        slot_mapping,
        block_size,
        state_width,
        compress_ratio,
        pdl_kwargs,
    )




class SavePartialStatesKernel(VllmJitKernel["SavePartialStatesKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        HEAD_SIZE: int
        TRITON_BLOCK_SIZE: int
        STATE_WIDTH: int
        COMPRESS_RATIO: int
        kv_stride: int
        score_stride: int
        ape_stride: int
        state_cache_stride0: int
        state_cache_stride1: int
        block_size: int
        launch_pdl: bool

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
        state_width: int,
        compress_ratio: int,
        kv_stride: int,
        score_stride: int,
        ape_stride: int,
        state_cache_stride0: int,
        state_cache_stride1: int,
        block_size: int,
        launch_pdl: bool,
    ) -> CompileKey:
        return self.CompileKey(
            HEAD_SIZE=head_size,
            TRITON_BLOCK_SIZE=next_power_of_2(head_size),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=compress_ratio,
            kv_stride=kv_stride,
            score_stride=score_stride,
            ape_stride=ape_stride,
            state_cache_stride0=state_cache_stride0,
            state_cache_stride1=state_cache_stride1,
            block_size=block_size,
            launch_pdl=launch_pdl,
        )

    @staticmethod
    def _uses_compress_ratio(
        *,
        compress_ratio: int,
        compress_ratios: tuple[int, ...],
    ) -> bool:
        return compress_ratio in compress_ratios

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        head_dim = int(getattr(hf_config, "head_dim", 0) or 0)
        compress_ratios = tuple(
            sorted(
                {
                    int(compress_ratio)
                    for compress_ratio in getattr(hf_config, "compress_ratios", ())
                    if int(compress_ratio) > 1
                }
            )
        )
        if head_dim <= 0 or not compress_ratios:
            return []

        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    head_size=2 * head_dim,
                    state_width=2 * head_dim,
                    compress_ratio=4,
                    kv_stride=4 * head_dim,
                    score_stride=4 * head_dim,
                    ape_stride=2 * head_dim,
                    state_cache_stride0=16 * head_dim,
                    state_cache_stride1=4 * head_dim,
                    block_size=4,
                    launch_pdl=False,
                ),
                dict(
                    head_size=head_dim,
                    state_width=head_dim,
                    compress_ratio=128,
                    kv_stride=2 * head_dim,
                    score_stride=2 * head_dim,
                    ape_stride=head_dim,
                    state_cache_stride0=16 * head_dim,
                    state_cache_stride1=2 * head_dim,
                    block_size=8,
                    launch_pdl=False,
                ),
            ),
            compress_ratios=compress_ratios,
            _when=self._uses_compress_ratio,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        fp32_ptr = TritonWarmupTensor(torch.float32)
        warmup(
            fp32_ptr,
            compile_key.kv_stride,
            fp32_ptr,
            compile_key.score_stride,
            fp32_ptr,
            compile_key.ape_stride,
            TritonWarmupTensor(torch.int64),
            fp32_ptr,
            compile_key.state_cache_stride0,
            compile_key.state_cache_stride1,
            TritonWarmupTensor(torch.int64),
            compile_key.block_size,
            HEAD_SIZE=compile_key.HEAD_SIZE,
            TRITON_BLOCK_SIZE=compile_key.TRITON_BLOCK_SIZE,
            STATE_WIDTH=compile_key.STATE_WIDTH,
            COMPRESS_RATIO=compile_key.COMPRESS_RATIO,
            launch_pdl=compile_key.launch_pdl,
            grid=(1,),
        )

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
    ) -> None:
        """Write packed [kv, score+ape] partial states into the compressor cache.

        One program per token; pads (slot_id == -1) are skipped.
        """
        num_actual = slot_mapping.shape[0]
        head_size = kv.shape[-1]
        compile_key = self.dispatch(
            head_size=head_size,
            state_width=state_width,
            compress_ratio=compress_ratio,
            kv_stride=kv.stride(0),
            score_stride=score.stride(0),
            ape_stride=ape.stride(0),
            state_cache_stride0=state_cache.stride(0),
            state_cache_stride1=state_cache.stride(1),
            block_size=block_size,
            launch_pdl=bool((pdl_kwargs or {}).get("launch_pdl", False)),
        )
        self.kernel[(num_actual,)](
            kv,
            compile_key.kv_stride,
            score,
            compile_key.score_stride,
            ape,
            compile_key.ape_stride,
            positions,
            state_cache,
            compile_key.state_cache_stride0,
            compile_key.state_cache_stride1,
            slot_mapping,
            compile_key.block_size,
            HEAD_SIZE=compile_key.HEAD_SIZE,
            TRITON_BLOCK_SIZE=compile_key.TRITON_BLOCK_SIZE,
            STATE_WIDTH=compile_key.STATE_WIDTH,
            COMPRESS_RATIO=compile_key.COMPRESS_RATIO,
            **(pdl_kwargs or {}),
        )


_SAVE_PARTIAL_STATES_KERNEL = SavePartialStatesKernel()
