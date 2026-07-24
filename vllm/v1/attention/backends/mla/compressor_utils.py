# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.triton_utils import tl, triton


class CompressedSlotMappingKernel(
    VllmJitKernel["CompressedSlotMappingKernel.CompileKey"]
):
    TRITON_BLOCK_SIZE = 1024

    @dataclass(frozen=True)
    class CompileKey:
        COMPRESS_RATIO: int
        TRITON_BLOCK_SIZE: int

    @staticmethod
    @triton.jit
    def kernel(
        # [num_tokens]
        slot_mapping_ptr,
        # [num_reqs + 1]
        query_start_loc_ptr,
        # [num_reqs]
        seq_lens_ptr,
        # [num_reqs, max_num_blocks]
        block_table_ptr,
        block_table_stride,
        block_size,
        COMPRESS_RATIO: tl.constexpr,
        PAD_ID: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        query_len = query_end - query_start

        seq_len = tl.load(seq_lens_ptr + batch_idx)
        start_pos = seq_len - query_len

        for i in range(0, query_len, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            mask = offset < query_len

            pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
            is_valid = (pos + 1) % COMPRESS_RATIO == 0
            pos_after_compress = pos // COMPRESS_RATIO

            block_ids = pos_after_compress // block_size
            block_numbers = tl.load(
                block_table_ptr + batch_idx * block_table_stride + block_ids,
                mask=mask & is_valid,
            )
            slot_ids = block_numbers * block_size + pos_after_compress % block_size

            # NOTE
            slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
            tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
    ) -> CompileKey:
        return self.CompileKey(
            COMPRESS_RATIO=compress_ratio,
            TRITON_BLOCK_SIZE=self.TRITON_BLOCK_SIZE,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None) or ()
        compress_ratios = [int(ratio) for ratio in compress_ratios if int(ratio) > 1]
        if not compress_ratios:
            return []
        return self._trace_dispatch(self.dispatch)(
            compress_ratio=compress_ratios,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            TritonWarmupTensor(torch.int64),
            int32_ptr,
            int32_ptr,
            int32_ptr,
            1,
            1,
            compile_key.COMPRESS_RATIO,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=compile_key.TRITON_BLOCK_SIZE,
            grid=(1,),
        )

    def __call__(
        self,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
    ) -> None:
        self.kernel[(block_table.shape[0],)](
            slot_mapping,
            query_start_loc,
            seq_lens,
            block_table,
            block_table.stride(0),
            block_size,
            compress_ratio,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=self.TRITON_BLOCK_SIZE,
        )


_COMPRESSED_SLOT_MAPPING_KERNEL = CompressedSlotMappingKernel()


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    _COMPRESSED_SLOT_MAPPING_KERNEL(
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
    )
    return slot_mapping
