# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    triton_scalar_specialization_rep,
)
from vllm.transformers_utils.deepseek_v4_utils import get_compress_ratios
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.decode_prologue import _compressed_slot_body

_DSPARK_SWA_INDEX_ALIGNMENT = 64


def get_dspark_swa_index_width(
    window_size: int,
    num_speculative_tokens: int,
) -> int:
    """Return the padded width of non-causal DSpark SWA indices."""
    width = max(int(window_size), 0) + max(int(num_speculative_tokens), 0)
    return cdiv(width, _DSPARK_SWA_INDEX_ALIGNMENT) * _DSPARK_SWA_INDEX_ALIGNMENT


class CompressedSlotMappingKernel(
    VllmTritonJitKernel["CompressedSlotMappingKernel.CompileKey"]
):
    TRITON_BLOCK_SIZE = 1024

    @dataclass(frozen=True)
    class CompileKey:
        compress_ratio: int
        triton_block_size: int
        block_size: int
        dcp_world_size: int
        cp_interleave: int

    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "block_table_stride",
            "num_tokens",
            "out_len",
            "num_rows",
        ]
    )
    def kernel(
        # [out_len]
        compressed_slot_mapping_ptr,
        # [num_tokens] the tokens' own slots
        slot_mapping_ptr,
        # [num_reqs + 1]
        query_start_loc_ptr,
        # [num_reqs]
        seq_lens_ptr,
        # [num_rows, max_num_blocks]
        block_table_ptr,
        block_table_stride,
        block_size,
        dcp_world_size,
        cp_interleave,
        num_tokens,
        out_len,
        num_rows,
        COMPRESS_RATIO: tl.constexpr,
        PAD_ID: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
    ):
        _compressed_slot_body(
            compressed_slot_mapping_ptr,
            slot_mapping_ptr,
            query_start_loc_ptr,
            seq_lens_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            COMPRESS_RATIO,
            dcp_world_size,
            cp_interleave,
            num_tokens,
            out_len,
            num_rows,
            PAD_ID=PAD_ID,
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        block_size: int,
        dcp_world_size: int,
        cp_interleave: int,
    ) -> CompileKey:
        return self.CompileKey(
            compress_ratio=compress_ratio,
            triton_block_size=self.TRITON_BLOCK_SIZE,
            block_size=triton_scalar_specialization_rep(block_size),
            dcp_world_size=triton_scalar_specialization_rep(dcp_world_size),
            cp_interleave=triton_scalar_specialization_rep(cp_interleave),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_text_config = vllm_config.model_config.hf_text_config
        configured_ratios = (
            *get_compress_ratios(hf_text_config),
            getattr(hf_text_config, "index_kpool", 1) or 1,
        )
        compress_ratios = tuple(
            dict.fromkeys(int(ratio) for ratio in configured_ratios if int(ratio) > 1)
        )
        if not compress_ratios:
            return []
        parallel_config = vllm_config.parallel_config
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                *(
                    dict(
                        compress_ratio=ratio,
                        block_size=vllm_config.cache_config.block_size // ratio,
                        dcp_world_size=parallel_config.decode_context_parallel_size,
                        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
                    )
                    for ratio in compress_ratios
                )
            )
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            compressed_slot_mapping=TritonWarmupTensor(torch.int64),
            slot_mapping=TritonWarmupTensor(torch.int64),
            query_start_loc=int32_ptr,
            seq_lens=int32_ptr,
            block_table=int32_ptr,
            block_size=compile_key.block_size,
            compress_ratio=compile_key.compress_ratio,
            dcp_world_size=compile_key.dcp_world_size,
            cp_interleave=compile_key.cp_interleave,
            num_tokens=1,
            out_len=1,
            num_rows=1,
        )

    @kernel_launcher
    def __call__(
        self,
        compressed_slot_mapping: torch.Tensor,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
        dcp_world_size: int,
        cp_interleave: int,
        num_tokens: int,
        out_len: int,
    ) -> LaunchSpec:
        num_rows = block_table.shape[0]
        tail_blocks = cdiv(max(out_len - num_tokens, 0), self.TRITON_BLOCK_SIZE)
        return (num_rows + tail_blocks,), dict(
            block_table_stride=block_table.stride(0),
            num_tokens=num_tokens,
            out_len=out_len,
            num_rows=num_rows,
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=self.TRITON_BLOCK_SIZE,
        )


def get_compressed_slot_mapping(
    num_tokens: int,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    dcp_world_size: int = 1,
    cp_interleave: int = 1,
) -> torch.Tensor:
    """Slot mapping for writing the compressed states of ``num_tokens`` tokens.

    Every ``compress_ratio`` tokens share one compressed state, written by the
    last of them: that token maps to the state's slot, the others to PAD. A
    token whose own ``slot_mapping`` entry is PAD maps to PAD too: SWA bounded
    replay recomputes tokens whose KV is cached already, and their compressed
    states must not be rewritten either.
    """
    if out is not None:
        # The kernel clears its own padding tail; callers must size ``out``
        # to the padded width the captured graphs read.
        compressed_slot_mapping = out
        out_len = out.numel()
    else:
        compressed_slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )
        out_len = num_tokens

    _COMPRESSED_SLOT_MAPPING_KERNEL(
        compressed_slot_mapping,
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        dcp_world_size,
        cp_interleave,
        num_tokens,
        out_len,
    )
    return compressed_slot_mapping[:num_tokens]


_COMPRESSED_SLOT_MAPPING_KERNEL = CompressedSlotMappingKernel()
