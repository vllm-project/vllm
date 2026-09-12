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
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

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
        has_token_slot_mapping: bool

    @staticmethod
    @triton.jit(do_not_specialize=["block_table_stride"])
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
        # [num_tokens] slot mapping of the uncompressed tokens
        token_slot_mapping_ptr,
        COMPRESS_RATIO: tl.constexpr,
        PAD_ID: tl.constexpr,
        HAS_TOKEN_SLOT_MAPPING: tl.constexpr,
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
            if HAS_TOKEN_SLOT_MAPPING:
                # A token whose own slot is padded (SWA bounded replay) writes
                # no compressed state either.
                token_slot = tl.load(
                    token_slot_mapping_ptr + query_start + offset,
                    mask=mask,
                    other=PAD_ID,
                )
                is_valid = is_valid & (token_slot != PAD_ID)
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
        block_size: int,
        token_slot_mapping: Any,
    ) -> CompileKey:
        return self.CompileKey(
            compress_ratio=compress_ratio,
            triton_block_size=self.TRITON_BLOCK_SIZE,
            block_size=triton_scalar_specialization_rep(block_size),
            has_token_slot_mapping=token_slot_mapping is not None,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_text_config = vllm_config.model_config.hf_text_config
        configured_ratios = (
            *(getattr(hf_text_config, "compress_ratios", None) or ()),
            getattr(hf_text_config, "index_kpool", 1) or 1,
        )
        compress_ratios = tuple(
            dict.fromkeys(int(ratio) for ratio in configured_ratios if int(ratio) > 1)
        )
        if not compress_ratios:
            return []
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                *(
                    dict(
                        compress_ratio=ratio,
                        block_size=vllm_config.cache_config.block_size // ratio,
                        token_slot_mapping=token_slot_mapping,
                    )
                    for ratio in compress_ratios
                    for token_slot_mapping in (None, TritonWarmupTensor(torch.int64))
                )
            )
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            slot_mapping=TritonWarmupTensor(torch.int64),
            query_start_loc=int32_ptr,
            seq_lens=int32_ptr,
            block_table=int32_ptr,
            block_size=compile_key.block_size,
            compress_ratio=compile_key.compress_ratio,
            token_slot_mapping=(
                TritonWarmupTensor(torch.int64)
                if compile_key.has_token_slot_mapping
                else None
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
        token_slot_mapping: torch.Tensor | None = None,
    ) -> LaunchSpec:
        return (block_table.shape[0],), dict(
            block_table_stride=block_table.stride(0),
            # Unread when there is no token slot mapping.
            token_slot_mapping_ptr=(
                slot_mapping if token_slot_mapping is None else token_slot_mapping
            ),
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=-1,
            HAS_TOKEN_SLOT_MAPPING=token_slot_mapping is not None,
            TRITON_BLOCK_SIZE=self.TRITON_BLOCK_SIZE,
        )


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    token_slot_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slot mapping of the compressed states closed by ``num_tokens`` tokens.

    With ``token_slot_mapping`` (the tokens' own slot mapping), a padded token
    closes no state: SWA bounded replay pads the slots of recomputed tokens
    whose KV is already cached.
    """
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
        token_slot_mapping,
    )
    return slot_mapping


_COMPRESSED_SLOT_MAPPING_KERNEL = CompressedSlotMappingKernel()
