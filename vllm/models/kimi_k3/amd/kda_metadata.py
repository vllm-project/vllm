# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 specialization of GDN attention metadata.

The request classification and cudagraph staging intentionally mirror
``GDNAttentionMetadataBuilder``. Only the FLA chunk metadata is built
differently on device rather than on the host.
"""

from dataclasses import dataclass, fields

import torch

from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

logger = init_logger(__name__)

_BLOCK_T = 256
_MIN_BLOCK_N = 128


@triton.jit(do_not_specialize=["N"])
def _chunk_metadata_kernel(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    N,
    BT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    i_n = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    is_seq = offs_n < N
    bos = tl.load(cu_seqlens + offs_n, mask=is_seq, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens + offs_n + 1, mask=is_seq, other=0).to(tl.int32)
    nt = tl.where(is_seq, tl.cdiv(eos - bos, BT), 0)

    base = tl.sum(tl.where(offs_n < i_n, nt, 0))
    num_chunks = tl.sum(tl.where(offs_n == i_n, nt, 0))

    tl.store(chunk_offsets + i_n, base)
    if i_n == 0:
        tl.store(chunk_offsets + N, tl.sum(nt))

    for t0 in range(0, num_chunks, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < num_chunks
        row = (base + offs_t) * 2
        tl.store(chunk_indices + row, tl.full([BLOCK_T], i_n, tl.int32), mask=mask_t)
        tl.store(chunk_indices + row + 1, offs_t.to(tl.int32), mask=mask_t)


def prepare_chunk_metadata_device(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FLA chunk metadata on device, with no host<->device transfer."""
    num_seqs = cu_seqlens_cpu.numel() - 1
    seq_lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    num_chunks = int(((seq_lens + chunk_size - 1) // chunk_size).sum())

    chunk_indices = torch.empty(
        num_chunks, 2, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    chunk_offsets = torch.empty(
        num_seqs + 1, dtype=torch.int64, device=cu_seqlens.device
    )
    _chunk_metadata_kernel[(num_seqs,)](
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_seqs,
        BT=chunk_size,
        BLOCK_N=max(_MIN_BLOCK_N, next_power_of_2(num_seqs + 1)),
        BLOCK_T=_BLOCK_T,
        num_warps=4,
    )
    return chunk_indices, chunk_offsets


@dataclass
class KDACheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    state_indices: torch.Tensor


@dataclass
class KimiK3ROCmKDAMetadata(GDNAttentionMetadata):
    checkpoint: KDACheckpointMetadata | None = None


class KimiK3ROCmKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        checkpoint_enabled = (
            self.vllm_config.cache_config.mamba_cache_mode == "align"
            and isinstance(self.kv_cache_spec, MambaSpec)
            and self.kv_cache_spec.num_prefill_checkpoint_blocks > 0
        )
        if not checkpoint_enabled:
            return attn_metadata
        return KimiK3ROCmKDAMetadata(
            **{f.name: getattr(attn_metadata, f.name) for f in fields(attn_metadata)},
            checkpoint=self._build_checkpoint_metadata(
                common_attn_metadata, attn_metadata
            ),
        )

    def _build_checkpoint_metadata(
        self,
        m: CommonAttentionMetadata,
        attn_metadata: GDNAttentionMetadata,
    ) -> KDACheckpointMetadata | None:
        if attn_metadata.num_prefills == 0:
            return None
        # The chunk kernel is handed the prefill tail of a decode-first batch,
        # so these rows have to be that same contiguous group. A speculative
        # batch interleaves its non-spec rows instead, which the layer's opt-in
        # already refuses.
        if attn_metadata.spec_sequence_masks is not None:
            return None
        assert m.seq_lens_cpu_upper_bound is not None
        num_decodes = attn_metadata.num_decodes
        request_rows = list(
            range(num_decodes, num_decodes + attn_metadata.num_prefills)
        )
        all_query_lens = m.query_start_loc_cpu.diff().tolist()
        query_lens = [all_query_lens[row] for row in request_rows]
        seq_lens = m.seq_lens_cpu_upper_bound.tolist()
        block_size = self.kv_cache_spec.block_size
        hash_block_size = self.vllm_config.cache_config.prefix_match_unit or block_size
        speculative_config = self.vllm_config.speculative_config
        drop_eagle_block = (
            speculative_config is not None and speculative_config.use_eagle_block_drop()
        )
        checkpoint_offsets = []
        checkpoint_cols = []
        for row, query_len in zip(request_rows, query_lens):
            seq_len = seq_lens[row]
            query_start = seq_len - query_len
            checkpoint_position = get_mamba_prefill_checkpoint_position(
                seq_len,
                hash_block_size,
                drop_eagle_block=drop_eagle_block,
            )
            offset = checkpoint_position - query_start
            valid = is_mamba_prefill_checkpoint_valid(
                query_start=query_start,
                query_end=seq_len,
                checkpoint_position=checkpoint_position,
                hash_block_size=hash_block_size,
                mamba_block_size=block_size,
                checkpoint_alignment=self.kv_cache_spec.prefill_checkpoint_alignment,
            )
            checkpoint_offsets.append(offset if valid else 0)
            checkpoint_cols.append(cdiv(seq_len, block_size) - 2 if valid else -1)
        if not any(checkpoint_offsets):
            return None

        device = m.query_start_loc.device
        checkpoint_offsets_tensor = async_tensor_h2d(
            checkpoint_offsets, device, torch.int32
        )
        request_rows_tensor = async_tensor_h2d(request_rows, device, torch.int64)
        checkpoint_cols_tensor = async_tensor_h2d(checkpoint_cols, device, torch.int64)
        checkpoint_state_indices = m.block_table_tensor[
            request_rows_tensor, checkpoint_cols_tensor
        ].to(torch.int32)
        # ROCm's `fused_kda_chunk` disables the export for sequences whose
        # rows are negative
        checkpoint_state_indices = torch.where(
            (checkpoint_cols_tensor >= 0) & (checkpoint_state_indices != NULL_BLOCK_ID),
            checkpoint_state_indices,
            -1,
        )
        return KDACheckpointMetadata(
            checkpoint_offsets_tensor,
            checkpoint_state_indices,
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_chunk_metadata_device(
            prefill_query_start_loc,
            prefill_query_start_loc_cpu,
            FLA_CHUNK_SIZE,
        )


class KimiK3ROCmKDABackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KDA_ROCM"

    @staticmethod
    def get_builder_cls() -> type[KimiK3ROCmKDAMetadataBuilder]:
        return KimiK3ROCmKDAMetadataBuilder
