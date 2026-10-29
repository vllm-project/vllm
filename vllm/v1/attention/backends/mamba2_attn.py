# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.ops.ssd_dispatch import (
    FlashInferSSDBackend,
    get_mamba_ssd_backend,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:
        cu_chunk_seqlens_list = [0] + list(itertools.accumulate(chunk_lens))
        # Final boundary must equal total tokens (check on host to avoid a sync)
        assert cu_chunk_seqlens_list[-1] == total
    else:
        cu_chunk_seqlens_list = [0]
    cu_chunk_seqlens = async_tensor_h2d(
        cu_chunk_seqlens_list, dtype=torch.int32, device=device
    )

    # last_chunk_indices is empty when there are no sequences (len(starts) == 0).
    last_chunk_indices_t = async_tensor_h2d(
        last_chunk_indices, dtype=torch.int32, device=device
    )
    seq_idx_chunks_t = async_tensor_h2d(
        seq_idx_chunks, dtype=torch.int32, device=device
    )
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Decode kernels take device offsets; ReplaySSM plans off CPU counts.
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return True
        return not vllm_config.cache_config.use_replayssm


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Sequence index of each scan unit (only for prefill): each Triton chunk
    # (see the base cu_chunk_seqlen_p and last_chunk_indices_p), or each
    # FlashInfer token.
    seq_idx_p: torch.Tensor | None = None

    # Internal prefill checkpoints, one entry per prefill row. The index
    # selects the row's checkpoint in the states the SSD backend returns:
    # a Triton chunk state, or the row itself for FlashInfer.
    checkpoint_chunk_idx: torch.Tensor | None = None
    checkpoint_meta: MambaPrefillCheckpointMetadata | None = None

    # FlashInfer SSD inputs (only for prefill): segments on its packed token
    # chunk grid, zero initial states for batches without cached state
    # (FlashInfer takes the state dtype from them), and each row's
    # checkpoint token with the checkpoint output.
    chunk_indices_p: torch.Tensor | None = None
    chunk_offsets_p: torch.Tensor | None = None
    zero_initial_states_p: torch.Tensor | None = None
    checkpoint_token_indices_p: torch.Tensor | None = None
    checkpoint_states_p: torch.Tensor | None = None


# FlashInfer's SSD tiles packed tokens in fixed 128-token chunks.
_FLASHINFER_SSD_CHUNK_SIZE = 128


def build_ssd_prefill_metadata(
    metadata: Mamba2AttentionMetadata,
    num_computed_tokens_p_cpu: torch.Tensor,
    query_start_loc_p_cpu: torch.Tensor,
    state_shape: tuple[int, ...],
    state_dtype: torch.dtype,
) -> Mamba2AttentionMetadata:
    """Add the prefill tensors read by the selected SSD backend."""
    query_start_loc_p = metadata.query_start_loc_p
    assert query_start_loc_p is not None
    device = query_start_loc_p.device
    checkpoint = metadata.checkpoint_meta
    checkpoint_offsets_p = checkpoint.offsets if checkpoint is not None else None
    if not isinstance(get_mamba_ssd_backend(), FlashInferSSDBackend):
        # The host offsets place a chunk boundary on each checkpoint.
        cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p, checkpoint_chunk_idx = (
            BaseMambaAttentionMetadataBuilder._build_chunk_metadata_tensors(
                metadata.chunk_size,
                num_computed_tokens_p_cpu,
                query_start_loc_p_cpu,
                device,
                checkpoint_offsets_p,
            )
        )
        return replace(
            metadata,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            seq_idx_p=seq_idx_p,
            last_chunk_indices_p=last_chunk_indices_p,
            checkpoint_chunk_idx=checkpoint_chunk_idx,
        )

    num_prefills = query_start_loc_p_cpu.numel() - 1
    num_tokens = int(query_start_loc_p_cpu[-1])
    starts = query_start_loc_p_cpu[:-1]
    # Segments start at each sequence, each checkpoint, and each grid line
    # inside a sequence. Sequences tile the tokens, so that is every line.
    segment_starts = [
        starts,
        torch.arange(
            _FLASHINFER_SSD_CHUNK_SIZE,
            num_tokens,
            _FLASHINFER_SSD_CHUNK_SIZE,
            dtype=starts.dtype,
        ),
    ]
    if checkpoint_offsets_p is not None:
        offsets = torch.tensor(checkpoint_offsets_p, dtype=starts.dtype)
        segment_starts.append((starts + offsets)[offsets > 0])
    segment_starts_cpu = torch.cat(segment_starts).unique()
    rows = torch.arange(num_prefills, dtype=torch.int32, device=device)
    states_shape = (num_prefills, *state_shape)
    metadata = replace(
        metadata,
        seq_idx_p=rows.repeat_interleave(
            query_start_loc_p.diff(), output_size=num_tokens
        ),
        chunk_indices_p=async_tensor_h2d(
            segment_starts_cpu // _FLASHINFER_SSD_CHUNK_SIZE,
            dtype=torch.int32,
            device=device,
        ),
        chunk_offsets_p=async_tensor_h2d(
            segment_starts_cpu % _FLASHINFER_SSD_CHUNK_SIZE,
            dtype=torch.int32,
            device=device,
        ),
        zero_initial_states_p=None
        if metadata.prep_initial_states
        else torch.zeros(states_shape, dtype=state_dtype, device=device),
    )
    if checkpoint is None:
        return metadata
    checkpoint_offsets = checkpoint.checkpoint_offsets
    # Each row's checkpoint goes to its own row of the output; rows without
    # one have no token to capture.
    return replace(
        metadata,
        checkpoint_chunk_idx=rows,
        checkpoint_token_indices_p=torch.where(
            checkpoint_offsets > 0, query_start_loc_p[:-1] + checkpoint_offsets, -1
        ),
        checkpoint_states_p=torch.empty(states_shape, dtype=state_dtype, device=device),
    )


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        return max_decode_query_len(vllm_config)

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size
        self.checkpoint_builder = MambaPrefillCheckpointBuilder(
            vllm_config, kv_cache_spec
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
            num_decode_draft_tokens_cpu=kwargs.get("num_decode_draft_tokens_cpu"),
        )

        if common.num_prefills == 0:
            return replace(common, chunk_size=self.chunk_size)

        num_computed_tokens_p_cpu, query_start_loc_p_cpu = self._prefill_cpu_metadata(
            common_attn_metadata,
            common.num_reqs,
            common.num_prefills,
            common.num_decode_tokens,
        )
        prep_initial_states = False
        if common.has_initial_states_p is not None:
            # Same condition as `has_initial_states_p`, but derived from CPU
            # data so it needs no D2H. `seq_lens_cpu_upper_bound` is precise
            # for prefill rows, which is all this slice covers.
            prep_initial_states = bool((num_computed_tokens_p_cpu > 0).any())
        first = common.num_reqs - common.num_prefills
        metadata = replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            checkpoint_meta=self.checkpoint_builder.build(
                common_attn_metadata,
                list(range(first, common.num_reqs)),
            ),
        )
        return build_ssd_prefill_metadata(
            metadata,
            num_computed_tokens_p_cpu,
            query_start_loc_p_cpu,
            self.kv_cache_spec.shapes[1],
            self.kv_cache_spec.dtypes[1],
        )

    def update_block_table(
        self,
        metadata: Mamba2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Mamba2AttentionMetadata:
        new_metadata = super().update_block_table(metadata, blk_table, slot_mapping)
        if metadata.checkpoint_meta is None:
            return new_metadata
        # Checkpoint destinations are block-table entries, so each group
        # re-gathers its own rather than writing into the source group's.
        return replace(
            new_metadata,
            checkpoint_meta=metadata.checkpoint_meta.regather_state_indices(blk_table),
        )
