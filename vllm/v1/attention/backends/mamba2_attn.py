# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

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
        cu_chunk_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(chunk_lens)),
            device=device,
            dtype=torch.int32,
        )
        # Final boundary must equal total tokens
        assert int(cu_chunk_seqlens[-1].item()) == total
    else:
        cu_chunk_seqlens = torch.tensor([0], device=device, dtype=torch.int32)

    last_chunk_indices_t = (
        torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)
        if len(starts) > 0
        else torch.empty((0,), device=device, dtype=torch.int32)
    )
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, device=device, dtype=torch.int32)
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None
    # cu_chunk_seqlen_p is a tensor of shape (nchunks+1,) that contains, for
    # each chunk, its offests into the varlen sequence dimension. It is defined
    # such that the i-th chunk contains tokens from cu_chunk_seqlen_p[i] to
    # cu_chunk_seqlen_p[i+1].
    cu_chunk_seqlen_p: torch.Tensor | None = None
    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    last_chunk_indices_p: torch.Tensor | None = None


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
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

    def _compute_chunk_metadata(
        self,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute chunk-specific metadata for Mamba2.

        The code below carefully constructs the chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. For every sequence, we are guaranteed that we can
           retrieve the mamba state *every* chunk_size tokens.
        Constraint (1) dramatically simplifies the mamba2 kernels.
        Constraint (2) dramatically simplifies the implementation
        of prefix caching for mamba2 (wip). We need to take care
        of the interaction with chunked prefill in order to
        satisfy constraint (2).
        """
        # TODO (tdoublep): This code could probably be optimized.
        cu_chunk_seqlen = []
        seq_idx = []
        last_chunk_indices = []
        seqlen_pos = 0

        for req_idx in range(num_prefills):
            this_num_computed = num_computed_tokens_p_cpu[req_idx].item()
            this_new_tokens = (
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )

            # if computed tokens are not chunk-aligned, use the first
            # chunk to finish it off
            if this_num_computed % self.chunk_size != 0:
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                # how many tokens to finish the chunk?
                chunk_len = (
                    cdiv(this_num_computed, self.chunk_size) * self.chunk_size
                    - this_num_computed
                )
                # we can only use at most this_new_tokens
                chunk_len = min(chunk_len, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            n_chunks = cdiv(this_new_tokens, self.chunk_size)
            for chunk in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(self.chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Mamba2AttentionMetadata:
        common = self._compute_common_metadata(common_attn_metadata)

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = (
                torch.any(common.has_initial_states_p).item()
                if common.has_initial_states_p is not None
                else False
            )

            num_reqs = common.num_reqs
            num_prefills = common.num_prefills
            num_decode_tokens = common.num_decode_tokens

            num_computed_tokens_cpu = (
                common_attn_metadata.compute_num_computed_tokens().cpu()
            )
            num_computed_tokens_p_cpu = num_computed_tokens_cpu[
                num_reqs - num_prefills : num_reqs
            ]
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )

            cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
                num_prefills,
                num_computed_tokens_p_cpu,
                query_start_loc_p_cpu,
            )

            seq_idx_p = torch.as_tensor(
                seq_idx,
                device=common_attn_metadata.query_start_loc.device,
                dtype=torch.int32,
            )
            cu_chunk_seqlen_p = torch.as_tensor(
                cu_chunk_seqlen,
                device=common_attn_metadata.query_start_loc.device,
                dtype=torch.int32,
            )
            last_chunk_indices_p = torch.as_tensor(
                last_chunk_indices,
                device=common_attn_metadata.query_start_loc.device,
                dtype=torch.int32,
            )

        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
        )
