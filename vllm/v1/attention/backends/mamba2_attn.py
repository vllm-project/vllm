# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


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

    # Batch-invariant mode is only sound for Mamba2 layers when prefill, chunked
    # prefill and decode produce the same bits. The plain SSD/SSU pair does not
    # (supports_batch_invariance() stays False); exact-replay mode does, so the
    # mamba backend selector accepts this backend when that cache option is on.
    supports_batch_invariance_with_exact_replay: ClassVar[bool] = True


@dataclass
class ExactReplayMetadata:
    """Per-step metadata for Mamba2 exact-replay mode.

    One instance describes the prefill rows of a step, another the decode
    rows. Every sequence is re-expanded to an *augmented* sequence that starts
    at its last chunk boundary: the buffered inputs of the partial chunk come
    first, then this step's tokens. Index tensors are int64 device tensors;
    the varlen/chunk metadata consumed by the SSD kernels is int32.

    The metadata refers to sequences by their row in the batch and never to
    state slots: the slots come from the layer's own state indices at call
    time. This keeps one instance valid for every KV cache group of a hybrid
    model, where the model runner builds the metadata once and only swaps the
    block table per group.

    Attributes:
        num_aug_tokens: total number of tokens in the augmented layout.
        buffered_seq: batch row of the sequence each buffered token belongs to.
        buffered_pos: position of each buffered token inside its slot's buffer.
        buffered_dst: destination of each buffered token in the augmented layout.
        step_dst: destination of each of this step's tokens in the augmented
            layout (in input order).
        cu_seqlens: ``(num_seqs + 1,)`` cumulative augmented sequence lengths.
        cu_chunk_seqlens: ``(num_chunks + 1,)`` chunk offsets in the augmented
            layout.
        last_chunk_indices: ``(num_seqs,)`` index of each sequence's last chunk.
        seq_idx: ``(num_chunks,)`` sequence index of each chunk.
        has_boundary_state: ``(num_seqs,)`` bool, whether the slot holds a valid
            boundary state (False while a sequence is still in its first chunk).
        boundary_rows: sequences that complete at least one chunk this step.
        boundary_chunk_idx: for each of those sequences, the chunk (indexed into
            the kernel's intermediate states) whose end state becomes the new
            boundary state.
        store_src: positions in the augmented layout of the trailing partial
            chunk's tokens that must be stored into the buffers.
        store_seq: batch row of the sequence each stored token belongs to.
        store_pos: destination position of each stored token.
    """

    num_aug_tokens: int
    buffered_seq: torch.Tensor
    buffered_pos: torch.Tensor
    buffered_dst: torch.Tensor
    step_dst: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_chunk_seqlens: torch.Tensor
    last_chunk_indices: torch.Tensor
    seq_idx: torch.Tensor
    has_boundary_state: torch.Tensor
    boundary_rows: torch.Tensor
    boundary_chunk_idx: torch.Tensor
    store_src: torch.Tensor
    store_seq: torch.Tensor
    store_pos: torch.Tensor


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def build_exact_replay_metadata(
    num_computed: list[int],
    query_lens: list[int],
    chunk_size: int,
    device: torch.device,
) -> ExactReplayMetadata:
    """Build :class:`ExactReplayMetadata` on the host.

    Args:
        num_computed: per sequence, tokens processed before this step.
        query_lens: per sequence, tokens scheduled in this step.
        chunk_size: the model's SSD chunk size.
        device: device for the returned tensors.

    Returns:
        The metadata describing the augmented layout of these sequences.
    """
    buffered_seq: list[int] = []
    buffered_pos: list[int] = []
    buffered_dst: list[int] = []
    step_dst: list[int] = []
    cu_seqlens = [0]
    cu_chunk: list[int] = []
    seq_idx: list[int] = []
    last_chunk: list[int] = []
    has_boundary: list[bool] = []
    boundary_rows: list[int] = []
    boundary_chunk_idx: list[int] = []
    store_src: list[int] = []
    store_seq: list[int] = []
    store_pos: list[int] = []
    offset = 0
    for i, (nc, q) in enumerate(zip(num_computed, query_lens)):
        n_pre = nc % chunk_size
        aug_len = n_pre + q
        buffered_seq.extend([i] * n_pre)
        buffered_pos.extend(range(n_pre))
        buffered_dst.extend(range(offset, offset + n_pre))
        step_dst.extend(range(offset + n_pre, offset + aug_len))
        first_chunk = len(cu_chunk)
        n_chunks = _cdiv(aug_len, chunk_size)
        cu_chunk.extend(offset + k * chunk_size for k in range(n_chunks))
        seq_idx.extend([i] * n_chunks)
        last_chunk.append(len(cu_chunk) - 1)
        has_boundary.append(nc - n_pre > 0)
        full = aug_len // chunk_size
        if full >= 1:
            boundary_rows.append(i)
            boundary_chunk_idx.append(first_chunk + full - 1)
        tail_len = aug_len % chunk_size
        if full == 0:
            # the buffer already holds positions [0, n_pre): append this step
            store_src.extend(range(offset + n_pre, offset + aug_len))
            store_seq.extend([i] * q)
            store_pos.extend(range(n_pre, aug_len))
        elif tail_len > 0:
            base = offset + aug_len - tail_len
            store_src.extend(range(base, base + tail_len))
            store_seq.extend([i] * tail_len)
            store_pos.extend(range(tail_len))
        offset += aug_len
        cu_seqlens.append(offset)
    cu_chunk.append(offset)

    def i64(v: list[int]) -> torch.Tensor:
        return async_tensor_h2d(v, dtype=torch.int64, device=device)

    def i32(v: list[int]) -> torch.Tensor:
        return async_tensor_h2d(v, dtype=torch.int32, device=device)

    return ExactReplayMetadata(
        num_aug_tokens=offset,
        buffered_seq=i64(buffered_seq),
        buffered_pos=i64(buffered_pos),
        buffered_dst=i64(buffered_dst),
        step_dst=i64(step_dst),
        cu_seqlens=i32(cu_seqlens),
        cu_chunk_seqlens=i32(cu_chunk),
        last_chunk_indices=i32(last_chunk),
        seq_idx=i32(seq_idx),
        has_boundary_state=async_tensor_h2d(
            has_boundary, dtype=torch.bool, device=device
        ),
        boundary_rows=i64(boundary_rows),
        boundary_chunk_idx=i64(boundary_chunk_idx),
        store_src=i64(store_src),
        store_seq=i64(store_seq),
        store_pos=i64(store_pos),
    )


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None
    # Exact-replay mode (None when disabled or when there are no such rows)
    exact_replay_p: ExactReplayMetadata | None = None
    exact_replay_d: ExactReplayMetadata | None = None


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

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
        self.exact_replay: bool = vllm_config.cache_config.mamba_exact_replay

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
            prev_last_scheduled_idx=kwargs.get("prev_last_scheduled_idx"),
            num_decode_draft_tokens_cpu=kwargs.get("num_decode_draft_tokens_cpu"),
        )

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = False
            if common.has_initial_states_p is not None:
                # Same condition as `has_initial_states_p`, but derived from CPU
                # data so it needs no D2H. `seq_lens_cpu_upper_bound` is precise
                # for prefill rows, which is all this slice covers.
                num_computed_tokens_p_cpu, _ = self._prefill_cpu_metadata(
                    common, common_attn_metadata
                )
                prep_initial_states = bool((num_computed_tokens_p_cpu > 0).any())

            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.chunk_size,
                    common,
                    common_attn_metadata,
                )
            )

        exact_replay_p = None
        exact_replay_d = None
        if self.exact_replay:
            device = common_attn_metadata.query_start_loc.device
            # Derive per-row computed-token counts from CPU metadata only:
            # `seq_lens_cpu_upper_bound` is exact for every row when async
            # scheduling and speculative decoding are off, which exact-replay
            # mode requires, and it avoids the D2H sync of the deprecated
            # `num_computed_tokens_cpu` property.
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu is None:
                raise ValueError(
                    "mamba_exact_replay needs CPU sequence lengths in the "
                    "attention metadata"
                )
            query_lens = torch.diff(common_attn_metadata.query_start_loc_cpu)
            num_computed_cpu = (seq_lens_cpu - query_lens).tolist()
            query_lens_cpu = query_lens.tolist()
            num_reqs = common.num_reqs
            # The metadata carries batch rows, not state slots, so the copy
            # that `update_block_table` hands to the other KV cache groups of a
            # hybrid model stays valid: each layer resolves its own slots from
            # its state indices when it runs the SSD step.
            if common.num_decodes > 0:
                exact_replay_d = build_exact_replay_metadata(
                    num_computed_cpu[: common.num_decodes],
                    query_lens_cpu[: common.num_decodes],
                    self.chunk_size,
                    device,
                )
            if common.num_prefills > 0:
                exact_replay_p = build_exact_replay_metadata(
                    num_computed_cpu[num_reqs - common.num_prefills : num_reqs],
                    query_lens_cpu[num_reqs - common.num_prefills : num_reqs],
                    self.chunk_size,
                    device,
                )

        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            exact_replay_p=exact_replay_p,
            exact_replay_d=exact_replay_d,
        )
