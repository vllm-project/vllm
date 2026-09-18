# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.exact_replay import (
    ExactReplayMetadata,
    build_exact_replay_metadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
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
    def supports_batch_invariance(cls) -> bool:
        # In batch-invariant mode every SSD call starts at the sequence's last
        # chunk boundary (exact replay, see exact_replay.py), so prefill,
        # chunked prefill and decode produce the same bits.
        return True

    @classmethod
    def check_batch_invariant_config(cls, vllm_config: VllmConfig) -> None:
        """Reject engine settings the batch-invariant SSD path does not support.

        Raises:
            ValueError: for an unsupported setting, with the flag to change.

        """
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        prefix = "VLLM_BATCH_INVARIANT=1 with Mamba2 layers"
        if cache_config.use_replayssm:
            raise ValueError(f"{prefix} is not supported together with --use-replayssm")
        if cache_config.mamba_cache_mode != "none":
            raise ValueError(
                f"{prefix} does not support prefix caching yet; pass "
                "--no-enable-prefix-caching"
            )
        if vllm_config.num_speculative_tokens > 0:
            raise ValueError(f"{prefix} does not support speculative decoding")
        if vllm_config.mamba_config.backend != MambaBackendEnum.TRITON:
            raise ValueError(f"{prefix} requires --mamba-backend triton")
        if parallel_config.pipeline_parallel_size > 1:
            raise ValueError(f"{prefix} currently requires PP=1")
        if parallel_config.use_ubatching:
            raise ValueError(
                f"{prefix} does not support micro-batching (--enable-dbo or "
                "--ubatch-size > 1)"
            )
        if vllm_config.kv_transfer_config is not None:
            raise ValueError(f"{prefix} does not support KV connectors")


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
        self.exact_replay: bool = envs.VLLM_BATCH_INVARIANT

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        if envs.VLLM_BATCH_INVARIANT:
            # The replayed SSD step re-feeds each row's buffered partial chunk,
            # so its shapes are data dependent and cannot be captured.
            return AttentionCGSupport.NEVER
        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

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
            # `seq_lens_cpu_upper_bound` is exact for every row without
            # speculative decoding, which this mode rejects, and it avoids the
            # D2H sync of the deprecated `num_computed_tokens_cpu` property.
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu is None:
                raise ValueError(
                    "VLLM_BATCH_INVARIANT=1 needs CPU sequence lengths in the "
                    "attention metadata for Mamba2 layers"
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
