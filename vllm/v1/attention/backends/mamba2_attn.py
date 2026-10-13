# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import CAUSAL_CONV1D_BLOCK_M
from vllm.utils.math_utils import cdiv, round_up
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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

# Static launches run FlashInfer's scan with one sequence slot per this many
# padded tokens, and at least
# _STATIC_FLASHINFER_MIN_SEQS slots (see mamba2_static_flashinfer_seqs).
_STATIC_FLASHINFER_TOKENS_PER_SEQ = 512
_STATIC_FLASHINFER_MIN_SEQS = 8
# FlashInfer's Cake SSD scan computes whole tiles of this many tokens
# (SSDCombined: "the Cake programs tile 128 tokens internally",
# flashinfer-ai/flashinfer#6100).
_FLASHINFER_SSD_CHUNK = 128


def mamba2_mixer_in_piecewise_cudagraphs(vllm_config: VllmConfig) -> bool:
    """Whether the Mamba2 mixer runs inside the piecewise CUDA graphs, which
    is the case when compilation_config.splitting_ops leaves out
    vllm::mamba_mixer2. The mixer then takes its static-shape path."""
    compilation_config = vllm_config.compilation_config
    splitting_ops = compilation_config.splitting_ops
    return (
        compilation_config.mode == CompilationMode.VLLM_COMPILE
        and compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        and splitting_ops is not None
        and "vllm::mamba_mixer2" not in splitting_ops
    )


def validate_mamba2_mixer_in_piecewise_cudagraphs(vllm_config: VllmConfig) -> None:
    """Raise if the Mamba2 mixer is kept in the piecewise CUDA graphs without
    what its static-shape path needs."""
    cache_config = vllm_config.cache_config
    if not vllm_config.use_v2_model_runner:
        requirement = "the V2 model runner"
    elif not (
        cache_config.use_replayssm
        and vllm_config.mamba_config.backend == MambaBackendEnum.FLASHINFER
    ):
        requirement = "--use-replayssm --mamba-backend flashinfer"
    elif (
        cache_config.mamba_cache_mode != "none"
        or cache_config.enable_mamba_shared_prefix_checkpoint
    ):
        requirement = "--mamba-cache-mode none without prefill checkpoints"
    elif vllm_config.mamba_config.prefill_backend != MambaBackendEnum.FLASHINFER:
        requirement = "--mamba-prefill-backend flashinfer"
    else:
        return
    raise ValueError(
        "compilation_config.splitting_ops leaves out vllm::mamba_mixer2, which "
        f"captures the Mamba2 mixer in the piecewise CUDA graphs; this requires "
        f"{requirement}."
    )


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


def mamba2_static_sizes(
    num_tokens: int, max_num_seqs: int, max_decode_rows: int
) -> tuple[int, int, int]:
    """Static launch sizes of a Mamba2 mixer for `num_tokens` padded tokens:
    (prefill sequence slots, causal_conv1d_fn programs, decode rows). Each
    sequence adds at most one partial conv1d block."""
    num_seqs = min(max_num_seqs, num_tokens)
    num_conv_programs = cdiv(num_tokens, CAUSAL_CONV1D_BLOCK_M) + num_seqs
    num_decode_rows = min(max_decode_rows, num_tokens)
    return num_seqs, num_conv_programs, num_decode_rows


def mamba2_static_flashinfer_seqs(num_tokens: int, max_num_seqs: int) -> int:
    """Prefill sequence slots of FlashInfer's static scan for `num_tokens`
    padded tokens. Its kernel runs one work item per (slot, head), empty
    slots included, so the slots follow the step size: enough for prompts
    of about _STATIC_FLASHINFER_TOKENS_PER_SEQ tokens. Steps with more
    prefills run eagerly (the model runner checks this bound on the
    unpadded token count, which can only lower it)."""
    return min(
        max_num_seqs,
        num_tokens,
        max(
            _STATIC_FLASHINFER_MIN_SEQS,
            cdiv(num_tokens, _STATIC_FLASHINFER_TOKENS_PER_SEQ),
        ),
    )


@dataclass
class Mamba2StaticMetadata:
    """Fixed-shape metadata of a Mamba2 mixer captured in a piecewise CUDA
    graph (splitting_ops leave out vllm::mamba_mixer2).

    Shapes depend only on the padded token count, so one graph per capture
    size serves every mix of prefills and decodes. All tensors are views of
    persistent buffers that the builder rewrites before each step. Token
    offsets index the padded batch [decode tokens | prefill tokens | padding].
    Padding prefill sequences have state index PAD_SLOT_ID, padding decode
    rows NULL_BLOCK_ID and padding conv1d programs PAD_SLOT_ID.
    """

    num_tokens: int
    # Prefill sequences.
    query_start_loc_p: torch.Tensor  # (num_seqs + 1,) int32
    state_indices_p: torch.Tensor  # (num_seqs,) int32
    has_initial_states_p: torch.Tensor  # (num_seqs,) int32
    # FlashInfer scan of the whole batch, see
    # mamba_chunk_scan_flashinfer_static: sequence offsets
    # [0, starts of prefills 2..n, num_tokens, ...] over its
    # mamba2_static_flashinfer_seqs slots, and the prefill tokens [start, end).
    cu_seqlens_fi: torch.Tensor  # (slots + 1,) int32
    prefill_bounds_fi: torch.Tensor  # (2,) int32
    num_seqs_fi: torch.Tensor  # (1,) int32, the prefills it scans
    # causal_conv1d_fn launch metadata (it reads these attribute names).
    nums_dict: dict[int, dict[str, Any]]
    batch_ptr: torch.Tensor  # (num_conv_programs,) int32
    token_chunk_offset_ptr: torch.Tensor  # (num_conv_programs,) int32
    # Decode rows.
    query_start_loc_d: torch.Tensor  # (num_decode_rows + 1,) int32
    state_indices_d: torch.Tensor  # (num_decode_rows, 1 + num_spec) int32
    replayssm_state_indices_d: torch.Tensor | None  # (num_decode_rows,) int32
    num_accepted_tokens: torch.Tensor | None  # (num_decode_rows,) int32
    replayssm_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None

    # Internal prefill checkpoints, one entry per prefill row. The chunk
    # index selects the varlen_states row holding the checkpoint state.
    checkpoint_chunk_idx: torch.Tensor | None = None
    checkpoint_meta: MambaPrefillCheckpointMetadata | None = None

    # Set when the mixer may run inside a piecewise CUDA graph.
    static: Mamba2StaticMetadata | None = None


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

        # Persistent buffers for a mixer captured in piecewise CUDA graphs.
        self.static_max_tokens = 0
        if mamba2_mixer_in_piecewise_cudagraphs(vllm_config):
            self.static_max_tokens = (
                self.compilation_config.max_cudagraph_capture_size
                or vllm_config.scheduler_config.max_num_batched_tokens
            )
            num_seqs, num_conv_programs, num_decode_rows = mamba2_static_sizes(
                self.static_max_tokens,
                vllm_config.scheduler_config.max_num_seqs,
                self.decode_cudagraph_max_bs,
            )
            # Host-computed metadata goes up in one copy per step.
            segment_sizes = {
                "query_start_loc_p": num_seqs + 1,
                "cu_seqlens_fi": num_seqs + 1,
                "prefill_bounds_fi": 2,
                "num_seqs_fi": 1,
                "has_initial_states_p": num_seqs,
                "batch_ptr": num_conv_programs,
                "token_chunk_offset_ptr": num_conv_programs,
                "query_start_loc_d": num_decode_rows + 1,
            }
            offsets = itertools.accumulate(segment_sizes.values(), initial=0)
            self._static_offsets = dict(zip(segment_sizes, offsets))
            self._static_host_size = sum(segment_sizes.values())
            self._static_device = torch.empty(
                self._static_host_size, dtype=torch.int32, device=device
            )
            # Per KV cache group, staged by build() and update_block_table().
            self._static_state_indices_p = torch.empty(
                num_seqs, dtype=torch.int32, device=device
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

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        checkpoint_chunk_idx = None
        checkpoint_meta = None
        prep_initial_states = False

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = False
            if common.has_initial_states_p is not None:
                # Same condition as `has_initial_states_p`, but derived from CPU
                # data so it needs no D2H. `seq_lens_cpu_upper_bound` is precise
                # for prefill rows, which is all this slice covers.
                num_computed_tokens_p_cpu, _ = self._prefill_cpu_metadata(
                    common_attn_metadata,
                    common.num_reqs,
                    common.num_prefills,
                    common.num_decode_tokens,
                )
                prep_initial_states = bool((num_computed_tokens_p_cpu > 0).any())

            checkpoint_offsets_p = None
            first = common.num_reqs - common.num_prefills
            checkpoint = self.checkpoint_builder.build(
                common_attn_metadata,
                list(range(first, common.num_reqs)),
            )
            if checkpoint is not None:
                # The host offsets place a chunk boundary on the checkpoint.
                checkpoint_offsets_p = checkpoint.offsets
                checkpoint_meta = checkpoint

            (
                cu_chunk_seqlen_p,
                seq_idx_p,
                last_chunk_indices_p,
                checkpoint_chunk_idx,
            ) = self._build_chunk_metadata_tensors(
                self.chunk_size,
                common,
                common_attn_metadata,
                checkpoint_offsets_p,
            )

        metadata = replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            checkpoint_chunk_idx=checkpoint_chunk_idx,
            checkpoint_meta=checkpoint_meta,
        )
        # Only steps that run the piecewise graphs read the static metadata;
        # FULL decode graphs and eager steps take the mixer's dynamic path.
        if (
            self.static_max_tokens
            and kwargs.get("cudagraph_mode") == CUDAGraphMode.PIECEWISE
            and checkpoint_meta is None
        ):
            metadata = replace(
                metadata,
                static=self._build_static(metadata, common_attn_metadata),
            )
        return metadata

    def _build_static(
        self,
        metadata: Mamba2AttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> Mamba2StaticMetadata | None:
        """Fill the persistent static-shape metadata for this step's padded
        token count (the slot mapping's length). None if the step is larger
        than any captured graph and so always runs eagerly."""
        num_tokens = common_attn_metadata.slot_mapping.shape[0]
        if num_tokens > self.static_max_tokens:
            return None
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        num_seqs, num_conv_programs, num_decode_rows = mamba2_static_sizes(
            num_tokens, max_num_seqs, self.decode_cudagraph_max_bs
        )
        num_fi_seqs = mamba2_static_flashinfer_seqs(num_tokens, max_num_seqs)
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        num_decode_tokens = metadata.num_decode_tokens
        num_actual_tokens = num_decode_tokens + metadata.num_prefill_tokens
        assert num_prefills <= num_seqs and num_decodes <= num_decode_rows
        query_start_loc = common_attn_metadata.query_start_loc_cpu.numpy()
        offsets = self._static_offsets
        host = np.empty(self._static_host_size, dtype=np.int32)

        def segment(name: str, size: int) -> np.ndarray:
            return host[offsets[name] : offsets[name] + size]

        # Prefill sequences, in absolute token offsets.
        qsl_p = segment("query_start_loc_p", num_seqs + 1)
        qsl_p[: num_prefills + 1] = query_start_loc[
            num_decodes : num_decodes + num_prefills + 1
        ]
        qsl_p[num_prefills + 1 :] = num_actual_tokens
        # One FlashInfer sequence per prefill: the first also covers the decode
        # rows (identity rows of the scan); the last ends at the first 128-token
        # boundary after the real tokens, and the padding after it belongs to
        # no sequence, so the scan skips it. The empty slots start at that
        # boundary: an empty sequence starting inside a chunk breaks the final
        # state of the sequence before it. The model runner runs steps with
        # more prefills than the slots eagerly, so only dummy batches (graph
        # capture and warmup), which are never replayed, fold the surplus
        # prefills into the last slot.
        num_fi_prefills = min(num_prefills, num_fi_seqs)
        cu_fi = segment("cu_seqlens_fi", num_fi_seqs + 1)
        cu_fi[0] = 0
        cu_fi[1:num_fi_prefills] = qsl_p[1:num_fi_prefills]
        cu_fi[max(num_fi_prefills, 1) :] = min(
            num_tokens, round_up(num_actual_tokens, _FLASHINFER_SSD_CHUNK)
        )
        bounds = segment("prefill_bounds_fi", 2)
        bounds[0] = qsl_p[0]
        bounds[1] = num_actual_tokens
        segment("num_seqs_fi", 1)[0] = num_fi_prefills
        has_initial_states = segment("has_initial_states_p", num_seqs)
        has_initial_states[:] = 0
        if num_prefills > 0:
            num_computed_tokens_p_cpu, _ = self._prefill_cpu_metadata(
                common_attn_metadata,
                metadata.num_reqs,
                num_prefills,
                num_decode_tokens,
            )
            has_initial_states[:num_prefills] = num_computed_tokens_p_cpu.numpy() > 0
        # causal_conv1d_fn: one program per CAUSAL_CONV1D_BLOCK_M tokens of a prefill.
        seqlens = np.diff(qsl_p[: num_prefills + 1])
        programs = -(-seqlens // CAUSAL_CONV1D_BLOCK_M)
        num_real_programs = int(programs.sum())
        assert num_real_programs <= num_conv_programs
        batch_ptr = segment("batch_ptr", num_conv_programs)
        batch_ptr[:num_real_programs] = np.repeat(np.arange(num_prefills), programs)
        batch_ptr[num_real_programs:] = PAD_SLOT_ID
        chunk_offsets = segment("token_chunk_offset_ptr", num_conv_programs)
        chunk_offsets[:num_real_programs] = np.arange(num_real_programs) - np.repeat(
            np.cumsum(programs) - programs, programs
        )
        chunk_offsets[num_real_programs:] = PAD_SLOT_ID
        # Decode rows.
        qsl_d = segment("query_start_loc_d", num_decode_rows + 1)
        qsl_d[: num_decodes + 1] = query_start_loc[: num_decodes + 1]
        qsl_d[num_decodes + 1 :] = num_decode_tokens
        self._static_device.copy_(
            torch.from_numpy(host).pin_memory(), non_blocking=True
        )

        def view(name: str, size: int) -> torch.Tensor:
            return self._static_device[offsets[name] : offsets[name] + size]

        num_accepted_tokens = None
        if self.use_spec_decode:
            num_accepted_tokens = self.decode_num_accepted_tokens[:num_decode_rows]
            if (
                metadata.num_accepted_tokens is not None
                and metadata.num_accepted_tokens.data_ptr()
                != num_accepted_tokens.data_ptr()
            ):
                num_accepted_tokens[:num_decodes].copy_(
                    metadata.num_accepted_tokens[:num_decodes], non_blocking=True
                )
            num_accepted_tokens[num_decodes:] = 1
        replayssm_scratch = None
        if self.decode_replayssm_scratch is not None:
            cb_scaled, cumAdt_vec, cb_old = self.decode_replayssm_scratch
            replayssm_scratch = (
                cb_scaled[:num_decode_rows],
                cumAdt_vec[:num_decode_rows],
                cb_old[:num_decode_rows],
            )
        batch_ptr_view = view("batch_ptr", num_conv_programs)
        chunk_offsets_view = view("token_chunk_offset_ptr", num_conv_programs)
        static = Mamba2StaticMetadata(
            num_tokens=num_tokens,
            query_start_loc_p=view("query_start_loc_p", num_seqs + 1),
            state_indices_p=self._static_state_indices_p[:num_seqs],
            has_initial_states_p=view("has_initial_states_p", num_seqs),
            cu_seqlens_fi=view("cu_seqlens_fi", num_fi_seqs + 1),
            prefill_bounds_fi=view("prefill_bounds_fi", 2),
            num_seqs_fi=view("num_seqs_fi", 1),
            nums_dict={
                CAUSAL_CONV1D_BLOCK_M: {
                    "tot": num_conv_programs,
                    "batch_ptr": batch_ptr_view,
                    "token_chunk_offset_ptr": chunk_offsets_view,
                    "mlist": None,
                    "mlist_len": 0,
                    "offsetlist": None,
                }
            },
            batch_ptr=batch_ptr_view,
            token_chunk_offset_ptr=chunk_offsets_view,
            query_start_loc_d=view("query_start_loc_d", num_decode_rows + 1),
            state_indices_d=self.state_indices_tensor_d[:num_decode_rows],
            replayssm_state_indices_d=None,
            num_accepted_tokens=num_accepted_tokens,
            replayssm_scratch=replayssm_scratch,
        )
        return self._stage_static_state_indices(
            static,
            metadata,
            metadata.state_indices_tensor_d,
            metadata.state_indices_tensor_p,
        )

    def _stage_static_state_indices(
        self,
        static: Mamba2StaticMetadata,
        metadata: Mamba2AttentionMetadata,
        state_indices_tensor_d: torch.Tensor | None,
        state_indices_tensor_p: torch.Tensor | None,
    ) -> Mamba2StaticMetadata:
        """Stage this KV cache group's state indices into its own persistent
        static buffers (the other static metadata is batch-level)."""
        num_prefills = metadata.num_prefills
        state_indices_p = self._static_state_indices_p[
            : static.state_indices_p.shape[0]
        ]
        if num_prefills > 0:
            assert state_indices_tensor_p is not None
            state_indices_p[:num_prefills].copy_(
                state_indices_tensor_p, non_blocking=True
            )
        state_indices_p[num_prefills:] = PAD_SLOT_ID
        num_decode_rows = static.state_indices_d.shape[0]
        if state_indices_tensor_d is None:
            state_indices_tensor_d = self.state_indices_tensor_d[:0]
        state_indices_d, replayssm_state_indices_d = self._stage_static_decode_indices(
            metadata, state_indices_tensor_d[: metadata.num_decodes], num_decode_rows
        )
        return replace(
            static,
            state_indices_p=state_indices_p,
            state_indices_d=state_indices_d,
            replayssm_state_indices_d=replayssm_state_indices_d,
        )

    def update_block_table(
        self,
        metadata: Mamba2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Mamba2AttentionMetadata:
        new_metadata = super().update_block_table(metadata, blk_table, slot_mapping)
        if metadata.static is not None:
            new_metadata = replace(
                new_metadata,
                static=self._stage_static_state_indices(
                    metadata.static,
                    new_metadata,
                    new_metadata.state_indices_tensor_d,
                    new_metadata.state_indices_tensor_p,
                ),
            )
        if metadata.checkpoint_meta is None:
            return new_metadata
        # Checkpoint destinations are block-table entries, so each group
        # re-gathers its own rather than writing into the source group's.
        return replace(
            new_metadata,
            checkpoint_meta=metadata.checkpoint_meta.regather_state_indices(blk_table),
        )
