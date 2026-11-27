# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.linear_attention.mamba2.decode.base import SSUBuffers
from vllm.model_executor.layers.mamba.abstract import (
    MambaBase,
    get_kv_cache_checkpoint_alignment,
    get_mamba_num_speculative_blocks,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


@dataclass
class BaseMambaAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # The following tensors only contain prefill requests and will be None if
    # the batch has no prefill requests.
    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    state_indices_tensor_p: torch.Tensor | None

    # The following tensors are used for decode requests and
    # speculative decoding compatibility, and will be None if the batch
    # has no decode requests.
    state_indices_tensor_d: torch.Tensor | None
    query_start_loc_d: torch.Tensor | None  # shape: [num_decodes + 1,]

    # Number of accepted tokens for each spec sequence (for loading correct checkpoint)
    # Includes the bonus token (so minimum is 1)
    num_accepted_tokens: torch.Tensor | None  # shape: [batch,]

    # The following tensor is only used for prefix caching
    seq_lens: torch.Tensor

    # cu_chunk_seqlen_p is a tensor of shape (nchunks+1,) that contains, for
    # each chunk, its offsets into the varlen sequence dimension. It is defined
    # such that the i-th chunk contains tokens from cu_chunk_seqlen_p[i] to
    # cu_chunk_seqlen_p[i+1].
    cu_chunk_seqlen_p: torch.Tensor | None = None
    # last_chunk_indices_p is a tensor of shape (batch,) that contains the
    # index of the last chunk for every sequence in the (prefill) batch.
    last_chunk_indices_p: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None
    # ReplaySSM standard decode — Triton: per-row ring cursor and flush flag,
    # plus (decode_rows, ngroups, replayssm_buffer_len) fp32 scratch for
    # precomputed B·C products. The model runner fills the cursor and flag on
    # the GPU from per-slot ring counters before the forward (see
    # plan_replayssm_decode); prefix caching also reads the rows' sequence lengths.
    # All None when use_replayssm is disabled or ReplaySSM trackers are used.
    write_pos_d: torch.Tensor | None = None
    is_flush_d: torch.Tensor | None = None
    bc_pre_scratch: torch.Tensor | None = None
    replayssm_seq_lens_d: torch.Tensor | None = None
    replayssm_align_block_size: int = 0
    # ReplaySSM — FlashInfer checkpointing_ssu two-kernel scratch.
    replayssm_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    # Contiguous cache-slot indices shared by all ReplaySSM tracker layers.
    replayssm_state_indices_d: torch.Tensor | None = None

    def replayssm_step_slots(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The slots whose rings the step's prefills empty, and the decode
        rows' slots that Triton ReplaySSM plans."""
        prefill_slots = self.state_indices_tensor_p if self.num_prefills else None
        decode_slots = (
            self.state_indices_tensor_d
            if self.num_decodes and self.write_pos_d is not None
            else None
        )
        return prefill_slots, decode_slots


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    kv_cache_spec: MambaSpec
    metadata_cls: type[M]
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Will be disabled if speculative decoding is used
    supports_update_block_table: bool = True
    needs_causal_conv1d_metadata: bool = True

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.num_speculative_blocks = get_mamba_num_speculative_blocks(vllm_config)
        self.checkpoint_alignment = get_kv_cache_checkpoint_alignment(
            vllm_config, layer_names
        )

        # Enable speculative decoding support
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.num_spec_tokens: int = vllm_config.num_speculative_tokens
        self.use_spec_decode = self.num_spec_tokens > 0

        scheduler_config = vllm_config.scheduler_config
        self.decode_cudagraph_max_bs: int = scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.state_indices_tensor_d = torch.empty(
            (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
            dtype=torch.int32,
            device=device,
        )

        # For speculative decoding, we need to store the following buffers
        # for CUDA graph capture during decode
        if self.num_spec_tokens > 0:
            self.decode_num_accepted_tokens: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        # The kernels the layers selected; a cache group's layers share them.
        layers = get_layers_from_vllm_config(
            vllm_config,
            MambaBase,  # type: ignore[type-abstract]
            layer_names,
        ).values()
        self.ssu_kernel = _shared_kernel([layer.ssu_kernel for layer in layers])
        self.conv_prefill_kernel = _shared_kernel(
            [layer.conv_prefill_kernel for layer in layers]
        )
        # The decode kernel's persistent buffers, which it slices each step.
        self.ssu_buffers: SSUBuffers | None = None
        if self.ssu_kernel is not None:
            self.ssu_buffers = self.ssu_kernel.allocate_buffers(
                vllm_config, kv_cache_spec, self.decode_cudagraph_max_bs, device
            )

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            self.supports_update_block_table = False

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.max_query_len <= 1 + self.num_spec_tokens
            and m.num_reqs <= self.decode_cudagraph_max_bs
        ), (
            "Mamba only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only

        num_accepted_tokens = None
        if self.num_spec_tokens > 0:
            num_accepted_tokens = torch.diff(m.query_start_loc)

        return self.build(0, m, num_accepted_tokens=num_accepted_tokens)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M:
        """Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        return self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

    @staticmethod
    def _compute_chunk_metadata(
        chunk_size: int,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
        checkpoint_offsets_p: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Compute chunk-specific metadata for Mamba models.

        The code below carefully constructs the chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. Chunk boundaries fall on multiples of chunk_size in each
           sequence's absolute token positions, so the chunking does not
           depend on how chunked prefill split the sequence.
        Constraint (1) dramatically simplifies the mamba kernels.

        `checkpoint_offsets_p[i]` forces an extra chunk end that many tokens
        into row i's query (0 = none), because the SSD scan only materializes
        states at chunk ends. Returns its chunk index per row, or -1.
        """
        # TODO (tdoublep): This code could probably be optimized.
        cu_chunk_seqlen = []
        seq_idx = []
        last_chunk_indices = []
        checkpoint_chunk_indices = []
        seqlen_pos = 0

        # A checkpoint splits its request into two segments so that a chunk
        # ends exactly on it; each segment then chunks by the rules below.
        segments = []
        for req_idx in range(num_prefills):
            computed = num_computed_tokens_p_cpu[req_idx].item()
            tokens = (
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )
            ckpt = checkpoint_offsets_p[req_idx] if checkpoint_offsets_p else 0
            if ckpt:
                segments.append((req_idx, computed, ckpt, True))
            segments.append((req_idx, computed + ckpt, tokens - ckpt, False))
        checkpoint_chunk_indices = [-1] * num_prefills

        for req_idx, this_num_computed, this_new_tokens, is_ckpt_end in segments:
            # if computed tokens are not chunk-aligned, use the first
            # chunk to finish it off
            if this_num_computed % chunk_size != 0:
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                # how many tokens to finish the chunk?
                chunk_len = (
                    cdiv(this_num_computed, chunk_size) * chunk_size - this_num_computed
                )
                # we can only use at most this_new_tokens
                chunk_len = min(chunk_len, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            n_chunks = cdiv(this_new_tokens, chunk_size)
            for _ in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            if is_ckpt_end:
                checkpoint_chunk_indices[req_idx] = len(cu_chunk_seqlen) - 1
            else:
                last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices, checkpoint_chunk_indices

    def _prefill_cpu_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_reqs: int,
        num_prefills: int,
        num_decode_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill context lengths and query offsets, from CPU data only.

        `seq_lens_cpu_upper_bound` is precise for prefill rows in all modes
        (including async spec decode), so this avoids the D2H sync that
        `compute_num_computed_tokens().cpu()` would force.

        Returns (num_computed_tokens_p_cpu, query_start_loc_p_cpu).
        """
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        query_start_loc_p_cpu = (
            common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
            - num_decode_tokens
        )
        prefill_query_lens_cpu = query_start_loc_p_cpu[1:] - query_start_loc_p_cpu[:-1]
        num_computed_tokens_p_cpu = (
            seq_lens_cpu[num_reqs - num_prefills : num_reqs] - prefill_query_lens_cpu
        )
        return num_computed_tokens_p_cpu, query_start_loc_p_cpu

    @staticmethod
    def _build_chunk_metadata_tensors(
        chunk_size: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
        device: torch.device,
        checkpoint_offsets_p: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute chunk metadata and return as device tensors.
        Returns (cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p,
        checkpoint_chunk_idx_p). The last is None unless a row checkpoints;
        it has one entry per prefill row, holding the logical chunk that ends
        on the checkpoint, and 0 for rows that decline.
        """
        cu_chunk_seqlen, seq_idx, last_chunk_indices, ckpt_idxs = (
            BaseMambaAttentionMetadataBuilder._compute_chunk_metadata(
                chunk_size,
                query_start_loc_p_cpu.numel() - 1,
                num_computed_tokens_p_cpu,
                query_start_loc_p_cpu,
                checkpoint_offsets_p,
            )
        )

        # Build on pinned CPU and upload non-blocking to avoid the synchronous
        # H2D copy that `torch.as_tensor(list, device=cuda)` would force.
        cu_chunk_seqlen_p = async_tensor_h2d(
            cu_chunk_seqlen, dtype=torch.int32, device=device
        )
        seq_idx_p = async_tensor_h2d(seq_idx, dtype=torch.int32, device=device)
        last_chunk_indices_p = async_tensor_h2d(
            last_chunk_indices, dtype=torch.int32, device=device
        )
        ckpt_idx_p = (
            async_tensor_h2d(
                [max(idx, 0) for idx in ckpt_idxs], dtype=torch.int64, device=device
            )
            if any(idx >= 0 for idx in ckpt_idxs)
            else None
        )
        return cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p, ckpt_idx_p

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    ) -> M:
        """Compute metadata common to all Mamba-like backends."""
        num_reqs = common_attn_metadata.num_reqs

        # Treat multi-token queries as decode requests when
        # speculative decoding is enabled. Otherwise, use the
        # default decode threshold to prevent misclassification
        # of prefill queries as decode requests.
        decode_threshold = (
            self.reorder_batch_threshold if num_accepted_tokens is not None else 1
        )

        # FULL-CG dispatch is shape-based, so one-token prefills with
        # prior Mamba state can replay a decode graph while `is_prefilling`
        # is still true. Treat them as decode/update rows. This is required
        # for NIXL disagg's h(N-1)->N recompute path and for sporadic
        # final single-token prefill chunks that land in a `uniform` FULL-CG
        # batch. Relies on `reorder` putting short extends before pure prefills.
        is_prefilling = common_attn_metadata.is_prefilling
        assert is_prefilling is not None
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)

        # First prompt chunks have no prior Mamba state and must stay prefills.
        has_prior_state = seq_lens_cpu > query_lens_cpu
        stateful_prefill_rows = is_prefilling & has_prior_state

        # One-token prefills with prior state can use the decode/update path.
        prefill_to_decode = stateful_prefill_rows & (query_lens_cpu == 1)

        # The scheduler may pad a one-token remote prompt tail with placeholder
        # drafts to retain the uniform K+1 decode graph. This is a speculative
        # decode transaction even though the real token is still in the prompt:
        # the decode kernels keep h(N) in the running slot and h(N+i) in scratch
        # slots, so normal acceptance rollback remains valid. The prefill kernels
        # only return h(N+K) and cannot roll the placeholders back.
        if num_decode_draft_tokens_cpu is not None:
            padded_prompt_tail_rows = (
                stateful_prefill_rows
                & (num_decode_draft_tokens_cpu >= 0)
                & (query_lens_cpu == num_decode_draft_tokens_cpu + 1)
            )
            prefill_to_decode |= padded_prompt_tail_rows

        if torch.any(prefill_to_decode).item():
            # ReplaySSM appends these rows to the ring left empty by the
            # previous prefill chunk, like any other decode step.
            is_prefilling = is_prefilling.clone()
            is_prefilling[prefill_to_decode] = False
            common_attn_metadata = common_attn_metadata.replace(
                is_prefilling=is_prefilling
            )

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
                treat_short_extends_as_decodes=False,
            )
        )

        # Need flags to indicate if there are initial states
        has_initial_states_p = None
        query_start_loc_p = None
        query_start_loc_d = None

        state_indices_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.enable_prefix_caching,
            self.num_speculative_blocks,
        )

        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        state_indices_tensor_d = state_indices_tensor_d[:, : 1 + self.num_spec_tokens]
        state_indices_tensor_p = state_indices_tensor_p[:, 0]

        if num_decodes > 0 and self.use_spec_decode:
            query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
            if num_accepted_tokens is None:
                # Single-token prefill chunks can be reclassified as decodes before
                # speculative decoding has produced acceptance counts. Treat each
                # token as accepted so recurrent state and ReplaySSM trackers follow
                # the normal speculative-decode path.
                num_accepted_tokens = torch.diff(query_start_loc_d)
            else:
                num_accepted_tokens = num_accepted_tokens[:num_decodes]

        if num_prefills > 0:
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_p = (
                num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0
            )

        metadata = self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
            num_reqs=num_reqs,
            seq_lens=common_attn_metadata.seq_lens,
        )
        if num_prefills > 0:
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            if self.conv_prefill_kernel is not None:
                metadata = self.conv_prefill_kernel.prepare_metadata(
                    metadata, query_start_loc_p_cpu
                )
            elif self.needs_causal_conv1d_metadata:
                # ShortConv calls the conv op directly.
                nums_dict, batch_ptr, token_chunk_offset_ptr = (
                    compute_causal_conv1d_metadata(
                        query_start_loc_p_cpu,
                        device=common_attn_metadata.query_start_loc.device,
                    )
                )
                metadata = replace(
                    metadata,
                    nums_dict=nums_dict,
                    batch_ptr=batch_ptr,
                    token_chunk_offset_ptr=token_chunk_offset_ptr,
                )

        return self._update_metadata_for_cudagraph_capture(metadata)

    def _update_metadata_for_cudagraph_capture(
        self,
        metadata: M,
    ) -> M:
        """Update the metadata for cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        state_indices_tensor_d = metadata.state_indices_tensor_d
        query_start_loc_d = metadata.query_start_loc_d
        num_accepted_tokens = metadata.num_accepted_tokens
        num_decode_rows = metadata.num_decodes
        padded = (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        if padded:
            padded_bs = num_decode_rows = metadata.num_reqs
            self.state_indices_tensor_d[: metadata.num_decodes].copy_(
                state_indices_tensor_d, non_blocking=True
            )
            state_indices_tensor_d = self.state_indices_tensor_d[:padded_bs]
            state_indices_tensor_d[metadata.num_decodes :] = NULL_BLOCK_ID

            if self.use_spec_decode and num_accepted_tokens is not None:
                assert query_start_loc_d is not None
                query_start_loc_d = query_start_loc_d[: padded_bs + 1]
                self.decode_num_accepted_tokens[: metadata.num_decodes].copy_(
                    num_accepted_tokens, non_blocking=True
                )
                num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
                num_accepted_tokens[metadata.num_decodes :] = (
                    1  # pad with 1st slot index
                )

        metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            query_start_loc_d=query_start_loc_d,
            num_accepted_tokens=num_accepted_tokens,
        )
        if self.ssu_kernel is not None and num_decode_rows > 0:
            metadata = self.ssu_kernel.prepare_metadata(
                self.ssu_buffers, metadata, num_decode_rows, padded
            )
        return metadata

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        state_indices_tensor = mamba_get_block_table_tensor(
            blk_table,
            metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.enable_prefix_caching,
            self.num_speculative_blocks,
        )
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        assert (
            metadata.num_prefills + metadata.num_decodes
            == state_indices_tensor.shape[0]
        ), (
            "Mismatch in number of requests when updating block table."
            f" Expected {metadata.num_prefills + metadata.num_decodes}, "
            f"got {state_indices_tensor.shape[0]}."
        )

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [metadata.num_decodes, metadata.num_prefills],
            dim=0,
        )
        state_indices_tensor_d = state_indices_tensor_d[:, : 1 + self.num_spec_tokens]
        state_indices_tensor_p = state_indices_tensor_p[:, 0]

        new_metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
        )

        return self._update_metadata_for_cudagraph_capture(new_metadata)


K = TypeVar("K", bound=type)


def _shared_kernel(kernels: list[K | None]) -> K | None:
    """The one kernel the layers selected, if any."""
    selected = {kernel for kernel in kernels if kernel is not None}
    if len(selected) > 1:
        names = sorted(kernel.__name__ for kernel in selected)
        raise ValueError(
            f"A KV cache group's layers selected different kernels {names}"
        )
    return next(iter(selected), None)
