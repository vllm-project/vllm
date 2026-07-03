# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass, replace
from typing import Any, ClassVar, TypeVar

import torch

from vllm.config import VllmConfig
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
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

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
    num_computed_tokens_p: torch.Tensor | None
    state_indices_tensor_p: torch.Tensor | None

    # The following tensors are used for decode requests and
    # speculative decoding compatibility, and will be None if the batch
    # has no decode requests.
    state_indices_tensor_d: torch.Tensor | None
    query_start_loc_d: torch.Tensor | None  # shape: [num_decodes + 1,]

    # Number of accepted tokens for each spec sequence (for loading correct checkpoint)
    # Includes the bonus token (so minimum is 1)
    num_accepted_tokens: torch.Tensor | None  # shape: [batch,]

    # The following tensors are only used for prefix caching in all mode and
    # are None if disabled
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None
    block_idx_last_scheduled_token_prev_step: torch.Tensor | None

    # The following tensor is only used for prefix caching in align mode
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
    write_pos_d: torch.Tensor | None = None
    is_flush_d: torch.Tensor | None = None
    # Shared per-step scratch for the output-only (output_only) variant:
    # (decode_rows, ngroups, max_cache_len) fp32. None for the recurrent
    # variant or when the cached kernel is disabled.
    bc_pre_scratch: torch.Tensor | None = None
    # cached-SPEC (hybrid) cursors: persistent, block-keyed (full (num_blocks,)
    # buffers indexed by physical SSM block id), shared across all Mamba2 layers
    # and advanced once per step by the commit. spec_bc_pre_scratch is the
    # per-step (decode_rows, ngroups, max_cache_len, block_spec) fp32 scratch.
    # All None unless the cached-spec kernel is enabled.
    spec_write_pos_d: torch.Tensor | None = None
    spec_post_origin_d: torch.Tensor | None = None
    spec_is_flush_d: torch.Tensor | None = None
    spec_bc_pre_scratch: torch.Tensor | None = None


class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    metadata_cls: type[M]
    reorder_batch_threshold: int = 1
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    # Will be disabled if speculative decoding is used
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable speculative decoding support
        self.speculative_config = vllm_config.speculative_config
        self.compilation_config = vllm_config.compilation_config
        self.num_spec_tokens: int = vllm_config.num_speculative_tokens
        self.use_spec_decode = self.num_spec_tokens > 0
        self.use_cached_kernel = vllm_config.cache_config.use_replayssm
        self.max_cache_len = vllm_config.cache_config.replayssm_buffer_len
        self.use_cache_spec_kernel = (
            vllm_config.cache_config.use_replayssm_spec
        )
        self.max_spec_len = 1 + self.num_spec_tokens
        # L = B + max_spec_len history window; physical pow2 ring = next_pow2(L).
        self.spec_flush_threshold = self.max_cache_len + self.max_spec_len
        self.spec_cache_buf_len = 1 << (self.spec_flush_threshold - 1).bit_length()

        assert isinstance(kv_cache_spec, MambaSpec)
        scheduler_config = vllm_config.scheduler_config
        self.decode_cudagraph_max_bs: int = scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            max_num_blocks = (
                cdiv(
                    self.vllm_config.model_config.max_model_len,
                    kv_cache_spec.block_size,
                )
                + kv_cache_spec.num_speculative_blocks
            )
            # TODO: reduce this size as needed for decode-only cudagraph capture
            self.state_indices_tensor_d: torch.Tensor = torch.empty(
                (
                    self.decode_cudagraph_max_bs,
                    max_num_blocks,
                ),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            if self.use_spec_decode:
                self.block_idx_last_scheduled_token_prev_step: torch.Tensor = (
                    torch.empty(
                        (self.decode_cudagraph_max_bs,),
                        dtype=torch.int32,
                        device=device,
                    )
                )
        else:
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
        if self.use_cached_kernel:
            self.decode_write_pos_d: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.decode_is_flush_d: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int8,
                device=device,
            )
            self.cached_kernel_variant: str = (
                vllm_config.cache_config.replayssm_route
            )
            if self.cached_kernel_variant == "output_only":
                # B_cache shape = (ngroups, max_cache_len, dstate). Index in
                # MambaSpec.shapes is (conv_state, ssm_state, x_cache,
                # dt_cache, B_cache) when the cached kernel is enabled.
                if len(kv_cache_spec.shapes) < 5:
                    raise ValueError(
                        "output-only variant requires the 5-tensor Mamba2 "
                        "page (conv, ssm, x_cache, dt_cache, B_cache)"
                    )
                bc_ngroups = kv_cache_spec.shapes[4][0]
                self.decode_bc_pre_scratch: torch.Tensor = torch.empty(
                    (
                        self.decode_cudagraph_max_bs,
                        bc_ngroups,
                        self.max_cache_len,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                self.decode_bc_pre_scratch = None
        else:
            self.cached_kernel_variant = "state_and_output"
            self.decode_bc_pre_scratch = None

        # cached-SPEC (hybrid): persistent block-keyed cursors are allocated
        # lazily on first build() (they need num_gpu_blocks). The per-step CB
        # scratch is fixed-address (CUDA-graph safe), sized here. ngroups is
        # derived from the page shapes (conv, ssm, post_conv_cache, dt_cache).
        self.spec_write_pos: torch.Tensor | None = None
        self.spec_post_origin: torch.Tensor | None = None
        self.spec_is_flush: torch.Tensor | None = None
        self.decode_spec_bc_pre: torch.Tensor | None = None
        if self.use_cache_spec_kernel:
            if len(kv_cache_spec.shapes) < 4:
                raise ValueError(
                    "cached-spec kernel requires the 4-tensor hybrid Mamba2 page "
                    "(conv, ssm, post_conv_cache, dt_cache)"
                )
            local_nheads, head_dim, dstate = kv_cache_spec.shapes[1]
            conv_dim_local = kv_cache_spec.shapes[2][1]
            d_inner_local = local_nheads * head_dim
            ngroups_local = (conv_dim_local - d_inner_local) // (2 * dstate)
            block_spec = 1 << (max(1, self.max_spec_len) - 1).bit_length()
            # This is a PER-STEP scratch consumed by the scatter on every decode
            # step (eager AND cudagraph), indexed by pid_b in [0, num_decodes).
            # It must therefore cover the max decode batch (max_num_seqs), NOT
            # decode_cudagraph_max_bs -- the latter is 0 under enforce_eager
            # (max_cudagraph_capture_size=0), which would make this scratch empty
            # and the scatter write bc_pre[pid_b] out of bounds (IMA). Sizing by
            # max_num_seqs is CUDA-graph safe: the captured [:num_decodes] slice
            # shares the same (offset-0) base pointer and row strides.
            spec_scratch_bs = max(
                self.decode_cudagraph_max_bs, scheduler_config.max_num_seqs
            )
            self.decode_spec_bc_pre = torch.empty(
                (
                    spec_scratch_bs,
                    ngroups_local,
                    self.spec_cache_buf_len,
                    block_spec,
                ),
                dtype=torch.float32,
                device=device,
            )

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            self.supports_update_block_table = False

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
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

        prev_last_scheduled_idx = None
        if (
            self.use_spec_decode
            and self.vllm_config.cache_config.mamba_cache_mode == "all"
        ):
            prev_last_scheduled_idx = torch.zeros(
                (m.num_reqs,),
                dtype=torch.int32,
                device=m.query_start_loc.device,
            )

        return self.build(
            0,
            m,
            num_accepted_tokens=num_accepted_tokens,
            prev_last_scheduled_idx=prev_last_scheduled_idx,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M:
        """
        Default build implementation for Mamba-like attention backends.
        Subclasses (e.g., Mamba2) can override to add additional metadata.
        """
        return self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            prev_last_scheduled_idx=prev_last_scheduled_idx,
        )

    def _compute_chunk_metadata(
        self,
        chunk_size: int,
        num_prefills: int,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute chunk-specific metadata for Mamba models.

        The code below carefully constructs the chunks such that:
        1. Chunks contain tokens from a *single* sequence only.
        2. For every sequence, we are guaranteed that we can
           retrieve the mamba state *every* chunk_size tokens.
        Constraint (1) dramatically simplifies the mamba kernels.
        Constraint (2) dramatically simplifies the implementation
        of prefix caching for mamba (wip). We need to take care
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
            for chunk in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(seqlen_pos)
                chunk_len = min(chunk_size, this_new_tokens)
                seqlen_pos += chunk_len
                this_new_tokens -= chunk_len

            assert this_new_tokens == 0
            last_chunk_indices.append(len(cu_chunk_seqlen) - 1)

        cu_chunk_seqlen.append(seqlen_pos)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def _build_chunk_metadata_tensors(
        self,
        chunk_size: int,
        common: M,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute chunk metadata and return as device tensors.
        Returns (cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p).
        """
        num_reqs = common.num_reqs
        num_prefills = common.num_prefills
        num_decode_tokens = common.num_decode_tokens

        # Derive prefill context lengths from CPU data only.
        # `seq_lens_cpu_upper_bound` is precise for prefill rows in all modes
        # (including async spec decode), so this avoids the D2H sync that
        # `compute_num_computed_tokens().cpu()` would force.
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

        cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
            chunk_size,
            num_prefills,
            num_computed_tokens_p_cpu,
            query_start_loc_p_cpu,
        )

        device = common_attn_metadata.query_start_loc.device
        # Build on pinned CPU and upload non-blocking to avoid the synchronous
        # H2D copy that `torch.as_tensor(list, device=cuda)` would force.
        cu_chunk_seqlen_p = async_tensor_h2d(
            cu_chunk_seqlen, dtype=torch.int32, device=device
        )
        seq_idx_p = async_tensor_h2d(seq_idx, dtype=torch.int32, device=device)
        last_chunk_indices_p = async_tensor_h2d(
            last_chunk_indices, dtype=torch.int32, device=device
        )
        return cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mamba_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        # Block index of the last computed token
        block_idx_last_computed_token = cdiv(num_computed_tokens, mamba_block_size) - 1
        # which is <= block index for the first scheduled token
        block_idx_first_scheduled_token = (
            cdiv(num_computed_tokens + 1, mamba_block_size) - 1
        )
        # which is <= block index of the last scheduled token
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, mamba_block_size) - 1
        )
        # -1 in case it's non-computed and causes later issues with indexing
        block_idx_last_computed_token = torch.clamp(
            block_idx_last_computed_token, min=0
        )
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = torch.clamp(
            block_idx_last_scheduled_token, min=0
        )

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        prev_last_scheduled_idx: torch.Tensor | None = None,
    ) -> M:
        """
        Compute metadata common to both Mamba1 and Mamba2.
        """
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
        single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)
        # First-token prefills have no prior Mamba state and must stay prefills.
        has_prior_state = seq_lens_cpu > 1
        prefill_to_decode = single_token_prefill_rows & has_prior_state
        if torch.any(prefill_to_decode).item():
            if self.use_cached_kernel and metadata.num_decodes > 0:
                raise ValueError(
                    "--use-replayssm does not support single-token "
                    "prefill rows replayed through the decode path"
                )
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
        num_computed_tokens = None
        num_computed_tokens_p = None

        # for prefix caching
        block_idx_first_scheduled_token = None
        block_idx_first_scheduled_token_p = None
        block_idx_last_computed_token = None
        block_idx_last_scheduled_token = None
        block_idx_last_scheduled_token_prev_step = None

        # for causal_conv1d
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        write_pos_d = None
        is_flush_d = None

        if self.vllm_config.cache_config.mamba_cache_mode == "all":
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            # Return a tensor of shape (#requests, #max blocks)
            state_indices_tensor = common_attn_metadata.block_table_tensor
            # Additional cache-related variables:
            mamba_block_size = self.kv_cache_spec.block_size
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, mamba_block_size
            )
            if self.use_spec_decode and prev_last_scheduled_idx is not None:
                fallback = torch.clamp(
                    (num_computed_tokens - 1) // mamba_block_size, min=0
                )
                block_idx_last_scheduled_token_prev_step = torch.where(
                    prev_last_scheduled_idx >= 0,
                    prev_last_scheduled_idx,
                    fallback,
                )
        else:
            state_indices_tensor = mamba_get_block_table_tensor(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )

        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [num_decodes, num_prefills],
            dim=0,
        )
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        # Sometimes even with specdec enabled we get single-token prefill chunks that
        # should be treated as decodes but don't have num_accepted_tokens set.
        # These should be fine to process as non-spec decodes since there's only
        # one token, so no risk of placing accepted tokens in the wrong slot.
        if num_decodes > 0 and self.use_spec_decode and num_accepted_tokens is not None:
            query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
            num_accepted_tokens = num_accepted_tokens[:num_decodes]

        if num_prefills > 0:
            if num_computed_tokens is None:
                num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_p = (
                num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0
            )

            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    query_start_loc_p_cpu,
                    device=common_attn_metadata.query_start_loc.device,
                )
            )

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert num_computed_tokens is not None
                num_computed_tokens_p = num_computed_tokens[
                    num_reqs - num_prefills : num_reqs
                ]
                assert block_idx_first_scheduled_token is not None
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                    num_reqs - num_prefills : num_reqs
                ]

        if self.use_cached_kernel and num_decodes > 0:
            num_prompt_tokens_cpu = common_attn_metadata.num_prompt_tokens_cpu
            num_computed_tokens_cpu = common_attn_metadata._num_computed_tokens_cpu
            if num_prompt_tokens_cpu is None or num_computed_tokens_cpu is None:
                raise ValueError(
                    "--use-replayssm requires CPU prompt and "
                    "computed-token counts to derive decode write positions"
                )
            decode_steps_cpu = (
                num_computed_tokens_cpu[:num_decodes]
                - num_prompt_tokens_cpu[:num_decodes]
            )
            query_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1 : num_decodes + 1]
                - common_attn_metadata.query_start_loc_cpu[:num_decodes]
            )
            valid_decode_rows = query_lens_cpu > 0
            if torch.any(decode_steps_cpu[valid_decode_rows] < 0).item():
                raise ValueError(
                    "--use-replayssm requires decode-step counts "
                    "that exclude prompt tokens and start at zero"
                )
            decode_steps_cpu = torch.where(
                valid_decode_rows,
                decode_steps_cpu,
                torch.zeros_like(decode_steps_cpu),
            )
            write_pos_cpu = torch.remainder(decode_steps_cpu, self.max_cache_len)
            is_flush_cpu = (write_pos_cpu == self.max_cache_len - 1).to(torch.int8)
            write_pos_d = async_tensor_h2d(
                write_pos_cpu.to(torch.int32).tolist(),
                dtype=torch.int32,
                device=common_attn_metadata.query_start_loc.device,
            )
            is_flush_d = async_tensor_h2d(
                is_flush_cpu.tolist(),
                dtype=torch.int8,
                device=common_attn_metadata.query_start_loc.device,
            )

        bc_pre_scratch = None
        if (
            self.use_cached_kernel
            and self.cached_kernel_variant == "output_only"
            and self.decode_bc_pre_scratch is not None
            and num_decodes > 0
        ):
            bc_pre_scratch = self.decode_bc_pre_scratch[:num_decodes]

        # cached-SPEC (hybrid): commit-at-start advances the persistent
        # block-keyed cursors using the previous step's num_accepted_tokens,
        # then first-decode rows are reset. The kernels read the full
        # (num_gpu_blocks,) cursor buffers, indexed by physical SSM block id.
        spec_write_pos_d = None
        spec_post_origin_d = None
        spec_is_flush_d = None
        spec_bc_pre_scratch = None
        if (
            self.use_cache_spec_kernel
            and num_decodes > 0
            and self.use_spec_decode
            and num_accepted_tokens is not None
        ):
            from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_spec import (  # noqa: E501
                commit_replayssm_spec,
                reset_replayssm_spec_cursors,
            )

            cursor_device = common_attn_metadata.query_start_loc.device
            if self.spec_write_pos is None:
                n_blocks = self.vllm_config.cache_config.num_gpu_blocks
                assert n_blocks is not None and n_blocks > 0, (
                    "--use-replayssm-spec needs num_gpu_blocks at "
                    "build time to size the block-keyed cursor buffers"
                )
                self.spec_write_pos = torch.zeros(
                    n_blocks, dtype=torch.int32, device=cursor_device
                )
                self.spec_post_origin = torch.zeros(
                    n_blocks, dtype=torch.int32, device=cursor_device
                )
                self.spec_is_flush = torch.zeros(
                    n_blocks, dtype=torch.int8, device=cursor_device
                )
            sbi = state_indices_tensor_d[:, 0]
            commit_replayssm_spec(
                self.spec_write_pos,
                self.spec_post_origin,
                self.spec_is_flush,
                num_accepted_tokens.to(torch.int32),
                sbi,
                max_cache_len=self.spec_flush_threshold,
                max_spec_len=self.max_spec_len,
                cache_buf_len=self.spec_cache_buf_len,
            )
            # prefill->decode reset for first-decode rows (cursors only; no conv
            # seed -- conv_state carries context). A request's first spec verify
            # has num_computed_tokens == num_prompt_tokens; that resets its
            # (possibly recycled) block's cursors to write_pos=0, and -- because
            # the commit above runs BEFORE this reset -- also undoes any write_pos
            # the first-decode commit advanced from the freshly-zeroed cursor.
            # Derive the mask from the DEVICE-side compute_num_computed_tokens()
            # (always populated), NOT _num_computed_tokens_cpu, which is None on
            # the spec verify path -> the old guard silently skipped the reset,
            # leaving recycled blocks with stale cursors and fresh blocks with a
            # wrong first-decode write_pos (coherent-but-divergent output +
            # acceptance drop).
            num_prompt_tokens_cpu = common_attn_metadata.num_prompt_tokens_cpu
            if num_prompt_tokens_cpu is not None:
                ctx_lens = common_attn_metadata.compute_num_computed_tokens()
                num_prompt_d = num_prompt_tokens_cpu.to(
                    ctx_lens.device, non_blocking=True
                )
                first_decode_d = (
                    ctx_lens[:num_decodes] == num_prompt_d[:num_decodes]
                ).to(torch.int8)
                reset_replayssm_spec_cursors(
                    self.spec_write_pos,
                    self.spec_post_origin,
                    self.spec_is_flush,
                    first_decode_d,
                    sbi,
                    max_cache_len=self.spec_flush_threshold,
                    max_spec_len=self.max_spec_len,
                )
            spec_write_pos_d = self.spec_write_pos
            spec_post_origin_d = self.spec_post_origin
            spec_is_flush_d = self.spec_is_flush
            if self.decode_spec_bc_pre is not None:
                spec_bc_pre_scratch = self.decode_spec_bc_pre[:num_decodes]

        metadata = self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_p=query_start_loc_p,
            has_initial_states_p=has_initial_states_p,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            write_pos_d=write_pos_d,
            is_flush_d=is_flush_d,
            bc_pre_scratch=bc_pre_scratch,
            spec_write_pos_d=spec_write_pos_d,
            spec_post_origin_d=spec_post_origin_d,
            spec_is_flush_d=spec_is_flush_d,
            spec_bc_pre_scratch=spec_bc_pre_scratch,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token_prev_step=(
                block_idx_last_scheduled_token_prev_step
            ),
            num_computed_tokens_p=num_computed_tokens_p,
            num_reqs=num_reqs,
            seq_lens=common_attn_metadata.seq_lens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

        return self._update_metadata_for_cudagraph_capture(metadata)

    def _update_metadata_for_cudagraph_capture(
        self,
        metadata: M,
    ) -> M:
        """
        Update the metadata for cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        state_indices_tensor_d = metadata.state_indices_tensor_d
        query_start_loc_d = metadata.query_start_loc_d
        num_accepted_tokens = metadata.num_accepted_tokens
        block_idx_last_scheduled_token = metadata.block_idx_last_scheduled_token
        block_idx_last_computed_token = metadata.block_idx_last_computed_token
        block_idx_last_scheduled_token_prev_step = (
            metadata.block_idx_last_scheduled_token_prev_step
        )
        write_pos_d = metadata.write_pos_d
        is_flush_d = metadata.is_flush_d
        bc_pre_scratch = metadata.bc_pre_scratch
        # cached-spec cursors are full (num_blocks,) fixed-address buffers indexed
        # by physical block id, so they need NO per-batch padding (padding rows
        # carry NULL_BLOCK_ID in state_indices and are skipped by the kernels).
        # Only the per-row CB scratch is re-sliced to the padded batch.
        spec_write_pos_d = metadata.spec_write_pos_d
        spec_post_origin_d = metadata.spec_post_origin_d
        spec_is_flush_d = metadata.spec_is_flush_d
        spec_bc_pre_scratch = metadata.spec_bc_pre_scratch
        if (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            padded_bs = metadata.num_reqs
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

            if self.vllm_config.cache_config.mamba_cache_mode == "all":
                assert block_idx_last_scheduled_token is not None
                assert block_idx_last_computed_token is not None
                self.block_idx_last_scheduled_token[: metadata.num_decodes].copy_(
                    block_idx_last_scheduled_token[: metadata.num_decodes],
                    non_blocking=True,
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :padded_bs
                ]
                block_idx_last_scheduled_token[metadata.num_decodes :] = 0

                self.block_idx_last_computed_token[: metadata.num_decodes].copy_(
                    block_idx_last_computed_token[: metadata.num_decodes],
                    non_blocking=True,
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :padded_bs
                ]
                block_idx_last_computed_token[metadata.num_decodes :] = 0

                if (
                    self.use_spec_decode
                    and block_idx_last_scheduled_token_prev_step is not None
                ):
                    self.block_idx_last_scheduled_token_prev_step[
                        : metadata.num_decodes
                    ].copy_(
                        block_idx_last_scheduled_token_prev_step[
                            : metadata.num_decodes
                        ],
                        non_blocking=True,
                    )
                    block_idx_last_scheduled_token_prev_step = (
                        self.block_idx_last_scheduled_token_prev_step[:padded_bs]
                    )
                    block_idx_last_scheduled_token_prev_step[metadata.num_decodes :] = 0

            if self.use_cached_kernel:
                assert write_pos_d is not None
                assert is_flush_d is not None
                self.decode_write_pos_d[: metadata.num_decodes].copy_(
                    write_pos_d[: metadata.num_decodes],
                    non_blocking=True,
                )
                write_pos_d = self.decode_write_pos_d[:padded_bs]
                write_pos_d[metadata.num_decodes :] = 0

                self.decode_is_flush_d[: metadata.num_decodes].copy_(
                    is_flush_d[: metadata.num_decodes],
                    non_blocking=True,
                )
                is_flush_d = self.decode_is_flush_d[:padded_bs]
                is_flush_d[metadata.num_decodes :] = 0

                if (
                    self.cached_kernel_variant == "output_only"
                    and self.decode_bc_pre_scratch is not None
                ):
                    bc_pre_scratch = self.decode_bc_pre_scratch[:padded_bs]

            if self.use_cache_spec_kernel and self.decode_spec_bc_pre is not None:
                spec_bc_pre_scratch = self.decode_spec_bc_pre[:padded_bs]

        return replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            query_start_loc_d=query_start_loc_d,
            num_accepted_tokens=num_accepted_tokens,
            write_pos_d=write_pos_d,
            is_flush_d=is_flush_d,
            bc_pre_scratch=bc_pre_scratch,
            spec_write_pos_d=spec_write_pos_d,
            spec_post_origin_d=spec_post_origin_d,
            spec_is_flush_d=spec_is_flush_d,
            spec_bc_pre_scratch=spec_bc_pre_scratch,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token_prev_step=(
                block_idx_last_scheduled_token_prev_step
            ),
        )

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
            self.vllm_config.cache_config.mamba_cache_mode,
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
        if self.vllm_config.cache_config.mamba_cache_mode != "all":
            state_indices_tensor_d = state_indices_tensor_d[
                :, : 1 + self.num_spec_tokens
            ]
            state_indices_tensor_p = state_indices_tensor_p[:, 0]

        new_metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
        )

        return self._update_metadata_for_cudagraph_capture(new_metadata)
