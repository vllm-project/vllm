# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    # The following tensor is only used for prefix caching in align mode
    seq_lens: torch.Tensor

    has_initial_state: torch.Tensor | None = None
    block_size: int | None = None
    chunk_size: int | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # When APC is enabled, the state_indices_tensor is the mapping from logical
    # block indices to the state indices used to extract the (ssm, conv) state.
    # When prefix caching is enabled: shape [batch, max_num_blocks]
    # When APC is disabled, state_indices_tensor is None and unused and logic is
    # driven by the spec/non_spec_state_indices_tensor
    state_indices_tensor: torch.Tensor | None = None

    # The following tensors are only used for prefix caching and are None if disabled
    # E.g., for a request i with cached state we can think of the corresponding
    # cached state index as state_indices_tensor[i, block_idx_last_computed_token]
    block_idx_last_scheduled_token: torch.Tensor | None = None  # shape: [batch,]
    block_idx_first_scheduled_token_p: torch.Tensor | None = None  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor | None = None  # shape: [batch,]
    num_computed_tokens_p: torch.Tensor | None = None  # shape: [batch,]

    # These are non-None if there are prefill requests in the batch
    seq_idx_p: torch.Tensor | None = None  # shape: [batch,]
    cu_chunk_seqlen_p: torch.Tensor | None = None  # shape: [batch,]
    last_chunk_indices_p: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[GDNAttentionMetadata]  # type: ignore
):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    # GDN metadata has additional fields (non_spec_state_indices_tensor, etc.)
    # that are NOT handled by the base class update_block_table(), so we must
    # always rebuild via build() to keep every field consistent.
    supports_update_block_table: bool = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        all_prefix_caching_enabled = vllm_config.cache_config.mamba_cache_mode == "all"

        # 64 is a hardcoded value in the FLA GDN kernel.
        # https://github.com/fla-org/flash-linear-attention/blob/2e7336262c11f8bc6cd6a94b1eb5ee353ae8b4cd/fla/ops/common/chunk_delta_h.py#L439  # noqa: E501
        self.chunk_size = 64
        if all_prefix_caching_enabled and (
            kv_cache_spec.block_size % self.chunk_size != 0
        ):
            raise ValueError(
                "GDN prefix caching requires the mamba block size to be a "
                f"multiple of the kernel chunk size ({self.chunk_size})."
            )

        if all_prefix_caching_enabled and self.use_spec_decode:
            raise NotImplementedError(
                "GDN prefix caching is currently supported only for decode-only "
                "workloads; speculative decoding with APC will be added separately."
            )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    @staticmethod
    def _compute_chunk_metadata(
        num_prefills: int,
        query_start_loc_p_cpu: torch.Tensor,
        chunk_size: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Compute chunk-specific metadata for GDN.

        Unlike Mamba2, the FLA GDN kernel simply divides sequences into
        chunk_size-token chunks based on total sequence length, without
        any alignment logic based on num_computed_tokens. This function
        matches that behavior.

        Args:
            num_prefills: Number of prefill sequences
            query_start_loc_p_cpu: Cumulative sequence lengths for prefill sequences,
                shape [num_prefills + 1]
            chunk_size: Size of each chunk (64 for GDN)

        Returns:
            cu_chunk_seqlen: Cumulative chunk sequence lengths
            seq_idx: Sequence index for each chunk
            last_chunk_indices: Last chunk index for each sequence
        """
        cu_chunk_seqlen = []
        seq_idx = []
        last_chunk_indices = []
        chunk_pos = 0

        for req_idx in range(num_prefills):
            seq_len = (
                query_start_loc_p_cpu[req_idx + 1].item()
                - query_start_loc_p_cpu[req_idx].item()
            )
            # Simply divide the sequence into chunks based on total length
            # This matches how prepare_chunk_indices works in the FLA kernel
            n_chunks = cdiv(seq_len, chunk_size)
            for _ in range(n_chunks):
                seq_idx.append(req_idx)
                cu_chunk_seqlen.append(chunk_pos)
                chunk_pos += 1

            # Record the last chunk index for this sequence
            if n_chunks > 0:
                last_chunk_indices.append(chunk_pos - 1)
            else:
                last_chunk_indices.append(-1)

        return cu_chunk_seqlen, seq_idx, last_chunk_indices

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        prefix_caching_enabled = self.vllm_config.cache_config.mamba_cache_mode == "all"
        block_size: int | None = None
        chunk_size_value: int | None = None
        if prefix_caching_enabled:
            block_size = self.kv_cache_spec.block_size
            chunk_size_value = self.chunk_size

        # APC related tensors
        state_indices_tensor: torch.Tensor | None = None
        block_idx_first_scheduled_token: torch.Tensor | None = None
        block_idx_last_computed_token: torch.Tensor | None = None
        block_idx_last_scheduled_token: torch.Tensor | None = None
        block_idx_first_scheduled_token_p: torch.Tensor | None = None

        num_computed_tokens_p: torch.Tensor | None = None
        seq_idx_p: torch.Tensor | None = None
        cu_chunk_seqlen_p: torch.Tensor | None = None
        last_chunk_indices_p: torch.Tensor | None = None
        non_spec_query_start_loc_cpu: torch.Tensor | None = None

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            non_spec_query_lens = query_lens[~spec_sequence_masks]
            num_decodes = (non_spec_query_lens == 1).sum().item()
            num_prefills = non_spec_query_lens.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens.sum().item() - num_decode_tokens
            num_spec_decode_tokens = (
                query_lens.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                spec_state_indices_tensor = block_table_tensor[:, : self.num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if prefix_caching_enabled:
            assert block_size is not None
            state_indices_tensor = m.block_table_tensor
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(m, block_size)

        if prefix_caching_enabled and num_prefills > 0:
            assert block_idx_first_scheduled_token is not None
            assert non_spec_query_start_loc_cpu is not None

            prefill_start = num_decodes
            prefill_end = prefill_start + num_prefills
            block_idx_first_scheduled_token_p = block_idx_first_scheduled_token[
                prefill_start:prefill_end
            ]
            num_computed_tokens_p = context_lens_tensor[prefill_start:prefill_end]
            query_start_loc_p_cpu = (
                non_spec_query_start_loc_cpu[-num_prefills - 1 :] - num_decode_tokens
            )
            cu_chunk_seqlen, seq_idx, last_chunk_indices = self._compute_chunk_metadata(
                num_prefills=int(num_prefills),
                query_start_loc_p_cpu=query_start_loc_p_cpu,
                chunk_size=self.chunk_size,
            )
            seq_idx_p = torch.as_tensor(
                seq_idx,
                device=m.query_start_loc.device,
                dtype=torch.int32,
            )
            cu_chunk_seqlen_p = torch.as_tensor(
                cu_chunk_seqlen,
                device=m.query_start_loc.device,
                dtype=torch.int32,
            )
            last_chunk_indices_p = torch.as_tensor(
                last_chunk_indices,
                device=m.query_start_loc.device,
                dtype=torch.int32,
            )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

            if prefix_caching_enabled and num_decodes > 0:
                self.state_indices_tensor[:num_decodes].copy_(
                    state_indices_tensor, non_blocking=True
                )
                state_indices_tensor = self.state_indices_tensor[:num_decodes]

                assert block_idx_last_scheduled_token is not None
                assert block_idx_last_computed_token is not None
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token[:num_decodes],
                    non_blocking=True,
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_decodes
                ]
                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token[:num_decodes],
                    non_blocking=True,
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_decodes
                ]

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            seq_lens=m.seq_lens,
            block_size=block_size,
            chunk_size=chunk_size_value,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            state_indices_tensor=state_indices_tensor,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token=block_idx_last_computed_token,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            num_computed_tokens_p=num_computed_tokens_p,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
