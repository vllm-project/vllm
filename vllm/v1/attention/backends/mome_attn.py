# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, SlidingWindowMomeSpec


@dataclass
class MomeAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    num_computed_tokens_p: torch.Tensor | None
    state_indices_tensor_p: torch.Tensor | None

    num_computed_tokens_d: torch.Tensor | None
    state_indices_tensor_d: torch.Tensor | None
    query_start_loc_d: torch.Tensor | None
    num_accepted_tokens: torch.Tensor | None

    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token_p: torch.Tensor | None
    block_idx_first_scheduled_token_d: torch.Tensor | None
    block_idx_last_scheduled_token_d: torch.Tensor | None
    block_idx_last_computed_token_d: torch.Tensor | None

    seq_lens: torch.Tensor

    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None
    max_decode_query_len: int | None = None


class MomeAttentionMetadataBuilder(AttentionMetadataBuilder[MomeAttentionMetadata]):
    """Metadata builder for MoME short-conv states.

    MoME uses the SlidingWindowManager for cache ownership, but its causal
    convolution kernels need explicit read/write block offsets. This builder
    intentionally does not inherit the Mamba builder because Mamba cache modes
    collapse the block table in ways that are not valid for sliding-window MoME
    state.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, SlidingWindowMomeSpec)

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

        max_num_blocks = cdiv(
            self.vllm_config.model_config.max_model_len,
            kv_cache_spec.block_size,
        )
        self.state_indices_tensor_d: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, max_num_blocks),
            dtype=torch.int32,
            device=device,
        )
        self.block_idx_last_scheduled_token_d: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.block_idx_last_computed_token_d: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.num_computed_tokens_d: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.block_idx_first_scheduled_token_d: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )

        if self.num_spec_tokens > 0:
            self.decode_num_accepted_tokens: torch.Tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.supports_update_block_table = False

        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        self._draft_num_accepted_tokens: torch.Tensor | None = None

    def set_draft_attention_metadata(
        self,
        num_accepted_tokens: torch.Tensor | None,
    ) -> None:
        self._draft_num_accepted_tokens = num_accepted_tokens

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> MomeAttentionMetadata:
        del draft_index
        common_attn_metadata = self._treat_single_token_prefills_as_decodes(
            common_attn_metadata
        )
        return self._compute_metadata(
            common_attn_metadata,
            num_accepted_tokens=self._draft_num_accepted_tokens,
            is_drafting=True,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> MomeAttentionMetadata:
        m = common_attn_metadata
        assert (
            m.max_query_len <= 1 + self.num_spec_tokens
            and m.num_reqs <= self.decode_cudagraph_max_bs
        ), (
            "MoME only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )
        assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only

        num_accepted_tokens = None
        if self.num_spec_tokens > 0:
            num_accepted_tokens = torch.diff(m.query_start_loc)

        common_attn_metadata = self._treat_single_token_prefills_as_decodes(m)
        return self._compute_metadata(
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            require_uniform=True,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_prompt_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MomeAttentionMetadata:
        del common_prefix_len, fast_build, kwargs
        common_attn_metadata = self._treat_single_token_prefills_as_decodes(
            common_attn_metadata
        )
        return self._compute_metadata(
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_prompt_tokens=num_prompt_tokens,
        )

    def _treat_single_token_prefills_as_decodes(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> CommonAttentionMetadata:
        is_prefilling = common_attn_metadata.is_prefilling
        if is_prefilling is None:
            return common_attn_metadata

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if seq_lens_cpu is None:
            return common_attn_metadata

        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
        single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)
        has_prior_state = seq_lens_cpu > 1
        prefill_to_decode = single_token_prefill_rows & has_prior_state
        if torch.any(prefill_to_decode).item():
            is_prefilling = is_prefilling.clone()
            is_prefilling[prefill_to_decode] = False
            common_attn_metadata = common_attn_metadata.replace(
                is_prefilling=is_prefilling
            )
        return common_attn_metadata

    def _compute_block_indices(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = self.kv_cache_spec.block_size
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        block_idx_last_computed_token = cdiv(num_computed_tokens, block_size) - 1
        block_idx_first_scheduled_token = cdiv(num_computed_tokens + 1, block_size) - 1
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, block_size) - 1
        )
        block_idx_last_computed_token = torch.clamp(
            block_idx_last_computed_token, min=0
        )
        block_idx_last_scheduled_token = torch.clamp(
            block_idx_last_scheduled_token, min=0
        )
        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _compute_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_prompt_tokens: torch.Tensor | None = None,
        is_drafting: bool = False,
        require_uniform: bool = False,
    ) -> MomeAttentionMetadata:
        if num_accepted_tokens is not None:
            assert self.reorder_batch_threshold is not None
            decode_threshold = self.reorder_batch_threshold
        else:
            decode_threshold = 1
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=decode_threshold,
                require_uniform=require_uniform,
                treat_short_extends_as_decodes=False,
            )
        )

        num_reqs = common_attn_metadata.num_reqs
        state_indices_tensor = common_attn_metadata.block_table_tensor[:num_reqs]
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)
        state_indices_tensor = state_indices_tensor.contiguous()

        (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        ) = self._compute_block_indices(common_attn_metadata)

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor, [num_decodes, num_prefills], dim=0
        )
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        num_computed_tokens_d, num_computed_tokens_p = torch.split(
            num_computed_tokens, [num_decodes, num_prefills], dim=0
        )
        block_idx_last_computed_token_d, block_idx_last_computed_token_p = torch.split(
            block_idx_last_computed_token,
            [num_decodes, num_prefills],
            dim=0,
        )
        block_idx_first_scheduled_token_d, block_idx_first_scheduled_token_p = (
            torch.split(
                block_idx_first_scheduled_token, [num_decodes, num_prefills], dim=0
            )
        )
        block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
            torch.split(
                block_idx_last_scheduled_token,
                [num_decodes, num_prefills],
                dim=0,
            )
        )

        query_start_loc_d = None
        max_decode_query_len = None
        num_accepted_tokens_d = None
        if num_decodes > 0:
            query_start_loc_d = common_attn_metadata.query_start_loc[: num_decodes + 1]
            query_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )
            max_decode_query_len = int(query_lens_cpu.max().item())
            if num_accepted_tokens is not None:
                block_size = self.kv_cache_spec.block_size
                num_accepted_tokens_d = num_accepted_tokens[:num_decodes]
                prev_num_computed_tokens_d = (
                    num_computed_tokens_d - num_accepted_tokens_d
                )
                prev_scheduled_len_d = torch.clamp(
                    self.vllm_config.model_config.max_model_len
                    - 1
                    - prev_num_computed_tokens_d,
                    min=1,
                    max=self.num_spec_tokens + 1,
                )
                spec_block_idx_last_computed_token_d = (
                    cdiv(prev_num_computed_tokens_d + prev_scheduled_len_d, block_size)
                    - 1
                )
                use_spec_block_idx: torch.Tensor | None = None
                if is_drafting:
                    use_spec_block_idx = num_accepted_tokens_d > 0
                elif num_prompt_tokens is not None:
                    num_prompt_tokens = num_prompt_tokens.to(
                        device=num_computed_tokens.device, non_blocking=True
                    )
                    num_prompt_tokens_d = num_prompt_tokens[:num_decodes]
                    use_spec_block_idx = num_computed_tokens_d > num_prompt_tokens_d
                if use_spec_block_idx is not None:
                    block_idx_last_computed_token_d = torch.where(
                        use_spec_block_idx,
                        spec_block_idx_last_computed_token_d,
                        block_idx_last_computed_token_d,
                    )
                block_idx_last_computed_token_d = torch.clamp(
                    block_idx_last_computed_token_d, min=0
                )

        has_initial_states_p = None
        query_start_loc_p = None
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        if num_prefills > 0:
            query_start_loc_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1 :]
                - num_decode_tokens
            )
            query_start_loc_p = (
                common_attn_metadata.query_start_loc[-num_prefills - 1 :]
                - num_decode_tokens
            )
            has_initial_states_p = num_computed_tokens_p > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    query_start_loc_p_cpu,
                    device=common_attn_metadata.query_start_loc.device,
                )
            )

        metadata = MomeAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_reqs=num_reqs,
            has_initial_states_p=has_initial_states_p,
            query_start_loc_p=query_start_loc_p,
            num_computed_tokens_p=num_computed_tokens_p,
            num_computed_tokens_d=num_computed_tokens_d,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            query_start_loc_d=query_start_loc_d,
            num_accepted_tokens=num_accepted_tokens_d,
            max_decode_query_len=max_decode_query_len,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_scheduled_token_p=block_idx_last_scheduled_token_p,
            block_idx_last_computed_token_p=block_idx_last_computed_token_p,
            block_idx_first_scheduled_token_d=block_idx_first_scheduled_token_d,
            block_idx_last_scheduled_token_d=block_idx_last_scheduled_token_d,
            block_idx_last_computed_token_d=block_idx_last_computed_token_d,
            seq_lens=common_attn_metadata.seq_lens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return self._update_metadata_for_cudagraph_capture(metadata)

    def _update_metadata_for_cudagraph_capture(
        self, metadata: MomeAttentionMetadata
    ) -> MomeAttentionMetadata:
        if not (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            return metadata

        padded_bs = metadata.num_reqs
        state_indices_tensor_d = metadata.state_indices_tensor_d
        assert state_indices_tensor_d is not None
        num_cols = state_indices_tensor_d.shape[1]
        assert num_cols <= self.state_indices_tensor_d.shape[1], (
            "MoME decode block table is wider than the cudagraph buffer: "
            f"{num_cols} > {self.state_indices_tensor_d.shape[1]}."
        )

        padded_state_indices_tensor_d = self.state_indices_tensor_d[:padded_bs]
        padded_state_indices_tensor_d.fill_(NULL_BLOCK_ID)
        padded_state_indices_tensor_d[: metadata.num_decodes, :num_cols].copy_(
            state_indices_tensor_d, non_blocking=True
        )

        num_computed_tokens_d = metadata.num_computed_tokens_d
        block_idx_last_scheduled_token_d = metadata.block_idx_last_scheduled_token_d
        block_idx_last_computed_token_d = metadata.block_idx_last_computed_token_d
        block_idx_first_scheduled_token_d = metadata.block_idx_first_scheduled_token_d
        assert block_idx_last_scheduled_token_d is not None
        assert block_idx_last_computed_token_d is not None
        assert num_computed_tokens_d is not None
        assert block_idx_first_scheduled_token_d is not None

        padded_num_computed_tokens_d = self.num_computed_tokens_d[:padded_bs]
        padded_num_computed_tokens_d[: metadata.num_decodes].copy_(
            num_computed_tokens_d[: metadata.num_decodes], non_blocking=True
        )
        padded_num_computed_tokens_d[metadata.num_decodes :] = 0

        padded_block_idx_first_scheduled_token_d = (
            self.block_idx_first_scheduled_token_d[:padded_bs]
        )
        padded_block_idx_first_scheduled_token_d[: metadata.num_decodes].copy_(
            block_idx_first_scheduled_token_d[: metadata.num_decodes],
            non_blocking=True,
        )
        padded_block_idx_first_scheduled_token_d[metadata.num_decodes :] = 0

        padded_block_idx_last_scheduled_token_d = self.block_idx_last_scheduled_token_d[
            :padded_bs
        ]
        padded_block_idx_last_scheduled_token_d[: metadata.num_decodes].copy_(
            block_idx_last_scheduled_token_d[: metadata.num_decodes],
            non_blocking=True,
        )
        padded_block_idx_last_scheduled_token_d[metadata.num_decodes :] = 0

        padded_block_idx_last_computed_token_d = self.block_idx_last_computed_token_d[
            :padded_bs
        ]
        padded_block_idx_last_computed_token_d[: metadata.num_decodes].copy_(
            block_idx_last_computed_token_d[: metadata.num_decodes],
            non_blocking=True,
        )
        padded_block_idx_last_computed_token_d[metadata.num_decodes :] = 0

        num_accepted_tokens = metadata.num_accepted_tokens
        query_start_loc_d = metadata.query_start_loc_d
        assert query_start_loc_d is not None
        query_start_loc_d = query_start_loc_d[: padded_bs + 1]
        if self.use_spec_decode and num_accepted_tokens is not None:
            self.decode_num_accepted_tokens[: metadata.num_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
            num_accepted_tokens[metadata.num_decodes :] = 1

        return replace(
            metadata,
            state_indices_tensor_d=padded_state_indices_tensor_d,
            num_computed_tokens_d=padded_num_computed_tokens_d,
            block_idx_first_scheduled_token_d=padded_block_idx_first_scheduled_token_d,
            block_idx_last_scheduled_token_d=(padded_block_idx_last_scheduled_token_d),
            block_idx_last_computed_token_d=padded_block_idx_last_computed_token_d,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc_d=query_start_loc_d,
        )

    def update_block_table(
        self,
        metadata: MomeAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> MomeAttentionMetadata:
        del slot_mapping
        state_indices_tensor = blk_table
        if state_indices_tensor.dim() == 1:
            state_indices_tensor = state_indices_tensor.unsqueeze(-1)
        state_indices_tensor = state_indices_tensor.contiguous()

        assert (
            metadata.num_prefills + metadata.num_decodes
            == state_indices_tensor.shape[0]
        ), (
            "Mismatch in number of requests when updating MoME block table."
            f" Expected {metadata.num_prefills + metadata.num_decodes}, "
            f"got {state_indices_tensor.shape[0]}."
        )

        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor,
            [metadata.num_decodes, metadata.num_prefills],
            dim=0,
        )
        new_metadata = replace(
            metadata,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
        )
        return self._update_metadata_for_cudagraph_capture(new_metadata)


class MomeAttentionBackend(AttentionBackend):
    @staticmethod
    def get_builder_cls() -> type["MomeAttentionMetadataBuilder"]:
        return MomeAttentionMetadataBuilder
