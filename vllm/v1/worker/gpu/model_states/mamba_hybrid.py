# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel
from vllm.v1.worker.utils import AttentionGroup


@triton.jit
def _preprocess_mamba_fused_kernel(
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    src_ssm_col_ptr,
    conv_src_col_ptr,
    conv_src_off_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
):
    """Fused preprocess: compute src columns for SSM/conv AND advance
    state_idx with accepted-token reset, in a single kernel launch.

    Per batch_idx (0..num_reqs-1):
      1. Read pre-advance state_idx and num_accepted (last step's values).
      2. Compute and store src columns for the forward pass:
         - src_ssm_col = state_idx + num_accepted - 1  (or -1 if fresh)
         - conv_src_col = state_idx
         - conv_src_off = max(num_accepted - 1, 0)
      3. Advance state_idx to the new block and reset num_accepted=1
         when a block boundary is crossed.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    req_indices = tl.load(idx_mapping_ptr + offsets, mask=mask, other=0)

    state_idx = tl.load(state_idx_ptr + req_indices, mask=mask, other=-1)
    num_accepted = tl.load(num_accepted_tokens_ptr + req_indices, mask=mask, other=1)

    ssm_col = tl.where(state_idx >= 0, state_idx + num_accepted - 1, -1)
    conv_off = tl.maximum(num_accepted - 1, 0)
    tl.store(src_ssm_col_ptr + req_indices, ssm_col, mask=mask)
    tl.store(conv_src_col_ptr + req_indices, state_idx, mask=mask)
    tl.store(conv_src_off_ptr + req_indices, conv_off, mask=mask)

    num_computed = tl.load(num_computed_tokens_ptr + req_indices, mask=mask, other=0)
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    computed_after = num_computed + query_end - query_start
    new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    tl.store(state_idx_ptr + req_indices, new_state_idx, mask=mask)
    should_reset = (state_idx >= 0) & (state_idx != new_state_idx)
    tl.store(num_accepted_tokens_ptr + req_indices, 1, mask=mask & should_reset)


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None
    # SSM/conv src columns in request-state-slot order (-1 = fresh). gdn_attn
    # gathers them into batch order via align_idx_mapping and resolves them to
    # physical block ids, so kernels read init state from the src block directly.
    align_src_ssm_col: torch.Tensor | None = None
    align_conv_src_col: torch.Tensor | None = None
    align_conv_src_off: torch.Tensor | None = None
    align_idx_mapping: torch.Tensor | None = None
    align_num_reqs: int = 0

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
            # Src columns are passed unsliced: they are indexed by req-state slot
            # (which can be >= num_reqs); align_idx_mapping + align_num_reqs drive
            # the batch-order gather + resolve.
            "align_src_ssm_col": self.align_src_ssm_col,
            "align_conv_src_col": self.align_conv_src_col,
            "align_conv_src_off": self.align_conv_src_off,
            "align_idx_mapping": self.align_idx_mapping,
            "align_num_reqs": self.align_num_reqs,
        }


class MambaCacheAlignInfo:
    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        self.state_idx_gpu = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        # Src columns for SSM and conv (per request slot, -1 = fresh).
        # SSM: state_idx + (num_accepted - 1); conv: state_idx, offset separately.
        self.src_ssm_col_gpu = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.conv_src_col_gpu = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.conv_src_off_gpu = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.current_step_block_tables: tuple[torch.Tensor, ...] | None = None
        self.current_step_kv_cache_config: KVCacheConfig | None = None
        self.group_kv_cache_config: KVCacheConfig | None = None
        self.group_ids: list[int] = []
        self.spec: MambaSpec | None = None

        # Fused postprocess metadata (initialized lazily on first call)
        self.fused_initialized = False
        self.fused_state_base_addrs: torch.Tensor | None = None
        self.fused_state_block_strides: torch.Tensor | None = None
        self.fused_state_elem_sizes: torch.Tensor | None = None
        self.fused_state_inner_sizes: torch.Tensor | None = None
        self.fused_state_conv_widths: torch.Tensor | None = None
        self.fused_state_group_indices: torch.Tensor | None = None
        self.fused_state_dim_row_count: torch.Tensor | None = None
        self.fused_state_dim_row_stride: torch.Tensor | None = None
        self.fused_block_table_ptrs: torch.Tensor | None = None
        self.fused_block_table_stride_req: int = 0
        self.fused_num_layers: int = 0
        self.fused_num_state_types: int = 0


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        if self.cache_config.mamba_cache_mode == "align":
            self.mamba_cache_align_info = MambaCacheAlignInfo(
                self.max_num_reqs, self.device
            )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self.cache_config.mamba_cache_mode == "align":
            state_idx = (new_req_data.num_computed_tokens - 1) // (
                self.cache_config.block_size
            )
            self.mamba_cache_align_info.state_idx_gpu[req_index] = state_idx
        self.num_accepted_tokens_gpu[req_index] = 1

    @staticmethod
    def _get_mamba_group_ids(
        kv_cache_config: KVCacheConfig,
    ) -> tuple[list[int], MambaSpec]:
        mamba_group_ids: list[int] = []
        mamba_specs: list[MambaSpec] = []
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            spec = kv_cache_group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                mamba_group_ids.append(i)
                mamba_specs.append(spec)
        assert mamba_specs, "no mamba layers in the model"
        assert all(mamba_specs[0] == spec for spec in mamba_specs)
        return mamba_group_ids, mamba_specs[0]

    def _get_mamba_group_info(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[list[int], MambaSpec]:
        info = self.mamba_cache_align_info
        if info.group_kv_cache_config is not kv_cache_config:
            mamba_group_ids, mamba_spec = self._get_mamba_group_ids(kv_cache_config)
            info.group_kv_cache_config = kv_cache_config
            info.group_ids = mamba_group_ids
            info.spec = mamba_spec
        assert info.spec is not None
        return info.group_ids, info.spec

    def _preprocess_mamba_cache_align(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        info = self.mamba_cache_align_info

        info.current_step_block_tables = block_tables
        info.current_step_kv_cache_config = kv_cache_config

        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        block_size = 256
        grid = (triton.cdiv(num_reqs, block_size),)
        _preprocess_mamba_fused_kernel[grid](
            input_batch.idx_mapping,
            info.state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            info.src_ssm_col_gpu,
            info.conv_src_col_gpu,
            info.conv_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=block_size,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        if self.cache_config.mamba_cache_mode == "align":
            self._preprocess_mamba_cache_align(
                input_batch,
                block_tables,
                kv_cache_config,
                num_computed_tokens,
            )

    def _initialize_fused_postprocess(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> None:
        info = self.mamba_cache_align_info
        if info.fused_initialized:
            return

        forward_context = self.vllm_config.compilation_config.static_forward_context
        state_copy_funcs = self.model.get_mamba_state_copy_func()
        num_state_types = len(state_copy_funcs)
        num_layers = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        )
        total_states = num_layers * num_state_types

        base_addrs = torch.zeros(total_states, dtype=torch.int64, device=self.device)
        block_strides = torch.zeros(total_states, dtype=torch.int64, device=self.device)
        elem_sizes = torch.zeros(total_states, dtype=torch.int32, device=self.device)
        inner_sizes = torch.zeros(total_states, dtype=torch.int64, device=self.device)
        conv_widths = torch.zeros(total_states, dtype=torch.int32, device=self.device)
        group_indices = torch.zeros(total_states, dtype=torch.int32, device=self.device)
        # DS conv row metadata (zero keeps the single-region copy path for SD).
        dim_row_count = torch.zeros(total_states, dtype=torch.int32, device=self.device)
        dim_row_stride = torch.zeros(
            total_states, dtype=torch.int64, device=self.device
        )

        idx = 0
        for group_local_idx, gid in enumerate(mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
            for layer_name in layer_names:
                kv_caches = forward_context[layer_name].kv_cache
                for st_idx, state in enumerate(kv_caches):
                    base_addrs[idx] = state.data_ptr()
                    blk_stride = state.stride(0) if state.dim() > 1 else state.numel()
                    block_strides[idx] = blk_stride * state.element_size()
                    elem_sizes[idx] = state.element_size()

                    copy_func = state_copy_funcs[st_idx]
                    if copy_func is get_conv_copy_spec:
                        conv_w = state.size(1) if state.dim() > 1 else 0
                        conv_widths[idx] = conv_w
                        inner_sizes[idx] = state.stride(1) if state.dim() > 2 else 1
                    else:
                        conv_widths[idx] = 0
                        inner_sizes[idx] = state[0].numel() if state.dim() > 1 else 1
                    group_indices[idx] = group_local_idx
                    idx += 1

        num_groups = len(mamba_group_ids)
        bt_ptrs = torch.zeros(num_groups, dtype=torch.int64, device=self.device)
        strides = {block_tables[gid].stride(0) for gid in mamba_group_ids}
        assert len(strides) == 1
        for i, gid in enumerate(mamba_group_ids):
            bt_ptrs[i] = block_tables[gid].data_ptr()

        info.fused_state_base_addrs = base_addrs
        info.fused_state_block_strides = block_strides
        info.fused_state_elem_sizes = elem_sizes
        info.fused_state_inner_sizes = inner_sizes
        info.fused_state_conv_widths = conv_widths
        info.fused_state_group_indices = group_indices
        info.fused_state_dim_row_count = dim_row_count
        info.fused_state_dim_row_stride = dim_row_stride
        info.fused_block_table_ptrs = bt_ptrs
        info.fused_block_table_stride_req = int(next(iter(strides)))
        info.fused_num_layers = num_layers
        info.fused_num_state_types = num_state_types
        info.fused_initialized = True

    def _postprocess_mamba_cache_align(
        self,
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        num_reqs: int,
        query_start_loc: torch.Tensor | None,
    ) -> None:
        info = self.mamba_cache_align_info
        block_tables = info.current_step_block_tables
        kv_cache_config = info.current_step_kv_cache_config
        assert block_tables is not None
        assert kv_cache_config is not None
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)

        self._initialize_fused_postprocess(
            kv_cache_config, mamba_group_ids, block_tables
        )

        total_states = info.fused_num_layers * info.fused_num_state_types
        grid = (num_reqs, total_states)

        postprocess_mamba_fused_kernel[grid](
            self.num_accepted_tokens_gpu,
            info.state_idx_gpu,
            None,
            num_computed_tokens,
            None,
            info.fused_block_table_ptrs,
            info.fused_block_table_stride_req,
            info.fused_state_base_addrs,
            info.fused_state_block_strides,
            info.fused_state_elem_sizes,
            info.fused_state_inner_sizes,
            info.fused_state_conv_widths,
            info.fused_state_group_indices,
            info.fused_state_dim_row_count,
            info.fused_state_dim_row_stride,
            None,
            idx_mapping,
            num_reqs,
            block_size=mamba_spec.block_size,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=True,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                # A row is a spec-decode row only when its whole prompt is already
                # computed, i.e. exactly one non-draft (decode) token is scheduled.
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        # Copy-free src redirect: pass the req-order src columns + idx_mapping
        # straight through; gdn_attn does the batch-order gather and resolve (no
        # advanced-index gathers / padded allocations here).
        align_src_ssm_col = None
        align_conv_src_col = None
        align_conv_src_off = None
        align_idx_mapping = None
        align_num_reqs_out = 0
        if not for_capture and self.cache_config.mamba_cache_mode == "align":
            info = self.mamba_cache_align_info
            align_src_ssm_col = info.src_ssm_col_gpu
            align_conv_src_col = info.conv_src_col_gpu
            align_conv_src_off = info.conv_src_off_gpu
            align_idx_mapping = input_batch.idx_mapping
            align_num_reqs_out = input_batch.num_reqs

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            align_src_ssm_col=align_src_ssm_col,
            align_conv_src_col=align_conv_src_col,
            align_conv_src_off=align_conv_src_off,
            align_idx_mapping=align_idx_mapping,
            align_num_reqs=align_num_reqs_out,
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
        num_reqs: int | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        if not isinstance(num_sampled, int):
            # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
            # kernel skips them rather than scattering with a host-side gather.
            n = idx_mapping.shape[0]
            if n:
                _scatter_num_accepted_kernel[(n,)](
                    idx_mapping, num_sampled, self.num_accepted_tokens_gpu
                )
        else:
            self.num_accepted_tokens_gpu.index_fill_(
                0, idx_mapping, max(num_sampled, 1)
            )

        if (
            self.cache_config.mamba_cache_mode == "align"
            and num_computed_tokens is not None
            and num_reqs is not None
        ):
            self._postprocess_mamba_cache_align(
                idx_mapping,
                num_computed_tokens,
                num_reqs,
                query_start_loc,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))
