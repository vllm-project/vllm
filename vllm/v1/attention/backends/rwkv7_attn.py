# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

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
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class Rwkv7AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RWKV7_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Rwkv7AttentionMetadataBuilder"]:
        return Rwkv7AttentionMetadataBuilder


@dataclass
class Rwkv7AttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_actual_tokens: int

    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None

    # Cache block-table for the recurrent state.
    # - In default mode (mamba_cache_mode != 'all'): shape [batch],
    #   one cache block per active sequence; state is overwritten in place.
    # - In 'all' mode (prefix caching): shape [batch, max_blocks]; each
    #   sequence may use multiple blocks (one per ``mamba_block_size``
    #   tokens), and ``block_idx_*_*`` index into the second dim.
    state_indices_tensor: torch.Tensor | None = None

    # Boolean mask over prefill sequences only: True when the sequence has
    # cached state to load. None when num_prefills == 0.
    has_initial_state: torch.Tensor | None = None

    # Prefix-caching machinery (populated only when mamba_cache_mode == 'all').
    is_mamba_cache_all: bool = False
    mamba_block_size: int | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    num_computed_tokens_p: torch.Tensor | None = None

    # Resolved per-request cache slots (decodes + prefills). 1D, length =
    # num_decodes + num_prefills. Cudagraph-stable under prefix caching.
    read_slot: torch.Tensor | None = None
    write_slot: torch.Tensor | None = None


class Rwkv7AttentionMetadataBuilder(AttentionMetadataBuilder[Rwkv7AttentionMetadata]):
    reorder_batch_threshold: int = 1

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, MambaSpec)

        self.compilation_config = vllm_config.compilation_config
        self._init_reorder_batch_threshold(1, False)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        max_seqs = vllm_config.scheduler_config.max_num_seqs
        self.decode_cudagraph_max_bs: int = max_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # Persistent buffers used as cudagraph-stable destinations for the
        # prefix-cache decode path. With ``mamba_cache_mode == 'all'`` the
        # read / write block indices change per request as block boundaries
        # shift, so we ``copy_`` into these fixed-address tensors during
        # ``build()`` rather than allocating fresh ones.
        self.read_slot: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.write_slot: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Rwkv7AttentionMetadata:
        m = common_attn_metadata
        query_start_loc = m.query_start_loc
        seq_lens = m.seq_lens

        is_mamba_cache_all = self.vllm_config.cache_config.mamba_cache_mode == "all"
        mamba_block_size = self.kv_cache_spec.block_size if is_mamba_cache_all else None

        block_idx_last_computed_token: torch.Tensor | None = None
        block_idx_first_scheduled_token: torch.Tensor | None = None
        block_idx_last_scheduled_token: torch.Tensor | None = None
        num_computed_tokens_p: torch.Tensor | None = None

        if is_mamba_cache_all:
            assert mamba_block_size is not None
            block_table_tensor = m.block_table_tensor
            num_computed_tokens = m.compute_num_computed_tokens()
            block_idx_last_computed_token = torch.clamp(
                cdiv(num_computed_tokens, mamba_block_size) - 1, min=0
            )
            block_idx_first_scheduled_token = (
                cdiv(num_computed_tokens + 1, mamba_block_size) - 1
            )
            block_idx_last_scheduled_token = torch.clamp(
                cdiv(seq_lens, mamba_block_size) - 1, min=0
            )
            state_indices_tensor = block_table_tensor
            read_slot = block_table_tensor.gather(
                1, block_idx_last_computed_token.long().unsqueeze(1)
            ).squeeze(1)
            write_slot = block_table_tensor.gather(
                1, block_idx_last_scheduled_token.long().unsqueeze(1)
            ).squeeze(1)
        else:
            block_table_tensor = mamba_get_block_table_tensor(
                m.block_table_tensor,
                m.seq_lens,
                self.kv_cache_spec,
                self.vllm_config.cache_config.mamba_cache_mode,
            )
            state_indices_tensor = block_table_tensor[:, 0]
            read_slot = state_indices_tensor
            write_slot = state_indices_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                m,
                decode_threshold=self.reorder_batch_threshold,
                treat_short_extends_as_decodes=False,
            )
        )

        has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor[num_decodes:] > 0
            if is_mamba_cache_all:
                num_computed_tokens_p = context_lens_tensor[num_decodes:]

        # Pre-allocate cudagraph-stable buffers when the batch is a uniform
        # decode that fits in our captured size. The read / write slot
        # values change per request as prefix-cache block boundaries shift,
        # so we copy_ into fixed-address tensors.
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.read_slot[:num_decodes].copy_(
                read_slot[:num_decodes], non_blocking=True
            )
            read_slot = self.read_slot[: m.num_actual_tokens]
            self.write_slot[:num_decodes].copy_(
                write_slot[:num_decodes], non_blocking=True
            )
            write_slot = self.write_slot[: m.num_actual_tokens]

        return Rwkv7AttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            state_indices_tensor=state_indices_tensor,
            has_initial_state=has_initial_state,
            is_mamba_cache_all=is_mamba_cache_all,
            mamba_block_size=mamba_block_size,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            num_computed_tokens_p=num_computed_tokens_p,
            read_slot=read_slot,
            write_slot=write_slot,
        )
