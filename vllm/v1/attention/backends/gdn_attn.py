# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
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

    # Decode-side APC metadata
    state_indices_tensor_d: torch.Tensor | None = None
    state_indices_tensor_p: torch.Tensor | None = None
    block_idx_last_computed_token_d: torch.Tensor | None = None
    block_idx_last_scheduled_token_d: torch.Tensor | None = None

    # Prefill-side APC metadata
    block_idx_first_scheduled_token_p: torch.Tensor | None = None
    block_idx_last_computed_token_p: torch.Tensor | None = None
    block_idx_last_scheduled_token_p: torch.Tensor | None = None
    seq_idx_p: torch.Tensor | None = None
    cu_chunk_seqlen_p: torch.Tensor | None = None
    last_chunk_indices_p: torch.Tensor | None = None
    num_computed_tokens_p: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.chunk_size = vllm_config.model_config.get_mamba_chunk_size() or 64
        if self.vllm_config.cache_config.enable_prefix_caching and (
            kv_cache_spec.block_size % self.chunk_size != 0
        ):
            raise ValueError(
                "GDN prefix caching requires the mamba block size to be a "
                "multiple of the kernel chunk size."
            )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            self.compilation_config.max_capture_size,
        )

        self._max_cached_blocks = cdiv(
            vllm_config.model_config.max_model_len, kv_cache_spec.block_size
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

        if self.vllm_config.cache_config.enable_prefix_caching:
            self.state_indices_tensor_d_buf = torch.empty(
                (self.decode_cudagraph_max_bs, self._max_cached_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.state_indices_tensor_p_buf = torch.empty(
                (self.decode_cudagraph_max_bs, self._max_cached_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token_d_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token_d_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )

            max_num_prefill_chunks = (
                cdiv(vllm_config.model_config.max_model_len, self.chunk_size)
                * self.decode_cudagraph_max_bs
            )
            self.seq_idx_p_buf = torch.empty(
                (max_num_prefill_chunks,),
                dtype=torch.int32,
                device=device,
            )
            self.cu_chunk_seqlen_p_buf = torch.empty(
                (max_num_prefill_chunks + 1,),
                dtype=torch.int32,
                device=device,
            )
            self.last_chunk_indices_p_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.num_computed_tokens_p_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_first_scheduled_token_p_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token_p_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token_p_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.state_indices_tensor_d_buf = None
            self.block_idx_last_computed_token_d_buf = None
            self.block_idx_last_scheduled_token_d_buf = None
            self.state_indices_tensor_p_buf = None
            self.seq_idx_p_buf = None
            self.cu_chunk_seqlen_p_buf = None
            self.last_chunk_indices_p_buf = None
            self.num_computed_tokens_p_buf = None
            self.block_idx_first_scheduled_token_p_buf = None
            self.block_idx_last_computed_token_p_buf = None
            self.block_idx_last_scheduled_token_p_buf = None

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
        context_lens = m.num_computed_tokens_cpu
        context_lens_tensor = context_lens.to(query_start_loc.device)
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        enable_apc = self.vllm_config.cache_config.enable_prefix_caching
        block_size_value: int | None = None
        chunk_size_value: int | None = None
        if enable_apc:
            block_size_value = self.kv_cache_spec.block_size
            chunk_size_value = self.chunk_size
        state_indices_tensor_d: torch.Tensor | None = None
        state_indices_tensor_p: torch.Tensor | None = None
        block_idx_last_computed_token_d: torch.Tensor | None = None
        block_idx_last_scheduled_token_d: torch.Tensor | None = None
        block_idx_first_scheduled_token_p: torch.Tensor | None = None
        block_idx_last_computed_token_p: torch.Tensor | None = None
        block_idx_last_scheduled_token_p: torch.Tensor | None = None
        num_computed_tokens_p: torch.Tensor | None = None
        seq_idx_p: torch.Tensor | None = None
        cu_chunk_seqlen_p: torch.Tensor | None = None
        last_chunk_indices_p: torch.Tensor | None = None
        non_spec_query_start_loc_cpu: torch.Tensor | None = None

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
            spec_sequence_masks = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = int(spec_sequence_masks.sum().item())
            if num_spec_decodes == 0:
                spec_sequence_masks = None
            else:
                spec_sequence_masks = spec_sequence_masks.to(
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
            non_spec_state_indices_tensor = m.block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = m.query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

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
                spec_state_indices_tensor = m.block_table_tensor[:, : self.num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = m.block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = m.block_table_tensor[
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
                query_lens_cpu = m.query_start_loc_cpu[1:] - m.query_start_loc_cpu[:-1]
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks.cpu()],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if enable_apc:
            block_table_tensor_full = m.block_table_tensor
            block_size = self.kv_cache_spec.block_size
            num_computed_tokens_device = m.num_computed_tokens_cpu.to(
                self.device, dtype=torch.int32
            )
            seq_lens_device = m.seq_lens.to(self.device, dtype=torch.int32)

            block_idx_last_computed_all = (
                (cdiv(num_computed_tokens_device, block_size) - 1)
                .clamp(min=0)
                .to(torch.int32)
            )
            block_idx_first_scheduled_all = (
                cdiv(num_computed_tokens_device + 1, block_size) - 1
            ).to(torch.int32)
            block_idx_last_scheduled_all = (cdiv(seq_lens_device, block_size) - 1).to(
                torch.int32
            )

            if spec_sequence_masks is not None:
                non_spec_mask = ~spec_sequence_masks
                non_spec_block_table = block_table_tensor_full[non_spec_mask]
                block_idx_last_computed_non_spec = block_idx_last_computed_all[
                    non_spec_mask
                ]
                block_idx_last_scheduled_non_spec = block_idx_last_scheduled_all[
                    non_spec_mask
                ]
                block_idx_first_scheduled_non_spec = block_idx_first_scheduled_all[
                    non_spec_mask
                ]
                num_computed_tokens_non_spec = num_computed_tokens_device[non_spec_mask]
                spec_sequence_masks_cpu = spec_sequence_masks.cpu()
                non_spec_mask_cpu = ~spec_sequence_masks_cpu
                num_computed_tokens_cpu_non_spec = m.num_computed_tokens_cpu[
                    non_spec_mask_cpu
                ]
            else:
                non_spec_block_table = block_table_tensor_full
                block_idx_last_computed_non_spec = block_idx_last_computed_all
                block_idx_last_scheduled_non_spec = block_idx_last_scheduled_all
                block_idx_first_scheduled_non_spec = block_idx_first_scheduled_all
                num_computed_tokens_non_spec = num_computed_tokens_device
                num_computed_tokens_cpu_non_spec = m.num_computed_tokens_cpu

            if num_decodes > 0:
                state_indices_tensor_d = non_spec_block_table[:num_decodes].contiguous()
                block_idx_last_computed_token_d = block_idx_last_computed_non_spec[
                    :num_decodes
                ].contiguous()
                block_idx_last_scheduled_token_d = block_idx_last_scheduled_non_spec[
                    :num_decodes
                ].contiguous()

            if num_prefills > 0:
                start = num_decodes
                end = start + num_prefills
                state_indices_tensor_p = non_spec_block_table[start:end].contiguous()
                block_idx_first_scheduled_token_p = block_idx_first_scheduled_non_spec[
                    start:end
                ].contiguous()
                block_idx_last_computed_token_p = block_idx_last_computed_non_spec[
                    start:end
                ].contiguous()
                block_idx_last_scheduled_token_p = block_idx_last_scheduled_non_spec[
                    start:end
                ].contiguous()
                num_computed_tokens_p = num_computed_tokens_non_spec[
                    start:end
                ].contiguous()

                if spec_sequence_masks is None:
                    num_computed_tokens_p_cpu = m.num_computed_tokens_cpu[
                        m.num_reqs - num_prefills :
                    ]
                    query_start_loc_p_cpu = (
                        m.query_start_loc_cpu[-num_prefills - 1 :] - num_decode_tokens
                    )
                else:
                    num_computed_tokens_p_cpu = num_computed_tokens_cpu_non_spec[
                        num_decodes:
                    ]
                    assert non_spec_query_start_loc_cpu is not None
                    query_start_loc_p_cpu = (
                        non_spec_query_start_loc_cpu[-num_prefills - 1 :]
                        - num_decode_tokens
                    )

                cu_chunk_seqlen: list[int] = []
                seq_idx_list: list[int] = []
                last_chunk_indices_list: list[int] = []
                seqlen_pos = 0

                for req_idx in range(num_prefills):
                    this_num_computed = int(num_computed_tokens_p_cpu[req_idx].item())
                    this_new_tokens = int(
                        query_start_loc_p_cpu[req_idx + 1].item()
                        - query_start_loc_p_cpu[req_idx].item()
                    )

                    if this_num_computed % self.chunk_size != 0:
                        seq_idx_list.append(req_idx)
                        cu_chunk_seqlen.append(seqlen_pos)
                        chunk_len = (
                            cdiv(this_num_computed, self.chunk_size) * self.chunk_size
                            - this_num_computed
                        )
                        chunk_len = min(chunk_len, this_new_tokens)
                        seqlen_pos += chunk_len
                        this_new_tokens -= chunk_len

                    n_chunks = cdiv(this_new_tokens, self.chunk_size)
                    for _ in range(n_chunks):
                        seq_idx_list.append(req_idx)
                        cu_chunk_seqlen.append(seqlen_pos)
                        chunk_len = min(self.chunk_size, this_new_tokens)
                        seqlen_pos += chunk_len
                        this_new_tokens -= chunk_len

                    assert this_new_tokens == 0
                    last_chunk_indices_list.append(len(cu_chunk_seqlen) - 1)

                cu_chunk_seqlen.append(seqlen_pos)

                device = query_start_loc.device
                seq_idx_p = torch.as_tensor(
                    seq_idx_list, device=device, dtype=torch.int32
                )
                cu_chunk_seqlen_p = torch.as_tensor(
                    cu_chunk_seqlen, device=device, dtype=torch.int32
                )
                last_chunk_indices_p = torch.as_tensor(
                    last_chunk_indices_list, device=device, dtype=torch.int32
                )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None
        num_actual_tokens = (
            num_prefill_tokens + num_decode_tokens + num_spec_decode_tokens
        )

        # prepare tensors for cudagraph
        #
        # With speculative decoding, the xgrammar backend may rollback tokens
        # and causing some sequences has less draft tokens than self.num_spec.
        #
        # In above cases, the max possible batch size for n tokens, can be
        # min(n, cudagraph_max_bs).
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
            batch_size = min(self.decode_cudagraph_max_bs, num_actual_tokens)

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
            num_actual_tokens = self.vllm_config.pad_for_cudagraph(m.num_actual_tokens)
            batch_size = num_actual_tokens

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

            if enable_apc and num_decodes > 0:
                assert state_indices_tensor_d is not None
                num_blocks = state_indices_tensor_d.shape[1]
                self.state_indices_tensor_d_buf[:num_decodes, :num_blocks].copy_(
                    state_indices_tensor_d, non_blocking=True
                )
                state_indices_tensor_d = self.state_indices_tensor_d_buf[
                    :batch_size, :num_blocks
                ]
                state_indices_tensor_d[num_decodes:, :].fill_(PAD_SLOT_ID)

                assert block_idx_last_scheduled_token_d is not None
                self.block_idx_last_scheduled_token_d_buf[:num_decodes].copy_(
                    block_idx_last_scheduled_token_d, non_blocking=True
                )
                block_idx_last_scheduled_token_d = (
                    self.block_idx_last_scheduled_token_d_buf[:batch_size]
                )
                block_idx_last_scheduled_token_d[num_decodes:] = 0

                assert block_idx_last_computed_token_d is not None
                self.block_idx_last_computed_token_d_buf[:num_decodes].copy_(
                    block_idx_last_computed_token_d, non_blocking=True
                )
                block_idx_last_computed_token_d = (
                    self.block_idx_last_computed_token_d_buf[:batch_size]
                )
                block_idx_last_computed_token_d[num_decodes:] = 0

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=num_actual_tokens,
            has_initial_state=has_initial_state,
            block_size=block_size_value,
            chunk_size=chunk_size_value,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            state_indices_tensor_d=state_indices_tensor_d,
            state_indices_tensor_p=state_indices_tensor_p,
            block_idx_last_computed_token_d=block_idx_last_computed_token_d,
            block_idx_last_scheduled_token_d=block_idx_last_scheduled_token_d,
            block_idx_first_scheduled_token_p=block_idx_first_scheduled_token_p,
            block_idx_last_computed_token_p=block_idx_last_computed_token_p,
            block_idx_last_scheduled_token_p=block_idx_last_scheduled_token_p,
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
        m.num_computed_tokens_cpu = m.seq_lens_cpu - num_accepted_tokens.cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
