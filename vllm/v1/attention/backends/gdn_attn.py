# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
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

    has_initial_state: Optional[torch.Tensor] = None

    spec_query_start_loc: Optional[
        torch.Tensor] = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: Optional[
        torch.Tensor] = None  # shape: [batch - num_spec_decodes + 1,]

    spec_state_indices_tensor: Optional[
        torch.Tensor] = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: Optional[
        torch.Tensor] = None  # shape: [batch - num_spec_decodes,]
    spec_sequence_masks: Optional[torch.Tensor] = None  # shape: [batch,]
    spec_token_masks: Optional[
        torch.
        Tensor] = None  # shape: [num_prefill_tokens + num_decode_tokens,]
    num_accepted_tokens: Optional[torch.Tensor] = None  # shape: [batch,]


class GDNAttentionMetadataBuilder(
        AttentionMetadataBuilder[GDNAttentionMetadata]):

    cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens  # noqa: E501
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self.reorder_batch_threshold = self.num_spec + 1  # type: ignore[misc]

        self.use_full_cuda_graph = \
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_capture_size)

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_masks = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1), ),
            dtype=torch.bool,
            device=device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1, ),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1, ),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.int32,
            device=device,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        num_draft_tokens: Optional[torch.Tensor] = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        context_lens = m.num_computed_tokens_cpu
        context_lens_tensor = context_lens.to(query_start_loc.device)
        seq_lens_tensor = m.seq_lens

        if (not self.use_spec_decode or num_draft_tokens is None
                or num_draft_tokens.sum().item() == 0):
            spec_sequence_masks = None
        else:
            spec_sequence_masks = (num_draft_tokens > 0) & (
                context_lens_tensor +
                (num_draft_tokens + 1) == seq_lens_tensor)
            if spec_sequence_masks.sum().item() == 0:
                spec_sequence_masks = None

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1))
            num_spec_decodes = 0
            num_spec_decode_tokens = 0
            spec_token_masks = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = m.block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None
        else:
            num_spec_decodes = spec_sequence_masks.sum().item()
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            non_spec_query_lens = query_lens[~spec_sequence_masks]
            num_decodes = (non_spec_query_lens == 1).sum().item()
            num_prefills = non_spec_query_lens.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens.sum().item(
            ) - num_decode_tokens

            if num_prefills == 0 and num_decodes == 0:
                spec_token_masks = torch.ones(
                    (min(num_spec_decodes *
                         (self.num_spec + 1), query_start_loc[-1].item())),
                    dtype=torch.bool,
                    device=query_start_loc.device)
                spec_state_indices_tensor = m.block_table_tensor[:, :self.
                                                                 num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens)
                spec_state_indices_tensor = m.block_table_tensor[
                    spec_sequence_masks, :self.num_spec + 1]
                non_spec_state_indices_tensor = \
                    m.block_table_tensor[~spec_sequence_masks, 0]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device)
                torch.cumsum(query_lens[spec_sequence_masks],
                             dim=0,
                             out=spec_query_start_loc[1:])
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device)
                torch.cumsum(query_lens[~spec_sequence_masks],
                             dim=0,
                             out=non_spec_query_start_loc[1:])

            num_spec_decode_tokens = min(
                num_spec_decodes * (self.num_spec + 1),
                spec_token_masks.size(0))
            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
        else:
            has_initial_state = None

        # prepare tensors for cudagraph
        if (self.use_full_cuda_graph and num_prefills == 0 and num_decodes == 0
                and num_spec_decodes <= self.decode_cudagraph_max_bs
                and m.num_actual_tokens <= self.decode_cudagraph_max_bs):
            num_total_tokens = self.vllm_config.pad_for_cudagraph(
                m.num_actual_tokens)
            batch_size = num_total_tokens // (self.num_spec + 1)

            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True)
            spec_state_indices_tensor = self.spec_state_indices_tensor[:
                                                                       batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True)
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert spec_token_masks is not None
            self.spec_token_masks[:spec_token_masks.size(0)].copy_(
                spec_token_masks, non_blocking=True)
            spec_token_masks = self.spec_token_masks[:m.num_actual_tokens]
            spec_token_masks[spec_token_masks.size(0):].fill_(False)

            self.spec_query_start_loc[:num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True)
            spec_num_query_tokens = spec_query_start_loc[
                -1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[:batch_size + 1]
            spec_query_start_loc[num_spec_decodes +
                                 1:].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True)
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (self.use_full_cuda_graph and num_prefills == 0
                and num_spec_decodes == 0
                and num_decodes <= self.decode_cudagraph_max_bs):
            num_total_tokens = self.vllm_config.pad_for_cudagraph(
                m.num_actual_tokens)
            batch_size = num_total_tokens

            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True)
            non_spec_state_indices_tensor = \
                self.non_spec_state_indices_tensor[:batch_size]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[:num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True)
            non_spec_num_query_tokens = non_spec_query_start_loc[
                -1]  # type: ignore[index]
            non_spec_query_start_loc = \
                self.non_spec_query_start_loc[:batch_size + 1]
            non_spec_query_start_loc[num_decodes +
                                     1:].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_masks=spec_token_masks,
            num_accepted_tokens=num_accepted_tokens,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (m.num_reqs * (self.num_spec + 1) <= m.num_actual_tokens
                and ((m.num_reqs + 1) * (self.num_spec + 1)
                     >= m.num_actual_tokens)), \
            "GDN only supports decode-only full CUDAGraph capture. " \
            "Make sure all cudagraph capture sizes <= max_num_seq."

        num_accepted_tokens = torch.full((m.num_reqs, ),
                                         m.max_query_len,
                                         dtype=torch.int32,
                                         device=m.query_start_loc.device)
        num_drafted_tokens = torch.full((m.num_reqs, ),
                                        self.num_spec,
                                        dtype=torch.int32,
                                        device=m.query_start_loc.device)

        # Fixes query-start loc for spec-sequence-indices.
        m.query_start_loc = torch.arange(0,
                                         m.num_actual_tokens + 1,
                                         step=m.max_query_len,
                                         device=m.query_start_loc.device,
                                         dtype=torch.int32)
        m.num_computed_tokens_cpu = (m.seq_lens_cpu - torch.full(
            (m.num_reqs, ), m.max_query_len, dtype=torch.int32, device='cpu'))

        return self.build(0, m, num_accepted_tokens, num_drafted_tokens)
