# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class HSTUAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aot_schedule = False
        logger.info_once("Using HSTUAttention base FlashAttention")


class HSTUAttentionBackend(FlashAttentionBackend):
    """Use HSTU's jagged attention while retaining vLLM's attention lifecycle."""

    @staticmethod
    def get_name() -> str:
        return "HSTU_ATTN"

    @staticmethod
    def get_impl_cls() -> type["HSTUAttentionImpl"]:
        return HSTUAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["HSTUAttentionMetadataBuilder"]:
        return HSTUAttentionMetadataBuilder


class HSTUAttentionImpl(FlashAttentionImpl):
    """FlashAttentionImpl-compatible wrapper around hstu_attn_varlen_func."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "HSTU attention does not support fused output quantization."
            )
        if attn_metadata is None:
            return output.fill_(0)
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            raise NotImplementedError("HSTU attention is decoder-only.")

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        query_lens = (
            attn_metadata.query_start_loc[1:]
            - attn_metadata.query_start_loc[:-1]
        )
        # hstu_attn_varlen_func consumes the current contiguous K/V tensors.
        # A request with a non-empty historical KV cache requires HSTU's paged
        # operator, which is a separate contract and must not be approximated.

        gr_metadata = get_forward_context().additional_kwargs.get("gr_metadata")
        num_reqs = query_lens.shape[0]
        if gr_metadata is None:
            num_targets = torch.zeros(
                num_reqs, dtype=torch.int32, device=query.device
            )
        else:
            if len(gr_metadata) != num_reqs:
                raise ValueError("HSTU metadata and attention batch size differ.")
            targets = []
            for metadata in gr_metadata:
                if metadata.get("request_stage") == 1:
                    raise NotImplementedError(
                        "PD-separated HSTU decode requires hstu paged attention."
                    )
                candidate_num = metadata.get("candidate_num", [0])
                if not isinstance(candidate_num, list) or len(candidate_num) != 1:
                    raise ValueError("candidate_num must be a one-element list per request.")
                targets.append(candidate_num[0])
            num_targets = torch.tensor(targets, dtype=torch.int32, device=query.device)

        num_contexts = query_lens - num_targets
        try:
            from hstu_attn import hstu_attn_varlen_func
        except ImportError:
            from hstu_attn_interface import hstu_attn_varlen_func

        result = hstu_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_query_len,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=1,
            window_size=(-1, 0),
            rab=None,
            alpha=self.scale,
            has_drab=False,
        )

        output[:num_actual_tokens].copy_(result.view_as(output[:num_actual_tokens]))
        return output
