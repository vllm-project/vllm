# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ragged decode metadata shared by the MiniMax M3 indexer and sparse builders.

Under adaptive verification, each decode request verifies 1 to 1 + K tokens, so
the builders pass ``query_start_loc`` and a token-to-request map to their
kernels instead of one uniform per-request query length.
"""

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    max_decode_query_len,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


class VarlenDecodeBuilderMixin:
    """Builder hooks for ragged decode; subclasses opt in via
    ``supports_varlen_decode``."""

    @classmethod
    def supports_varlen_decode(cls) -> bool:
        return False

    @classmethod
    def _use_varlen_decode(cls, vllm_config: VllmConfig) -> bool:
        spec_config = vllm_config.speculative_config
        return (
            cls.supports_varlen_decode()
            and spec_config is not None
            and spec_config.enable_adaptive_verification
        )

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        # Varlen decode replays decode-only batches of 1 to 1 + K tokens per
        # request; mixed prefill+decode batches never replay FULL graphs.
        if not cls._use_varlen_decode(vllm_config) or (
            cls.get_cudagraph_support(vllm_config, kv_cache_spec)  # type: ignore[attr-defined]
            != AttentionCGSupport.UNIFORM_BATCH
        ):
            return None
        return max_decode_query_len(vllm_config)


def build_decode_query_layout(
    common_attn_metadata: CommonAttentionMetadata,
    query_start_loc: torch.Tensor,
    num_decodes: int,
    num_decode_tokens: int,
    token_to_req_buffer: torch.Tensor,
    varlen: bool,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """Return ``(decode_query_len, query_start_loc, token_to_req_indices)``.

    A ragged batch returns query length 0 plus the device layout; a uniform
    batch returns its query length and no layout tensors.
    """
    if varlen:
        token_to_req = common_attn_metadata.token_to_req_indices(token_to_req_buffer)
        return (
            0,
            query_start_loc[: num_decodes + 1],
            token_to_req[:num_decode_tokens],
        )
    qsl_cpu = common_attn_metadata.query_start_loc_cpu
    query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
    decode_query_len = int(query_lens_cpu[0].item())
    assert decode_query_len > 0
    assert torch.all((query_lens_cpu == decode_query_len) | (query_lens_cpu == 0))
    assert num_decode_tokens == num_decodes * decode_query_len
    return decode_query_len, None, None
