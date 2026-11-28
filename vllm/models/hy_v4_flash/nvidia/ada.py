# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash's learnable sinks on the SM89 Triton sparse MLA backend."""

from typing import Any

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
    TritonMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd


def apply_attention_sink(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> None:
    """Add a zero-value sink per head to the natural-log softmax normalizer."""
    weights = torch.sigmoid(lse - sinks).masked_fill_(torch.isneginf(lse), 0)
    out.mul_(weights.unsqueeze(-1))


class HYV4FlashAdaMetadataBuilder(TritonMLASparseMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        assert layer.topk_buffer_width is not None
        self.topk_tokens = layer.topk_buffer_width


class HYV4FlashAdaImpl(TritonMLASparseImpl):
    supports_dense_mha_prefill = False

    def __init__(
        self,
        *args: Any,
        sliding_window: int | None = None,
        sinks: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        # The model's window indexer already supplies only the causal window.
        super().__init__(*args, sliding_window=None, **kwargs)
        if sinks.dtype != torch.float32 or sinks.shape != (self.num_heads,):
            raise ValueError("HY4 Flash requires one FP32 attention sink per head")
        self.sinks = sinks

    def _forward_mqa_kernel(
        self,
        q,
        kv_cache,
        topk_indices,
        valid_counts,
        *,
        layer,
        block_size,
        is_decode,
    ):
        q_tensor = torch.cat(q, dim=-1) if isinstance(q, tuple) else q
        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)
        out, lse = triton_mla_sparse_fwd(
            q_tensor,
            kv_rows,
            topk_indices,
            sm_scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            return_lse=True,
        )
        assert lse is not None
        apply_attention_sink(out, lse, self.sinks)
        return (
            out,
            torch.logaddexp(lse, self.sinks)
            if self.need_to_return_lse_for_decode
            else None,
        )


class HYV4FlashAdaBackend(TritonMLASparseBackend):
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 9)

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_dcp(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls():
        return HYV4FlashAdaMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return HYV4FlashAdaImpl


class HYV4FlashAdaSWABackend(HYV4FlashAdaBackend):
    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True
