# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 DFly speculation for HYV4 Flash on ROCm."""

from torch import nn

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.models.hy_v4.nvidia.dfly_speculator import HYV4DFlySpeculator
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)

from .swa import FlashPackedSparseBackend, FlashPackedSparseImpl


class HYV4FlashDFlySpeculator(HYV4DFlySpeculator):
    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        draft = super().load_draft_model(target_model, target_attn_layer_names)
        # Draft top-k rows are not front-packed, which the stock AITER sparse
        # path requires. Rebind before the KV cache and metadata builders are
        # derived from the layers' backends.
        layers = [
            module for module in draft.modules() if isinstance(module, MLAAttention)
        ]
        for layer in layers:
            if layer.attn_backend is not ROCMAiterMLASparseBackend:
                raise NotImplementedError(
                    "HYV4 Flash DFly on ROCm requires the ROCM_AITER_MLA_SPARSE "
                    f"draft backend, got {layer.attn_backend.get_name()}"
                )
            layer.attn_backend = FlashPackedSparseBackend
            layer.impl.__class__ = FlashPackedSparseImpl  # type: ignore[assignment]
        if not layers:
            raise ValueError("HYV4 DFly draft has no MLA attention layers")
        return draft
