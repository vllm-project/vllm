# SPDX-License-Identifier: Apache-2.0
"""Per-layer, HND, blocks-first with fused K/V: ``NL x [NB, NH, BS, CS]``.

A ``list[NL]`` of the raw 4-D tensor an engine registers for its unified KV
cache, kept as-is. The trailing axis is the per-head content size
(``CS == 2 * head_size``, K/V packed). Produced by vLLM's non-MLA blocks-first
attention backends (HND layout).
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
import lmcache.c_ops as lmc_ops


class NL_X_NB_NH_BS_CS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_CS
    attention_backends = ("vLLM non-MLA blocks-first, fused K/V (HND)",)

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[2]

    def kv_size(self) -> int:
        # K/V stay packed in the content axis; a single fused plane.
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[1] * t.shape[3]

    def head_size(self, layer_idx: int = 0) -> int:
        # The per-head content size (2 * head_size, K/V packed).
        return self.kv_caches[layer_idx].shape[3]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[2]

    def elements_per_layer(self) -> int:
        t = self.kv_caches[0]
        return t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
