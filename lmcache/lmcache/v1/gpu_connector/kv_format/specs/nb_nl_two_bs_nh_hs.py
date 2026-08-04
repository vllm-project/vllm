# SPDX-License-Identifier: Apache-2.0
"""Cross-layer, NHD: a single bare tensor ``[NB, NL, 2, BS, NH, HS]``.

All layers are packed along dim-1. Produced e.g. by vLLM CROSS_LAYER.
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


class NB_NL_TWO_BS_NH_HS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS
    attention_backends = ("vLLM CROSS_LAYER",)

    def num_layers(self) -> int:
        return self.kv_caches.shape[1]

    def num_blocks(self) -> int:
        return self.kv_caches.shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3]

    def page_buffer_size(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[3]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4] * self.kv_caches.shape[5]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[5]

    def tokens_per_layer(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[3]

    def elements_per_layer(self) -> int:
        t = self.kv_caches
        return t.shape[0] * 2 * t.shape[3] * t.shape[4] * t.shape[5]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches.dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        tensor = cast(torch.Tensor, self.kv_caches)
        return [tensor.data_ptr()]
