# SPDX-License-Identifier: Apache-2.0
"""Two-list MHA, split NB/BS: ``2 x NL x [NB, BS, NH, HS]`` (SGLang MP daemon).

``[K_layers, V_layers]``, each a ``list[NL]`` of a 4-D tensor that keeps
num_blocks and block_size as separate axes.
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


class TWO_X_NL_X_NB_BS_NH_HS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
    attention_backends = ("SGLang MHA via MP daemon (4-D inner)",)

    def num_layers(self) -> int:
        return len(self.kv_caches[0])

    def num_blocks(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][0].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        inner = self.kv_caches[0][layer_idx]
        return inner.shape[2] * inner.shape[3]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[-1]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[0][layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        k, v = cast(list[list[torch.Tensor]], self.kv_caches)
        return [k[i].data_ptr() for i in layer_indices] + [
            v[i].data_ptr() for i in layer_indices
        ]
