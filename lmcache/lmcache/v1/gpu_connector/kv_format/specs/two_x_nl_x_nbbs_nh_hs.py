# SPDX-License-Identifier: Apache-2.0
"""Two-list MHA, fused PBS: ``2 x NL x [PBS, NH, HS]`` (e.g. SGLang MHA).

``[K_layers, V_layers]``, each a ``list[NL]`` of a 3-D tensor that folds
num_blocks*block_size into one PBS axis (so num_blocks/block_size are absent).
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


class TWO_X_NL_X_NBBS_NH_HS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
    attention_backends = ("SGLang MHA (flash attention and flash infer)",)

    def num_layers(self) -> int:
        return len(self.kv_caches[0])

    def num_blocks(self) -> int:
        raise ValueError(
            f"num_blocks is undefined for the fused format {self.engine_kv_format}"
        )

    def block_size(self, layer_idx: int = 0) -> int:
        raise ValueError(
            f"block_size is undefined for the fused format {self.engine_kv_format}"
        )

    def page_buffer_size(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        inner = self.kv_caches[0][layer_idx]
        return inner.shape[1] * inner.shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[-1]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[0][layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        k, v = cast(list[list[torch.Tensor]], self.kv_caches)
        return [k[i].data_ptr() for i in layer_indices] + [
            v[i].data_ptr() for i in layer_indices
        ]
