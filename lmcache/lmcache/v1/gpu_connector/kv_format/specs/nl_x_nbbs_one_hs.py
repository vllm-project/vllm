# SPDX-License-Identifier: Apache-2.0
"""Per-layer MLA, fused PBS: ``NL x [PBS, 1, HS]`` (e.g. SGLang MLA).

A ``list[NL]`` of a 3-D tensor with a singleton head axis; num_blocks*block_size
is folded into the first axis (so num_blocks/block_size are absent).
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


class NL_X_NBBS_ONE_HS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS
    attention_backends = ("SGLang MLA",)

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        raise ValueError(
            f"num_blocks is undefined for the fused format {self.engine_kv_format}"
        )

    def block_size(self, layer_idx: int = 0) -> int:
        raise ValueError(
            f"block_size is undefined for the fused format {self.engine_kv_format}"
        )

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0]

    def kv_size(self) -> int:
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0].numel()

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
