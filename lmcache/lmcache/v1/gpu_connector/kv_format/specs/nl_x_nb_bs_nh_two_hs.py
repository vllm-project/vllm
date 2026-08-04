# SPDX-License-Identifier: Apache-2.0
"""Per-layer, NHD, blocks-first with fused K/V: ``NL x [NB, BS, NH, 2, HS]``.

DEPRECATED: superseded by ``NL_X_NB_BS_NH_CS``, which keeps the raw 4-D
``[NB, BS, NH, 2*HS]`` registration instead of splitting it into this
canonical 5-D shape. Detection no longer produces this format.

A ``list[NL]`` of a 5-D tensor whose K/V (size-2) axis is second-to-last.
Produced by vLLM non-MLA blocks-first attention with the NHD layout.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.utils import lmcache_deprecate
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
import lmcache.c_ops as lmc_ops


class NL_X_NB_BS_NH_TWO_HS_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS
    attention_backends = ("vLLM non-MLA blocks-first, fused K/V (NHD)",)

    @lmcache_deprecate(
        "NL_X_NB_BS_NH_TWO_HS is superseded by NL_X_NB_BS_NH_CS "
        "(raw 4-D content-size layout)"
    )
    def __init__(self, kv_caches: DiscoverableKVCache) -> None:
        super().__init__(kv_caches)

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[1]

    def kv_size(self) -> int:
        # NOTE(ApostaC): for this special format, we treat it as
        # normal NHD format with fused KV, and the D-size is doubled
        # (as it's kv-packed)
        return 1

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[2] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        # NOTE(ApostaC): for this special format, we treat it as
        # normal NHD format with fused KV, and the D-size is doubled
        # (as it's kv-packed)
        return self.kv_caches[layer_idx].shape[4] * 2

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[1]

    def elements_per_layer(self) -> int:
        t = self.kv_caches[0]
        return t.shape[0] * t.shape[1] * t.shape[2] * t.shape[4] * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        # The base rendering pulls HS from head_size(), which is doubled here
        # (kv-packed D) and would double-count next to the literal 2 axis;
        # render the real tensor dims instead.
        return f"{len(self.kv_caches)} x {list(self.kv_caches[0].shape)}"
