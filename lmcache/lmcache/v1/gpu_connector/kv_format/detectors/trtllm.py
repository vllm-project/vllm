# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops


class TRTLLM_Detector(EngineDetector):
    engine_type = EngineType.TRTLLM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        # TRT-LLM hands a 4-D pool tensor (maybe wrapped in a 1-element list);
        # reshape its fused trailing dim into the canonical 6-D cross-layer form
        # [NB, NL, 2, num_kv_heads, tokens_per_block, head_dim].
        if isinstance(kv_caches, list) and len(kv_caches) == 1:
            kv_caches = kv_caches[0]
        if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 4:
            num_kv_heads = layout_hints.get("num_kv_heads")
            tokens_per_block = layout_hints.get("tokens_per_block")
            head_dim = layout_hints.get("head_dim")
            if num_kv_heads is None or tokens_per_block is None or head_dim is None:
                raise ValueError(
                    "TRT-LLM discovery needs layout_hints with "
                    "num_kv_heads, tokens_per_block, head_dim"
                )
            num_blocks, num_layers, kv_size, flat = kv_caches.shape
            if flat != num_kv_heads * tokens_per_block * head_dim:
                raise ValueError(
                    f"TRT-LLM 4-D flat dim {flat} != num_kv_heads ({num_kv_heads}) "
                    f"* tokens_per_block ({tokens_per_block}) * head_dim ({head_dim})"
                )
            kv_caches = kv_caches.view(
                num_blocks,
                num_layers,
                kv_size,
                num_kv_heads,
                tokens_per_block,
                head_dim,
            )

        list_depth, tensor_ndim, _first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )
        if list_depth == 0 and tensor_ndim == 6:
            return lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS, kv_caches
        return None, kv_caches
