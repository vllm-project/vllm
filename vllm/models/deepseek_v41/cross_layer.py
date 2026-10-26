# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Hashable
from dataclasses import dataclass, field

import torch

from vllm.forward_context import get_forward_context

_RUNTIME_KEY = "deepseek_v41_cross_layer_runtime"


@dataclass
class DeepseekV41CrossLayerRuntime:
    """DeepSeek-V4.1 derived values shared by layers within one forward."""

    # Backend-specific decode metadata keyed by source and sparse-index
    # geometry. NVIDIA stores slot mappings and lengths; ROCm stores ragged
    # indices, an indptr, and lengths.
    decode_topk: dict[Hashable, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    # Backend-specific prefill metadata keyed by source and compressed-cache
    # geometry. Entries are invalidated together when their source runs.
    prefill_indices: dict[Hashable, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    def reuse_decode_topk(
        self, key: Hashable, is_index_source: bool
    ) -> tuple[torch.Tensor, ...] | None:
        """Invalidate on a source or return the matching consumer result."""
        if is_index_source:
            self.decode_topk.clear()
            return None
        return self.decode_topk.get(key)

    def cache_decode_topk(self, key: Hashable, value: tuple[torch.Tensor, ...]) -> None:
        self.decode_topk[key] = value

    def reuse_prefill_indices(
        self, key: Hashable, is_index_source: bool
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Invalidate on a source or return the matching consumer result."""
        if is_index_source:
            self.prefill_indices.clear()
            return None
        return self.prefill_indices.get(key)

    def cache_prefill_indices(
        self,
        key: Hashable,
        value: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.prefill_indices[key] = value


def get_cross_layer_runtime() -> DeepseekV41CrossLayerRuntime:
    """Return the cross-layer runtime for the current forward context."""
    additional_kwargs = get_forward_context().additional_kwargs
    runtime = additional_kwargs.get(_RUNTIME_KEY)
    if runtime is None:
        runtime = DeepseekV41CrossLayerRuntime()
        additional_kwargs[_RUNTIME_KEY] = runtime
    assert isinstance(runtime, DeepseekV41CrossLayerRuntime)
    return runtime
