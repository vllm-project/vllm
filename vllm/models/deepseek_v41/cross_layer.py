# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.forward_context import get_forward_context

_RUNTIME_KEY = "deepseek_v41_cross_layer_runtime"


@dataclass
class DeepseekV41CrossLayerRuntime:
    """DeepSeek-V4.1 derived values shared by layers within one forward."""

    # `(index_source_layer_id, compress_ratio)` identifying the source layer
    # whose local top-k built the cached global KV-slot mapping.
    decode_topk_key: tuple[int, int] | None = None
    # `(global_kv_slots, valid_lengths)` for decode. An index source replaces
    # this slot; its consumers reuse it without rerunning the mapping kernel.
    decode_topk: tuple[torch.Tensor, torch.Tensor] | None = None
    # Source and compression ratio identifying the top-k buffer used to build
    # the cached combined top-k/SWA indices.
    prefill_indices_key: tuple[int | None, int] | None = None
    # `(combined_indices, combined_lengths)` for prefill. Like the decode slot,
    # this is replaced by each source and reused by its consumers.
    prefill_indices: tuple[torch.Tensor, torch.Tensor] | None = None

    def reuse_decode_topk(
        self, key: tuple[int, int], is_index_source: bool
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Invalidate on a source or return the matching consumer result."""
        if is_index_source:
            self.decode_topk_key = None
            self.decode_topk = None
            return None
        if self.decode_topk_key == key:
            assert self.decode_topk is not None
            return self.decode_topk
        return None

    def cache_decode_topk(
        self, key: tuple[int, int], value: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self.decode_topk_key = key
        self.decode_topk = value

    def reuse_prefill_indices(
        self, key: tuple[int | None, int], is_index_source: bool
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Invalidate on a source or return the matching consumer result."""
        if is_index_source:
            self.prefill_indices_key = None
            self.prefill_indices = None
            return None
        if self.prefill_indices_key == key:
            assert self.prefill_indices is not None
            return self.prefill_indices
        return None

    def cache_prefill_indices(
        self,
        key: tuple[int | None, int],
        value: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.prefill_indices_key = key
        self.prefill_indices = value


def get_cross_layer_runtime() -> DeepseekV41CrossLayerRuntime:
    """Return the cross-layer runtime for the current forward context."""
    additional_kwargs = get_forward_context().additional_kwargs
    runtime = additional_kwargs.get(_RUNTIME_KEY)
    if runtime is None:
        runtime = DeepseekV41CrossLayerRuntime()
        additional_kwargs[_RUNTIME_KEY] = runtime
    assert isinstance(runtime, DeepseekV41CrossLayerRuntime)
    return runtime
