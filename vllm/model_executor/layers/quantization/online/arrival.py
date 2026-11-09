# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization of each checkpoint shard as it arrives, for schemes whose
quantized values and scales depend only on the rows they belong to."""

import torch

__all__ = ["OnlineArrivalMixin"]


class OnlineArrivalMixin:
    """Lets a direct reload quantize a staged module's weight shard by shard.

    A scheme opts in per parameter with `arrival_row_group`: every output of
    `quantize_param` along dim -2 then depends only on its own group of that
    many input rows (all columns), so quantizing rows apart gives the bytes
    quantizing the whole weight gives. `process_weights_after_loading` gets
    its quantized weights from `quantized_param`, which returns what the
    update quantized on arrival, if anything.
    """

    def arrival_row_group(self, layer: torch.nn.Module, name: str) -> int | None:
        """Rows of parameter `name` quantized together, or None if its
        quantization needs the whole tensor (a per-tensor or global scale)."""
        return None

    def reduces_across_ranks(self, layer: torch.nn.Module, name: str) -> bool:
        """Whether the scale of `name` is reduced across ranks. Such a
        parameter is not quantized on arrival (once per shard would be a
        collective per load); `quantize_param` reduces through
        `reduce_amax`, which decides with this."""
        return False

    def reduce_amax(
        self, layer: torch.nn.Module, name: str, amax: torch.Tensor
    ) -> torch.Tensor:
        """`amax` of `name`, reduced across ranks if `reduces_across_ranks`."""
        if self.reduces_across_ranks(layer, name):
            amax = self.all_reduce_amax(amax)
        return amax

    def all_reduce_amax(self, amax: torch.Tensor) -> torch.Tensor:
        """Max-reduce `amax` over the ranks that shard the weight."""
        raise NotImplementedError

    def quantize_param(
        self, layer: torch.nn.Module, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Quantize `weight`, the high-precision parameter `name` or a
        group-aligned range of its rows (dim -2), with all its columns. Each
        output keeps the leading dims; its dim -2 holds one row per group of
        input rows that output's row covers."""
        raise NotImplementedError

    def quantized_param(
        self, layer: torch.nn.Module, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """`quantize_param(weight)`, unless an update already quantized
        `name` on arrival."""
        done = layer.__dict__.get("_quantized_on_arrival", {}).pop(name, None)
        if done is not None:
            return done
        return self.quantize_param(layer, name, weight)
