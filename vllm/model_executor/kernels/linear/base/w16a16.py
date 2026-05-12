# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import vllm.model_executor.kernels.linear.base.common as common


@dataclass
class Config(common.Config):
    weight_dtype: torch.dtype
    weight_shape: tuple[int, int]
    batch_invariant: bool = False
    is_weight_meta: bool = False
    clear_weight_after_processing: bool = True
    weight_contiguous: bool = False


class Kernel(common.Kernel[Config]):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: Config) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.utils import replace_parameter

        layer.processed_weight = layer.weight
        if self.config.clear_weight_after_processing:
            replace_parameter(layer, "weight", torch.empty(0))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return type(self).apply(x, layer.processed_weight, bias)

    @staticmethod
    def apply(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return F.linear(x, weight, bias)


def dispatch_fn(
    predicate,
    primary: type[Kernel],
    fallback_fn,
):
    def dispatch(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if predicate(x, weight, bias):
            return primary.apply(x, weight, bias)
        return fallback_fn(x, weight, bias)

    return dispatch


class PredicateKernel(common.PredicateKernel[Config], Kernel):
    """w16a16 predicate kernel. `predicate` takes (x, weight, bias)."""

    pass


class Composite(common.Composite[Config], Kernel):
    """w16a16-bound Composite. Concrete chains live in
    ``vllm/model_executor/kernels/linear/composed/``."""

    _dispatcher_fn = staticmethod(dispatch_fn)
    _native_impl = staticmethod(Kernel.apply)
