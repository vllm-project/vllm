# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loader wrapper that records which expert shard each write covers.

The wrapper is installed once, when the layer is built, and stays on the
parameter for the lifetime of the model (``replace_parameter`` carries
``weight_loader`` across post-load processing). Outside a reload session it
only records the shard keys a checkpoint actually provides. Inside one it also
redirects writes into the staging slab the tracker hands out, so a unit's
shards can be quantized and committed the moment the unit is covered.
"""

import inspect
import weakref
from collections.abc import Callable

import torch

from .units import TRACKER_ATTR, ShardKey

__all__ = ["ExpertShardLoader", "install_expert_shard_loaders"]

OBSERVED_ATTR = "_reload_observed_shards"


class ExpertShardLoader:
    """Wrap an expert weight loader to record and route per-shard writes."""

    def __init__(
        self,
        layer: torch.nn.Module,
        param_name: str,
        inner: Callable,
        param: torch.nn.Parameter,
    ) -> None:
        # Weak reference: parameters are reachable from the layer, so a strong
        # reference here would keep every layer alive through its own loaders.
        self._layer = weakref.ref(layer)
        self.param_name = param_name
        self.inner = inner
        # Attributes the loader dispatches on (`quant_method`, `is_transposed`,
        # `load_full_w2`, ...). They are captured here, while the parameter is
        # still in checkpoint schema, because post-load processing replaces the
        # parameter and `replace_parameter` only carries `weight_loader` over.
        self.checkpoint_attrs = {
            name: value
            for name, value in param.__dict__.items()
            if name != "weight_loader"
        }
        self.__wrapped__ = inner
        self.__name__ = getattr(inner, "__name__", "expert_shard_loader")
        self.__doc__ = getattr(inner, "__doc__", None)
        # Preserved so callers can still detect MoE-aware loaders.
        self.supports_moe_loading = getattr(inner, "supports_moe_loading", False)
        try:
            parameters = inspect.signature(inner).parameters
            self._accepts_return_success = "return_success" in parameters
        except (TypeError, ValueError):
            self._accepts_return_success = False

    def _shard_key(
        self, layer: torch.nn.Module, shard_id: str, expert_id: int
    ) -> ShardKey | None:
        mapper = getattr(layer, "_map_global_expert_id_to_local_expert_id", None)
        if mapper is None:
            return None
        local_expert_id = mapper(expert_id)
        if local_expert_id < 0:
            # Not resident on this rank; the inner loader will reject it.
            return None
        return (self.param_name, local_expert_id, shard_id)

    def __call__(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        layer = self._layer()
        key = None if layer is None else self._shard_key(layer, shard_id, expert_id)

        tracker = None if layer is None else getattr(layer, TRACKER_ATTR, None)
        target = param
        if tracker is not None and key is not None:
            target = tracker.target(key, param)

        # Restore any loader attribute post-load processing dropped, both on
        # staging proxies and on live parameters written in place.
        for name, value in self.checkpoint_attrs.items():
            target.__dict__.setdefault(name, value)

        kwargs: dict = {
            "param": target,
            "loaded_weight": loaded_weight,
            "weight_name": weight_name,
            "shard_id": shard_id,
            "expert_id": expert_id,
        }
        if self._accepts_return_success:
            # Ask for the success flag even when the caller did not: it is the
            # signal that this shard actually landed.
            result = self.inner(**kwargs, return_success=True)
            success = result is not False
        else:
            result = self.inner(**kwargs)
            success = True

        if success and key is not None:
            if tracker is not None:
                tracker.record(key)
            else:
                observed = getattr(layer, OBSERVED_ATTR, None)
                if observed is None:
                    observed = set()
                    setattr(layer, OBSERVED_ATTR, observed)
                observed.add(key)

        if return_success:
            return success
        return None if self._accepts_return_success else result


def install_expert_shard_loaders(layer: torch.nn.Module) -> None:
    """Wrap every expert parameter's weight loader on a freshly built layer."""
    for name, param in layer._parameters.items():
        if param is None:
            continue
        loader = getattr(param, "weight_loader", None)
        if loader is None or isinstance(loader, ExpertShardLoader):
            continue
        param.weight_loader = ExpertShardLoader(layer, name, loader, param)
