# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
import torch.fx.experimental._config as fx_config
import torch.nn as nn
from torch.overrides import TorchFunctionMode, _get_current_function_mode_stack
from torch.utils._pytree import tree_leaves

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.utils.torch_utils import set_default_torch_dtype

# Ops that map a host tensor into device address space without device memory.
_UVA_VIEWS = ("get_cuda_view_from_cpu_tensor", "get_xpu_view_from_cpu_tensor")


class _HostTensors(TorchFunctionMode):
    """Tracks the tensors a real run keeps in host memory, all built on meta:
    pinned allocations, tensors derived from them (views, UVA views), and those
    model code declares with `mark_host_tensors`."""

    def __init__(self) -> None:
        super().__init__()
        self.tensors: list[torch.Tensor] = []  # keeps their storage ids valid
        self.storages: set[int] = set()

    def add(self, tensors: Iterable[object]) -> None:
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                self.tensors.append(tensor)
                self.storages.add(tensor.untyped_storage()._cdata)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("pin_memory"):
            out = func(*args, **{**kwargs, "device": "meta", "pin_memory": False})
            self.add([out])
            return out
        from_host = self.storages and any(
            isinstance(t, torch.Tensor) and t.untyped_storage()._cdata in self.storages
            for t in tree_leaves((args, kwargs))
        )
        if from_host and getattr(func, "__name__", "").split(".")[0] in _UVA_VIEWS:
            return args[0]
        out = func(*args, **kwargs)
        if from_host:
            self.add(tree_leaves(out))
        return out


# Model -> the host tensors recorded while building it.
_HOST_TENSORS: "weakref.WeakKeyDictionary[nn.Module, _HostTensors]" = (
    weakref.WeakKeyDictionary()
)


def mark_host_tensors(*tensors: torch.Tensor) -> None:
    """Declare tensors model code built on meta in place of the host tensors a
    real run allocates. Only static memory estimation reads this."""
    for mode in _get_current_function_mode_stack():
        if isinstance(mode, _HostTensors):
            mode.add(tensors)


def host_storages(model: nn.Module) -> set[int]:
    """Storage ids of the model's tensors that a real run keeps in host memory."""
    host = _HOST_TENSORS.get(model)
    return host.storages if host is not None else set()


@contextmanager
def _meta_construction() -> Iterator[_HostTensors]:
    """Create every tensor on meta, including modules moved with `.to()`."""
    original_to = nn.Module.to

    def to_meta(module: nn.Module, *args, **kwargs):
        # Redirect device moves; dtype-only moves pass through unchanged.
        if args and isinstance(args[0], (torch.device, str, int)):
            args = (torch.device("meta"), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = torch.device("meta")
        return original_to(module, *args, **kwargs)

    # Construction-time nonzero() on meta tensors has no data to size from.
    assume_nonzero = fx_config.meta_nonzero_assume_all_nonzero
    nn.Module.to = to_meta  # type: ignore[method-assign]
    fx_config.meta_nonzero_assume_all_nonzero = True
    try:
        with torch.device("meta"), _HostTensors() as host:
            yield host
    finally:
        nn.Module.to = original_to  # type: ignore[method-assign]
        fx_config.meta_nonzero_assume_all_nonzero = assume_nonzero


class MetaModelLoader(BaseModelLoader):
    """Declare the model's tensors on meta: nothing is loaded and no post-load
    processing runs, so the model cannot execute. For static memory estimation."""

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass  # Nothing to load

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype), _meta_construction() as host:
            model = self.create_model(vllm_config, model_config, prefix)
        _HOST_TENSORS[model] = host
        return model.eval()
