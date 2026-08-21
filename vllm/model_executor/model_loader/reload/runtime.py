# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PWAL-free reload for tensors already in serving runtime format."""

from collections.abc import Iterable

import torch

from vllm.model_executor.reload_arena import peek_reload_arena

__all__ = ["RuntimeReloadSession"]

ARENA_TENSOR_PREFIX = "@reload_arena/"


class RuntimeReloadSession:
    """Copy backend-native tensors in place without post-load processing.

    Names, shapes, dtypes, and layouts must match the receiving rank's live
    Parameter/Buffer schema. In particular, callers are responsible for TP/EP
    sharding and FP8/FP4 backend packing before starting this transaction.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._active = False
        self._loaded: set[str] = set()

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self.active:
            raise RuntimeError("Runtime reload session is already active")
        self._active = True
        self._loaded.clear()

    def _get_target(self, name: str) -> torch.Tensor:
        if name.startswith(ARENA_TENSOR_PREFIX):
            location = name.removeprefix(ARENA_TENSOR_PREFIX)
            module_name, separator, slot = location.partition(":")
            if not separator:
                raise KeyError(
                    f"Invalid runtime arena tensor name {name!r}; expected "
                    f"{ARENA_TENSOR_PREFIX}<module>:<slot>"
                )
            module = self.model.get_submodule(module_name)
            arena = peek_reload_arena(module)
            target = None if arena is None else arena.slots().get(slot)
            if target is None:
                raise KeyError(f"Runtime arena tensor {name!r} does not exist")
            return target

        try:
            return self.model.get_parameter(name)
        except AttributeError:
            try:
                return self.model.get_buffer(name)
            except AttributeError:
                module_name, separator, attr_name = name.rpartition(".")
                module = self.model.get_submodule(module_name if separator else "")
                target = getattr(module, attr_name if separator else name, None)
                if isinstance(target, torch.Tensor):
                    return target
                raise KeyError(
                    f"Runtime tensor {name!r} is not a Parameter, Buffer, "
                    "tensor attribute, or reload-arena slot"
                ) from None

    @torch.no_grad()
    def _check_new_name(self, name: str) -> None:
        if name in self._loaded:
            raise ValueError(f"Runtime tensor {name!r} was received more than once")

    def resolve_target(
        self,
        name: str,
        shape,
        dtype: torch.dtype,
        layout: torch.layout = torch.strided,
    ) -> torch.Tensor:
        """Return a validated live destination for direct transport writes."""
        if not self._active:
            raise RuntimeError("Runtime reload session is not active")
        self._check_new_name(name)
        target = self._get_target(name)
        if (
            target.shape != tuple(shape)
            or target.dtype != dtype
            or target.layout != layout
        ):
            raise ValueError(
                f"Incompatible runtime tensor {name!r}: expected "
                f"{tuple(target.shape)}/{target.dtype}/{target.layout}, got "
                f"{tuple(shape)}/{dtype}/{layout}"
            )
        return target

    def record_direct_write(self, name: str) -> None:
        """Record a successful transport write into a resolved destination."""
        if not self._active:
            raise RuntimeError("Runtime reload session is not active")
        self._check_new_name(name)
        self._loaded.add(name)

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not self._active:
            raise RuntimeError("Runtime reload session is not active")

        loaded = set()
        for name, value in weights:
            target = self.resolve_target(name, value.shape, value.dtype, value.layout)
            target.copy_(value)
            self._loaded.add(name)
            loaded.add(name)
        return loaded

    @torch.no_grad()
    def finish(self) -> set[str]:
        if not self._active:
            raise RuntimeError("Runtime reload session is not active")
        self._active = False
        loaded = set(self._loaded)
        self._loaded.clear()
        return loaded

    def abort(self) -> None:
        """Close the session; already applied in-place writes are not reverted."""
        self._active = False
        self._loaded.clear()
