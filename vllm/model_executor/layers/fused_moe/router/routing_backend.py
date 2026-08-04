# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform extension interface for fused MoE routers."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TypeVar, cast

from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname

from .fused_moe_router import FusedMoERouter

RouterT = TypeVar("RouterT", bound=FusedMoERouter)


class FusedMoERoutingBackend(ABC):
    """Resolve upstream-selected routers to platform-specific subclasses."""

    @classmethod
    @abstractmethod
    def resolve_router_cls(cls, router_cls: type[RouterT]) -> type[RouterT]:
        """Return ``router_cls`` or a platform-specific subclass."""
        raise NotImplementedError


@functools.cache
def resolve_fused_moe_routing_backend_cls() -> type[FusedMoERoutingBackend] | None:
    """Resolve and validate the current Platform's routing backend class."""
    backend_cls_qualname = current_platform.get_fused_moe_routing_backend_cls()
    if backend_cls_qualname is None:
        return None

    backend_cls = resolve_obj_by_qualname(backend_cls_qualname)
    if not isinstance(backend_cls, type) or not issubclass(
        backend_cls, FusedMoERoutingBackend
    ):
        raise TypeError(
            f"Fused MoE routing backend {backend_cls_qualname!r} must subclass "
            "FusedMoERoutingBackend."
        )
    return backend_cls


@functools.cache
def resolve_fused_moe_router_cls(router_cls: type[RouterT]) -> type[RouterT]:
    """Resolve the Platform subclass for an upstream-selected Router class."""
    backend_cls = resolve_fused_moe_routing_backend_cls()
    if backend_cls is None:
        return router_cls

    platform_router_cls = backend_cls.resolve_router_cls(router_cls)
    if not isinstance(platform_router_cls, type) or not issubclass(
        platform_router_cls, router_cls
    ):
        raise TypeError(
            f"Fused MoE routing backend must resolve {router_cls.__name__} to "
            "that class or one of its subclasses."
        )
    return cast(type[RouterT], platform_router_cls)
