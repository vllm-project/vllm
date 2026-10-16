# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Shared contract for modules that consume a registry-owned rotation."""

from __future__ import annotations

import torch

from .base import RotationTransform


class RotationAdapter:
    """Mixin that binds and applies a rotation without owning its parameters."""

    rotation_tag: str
    inverse: bool
    rotation_hidden_size: int
    _rotation: RotationTransform | None

    def init_rotation_adapter(
        self,
        *,
        tag: str,
        inverse: bool,
        hidden_size: int,
    ) -> None:
        self.rotation_tag = tag
        self.inverse = inverse
        self.rotation_hidden_size = hidden_size
        # Bypass module registration: the registry must remain the sole
        # parameter owner and checkpoint namespace.
        object.__setattr__(self, "_rotation", None)

    @property
    def rotation(self) -> RotationTransform | None:
        return self._rotation

    def bind_rotation(self, rotation: RotationTransform) -> None:
        """Bind a shared registry entry without registering it as a child."""
        rotation_hidden_size = getattr(rotation, "hidden_size", None)
        if rotation_hidden_size != self.rotation_hidden_size:
            raise ValueError(
                f"rotation tag {self.rotation_tag} hidden_size="
                f"{rotation_hidden_size} does not match consumer hidden_size="
                f"{self.rotation_hidden_size}"
            )
        object.__setattr__(self, "_rotation", rotation)

    def apply_rotation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the bound transform in the configured direction."""
        if self._rotation is None:
            raise RuntimeError(f"rotation tag {self.rotation_tag} was not bound")
        return self._rotation(hidden_states, inverse=self.inverse)


__all__ = ["RotationAdapter"]
