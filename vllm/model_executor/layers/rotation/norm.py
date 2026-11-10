# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""RMSNorm adapter for rotation transforms."""

from __future__ import annotations

from typing import cast

import torch

from vllm.model_executor.layers.layernorm import RMSNorm

from .adapter import RotationAdapter


class PreRotationRMSNorm(RotationAdapter, RMSNorm):
    """Apply a bound rotation immediately before RMS normalization."""

    # Fused helpers bypass forward(), which would skip the rotation.
    supports_fused_allreduce_rms_norm = False

    @classmethod
    def upgrade(
        cls, original: RMSNorm, *, tag: str, inverse: bool
    ) -> PreRotationRMSNorm:
        """Add pre-rotation behavior without rebuilding the RMSNorm."""
        if type(original) is not RMSNorm:
            raise TypeError(
                "pre-rotation only supports plain RMSNorm, got "
                f"{type(original).__name__}"
            )

        # Upgrade in place to preserve parameter names and CustomOp dispatch.
        original.__class__ = cls
        upgraded = cast(PreRotationRMSNorm, original)
        upgraded.init_rotation_adapter(
            tag=tag,
            inverse=inverse,
            hidden_size=upgraded.hidden_size,
        )
        return upgraded

    def _apply_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Call the original CustomOp dispatch without re-entering this adapter.
        output = self._forward_method(hidden_states, None)
        assert isinstance(output, torch.Tensor)
        return output

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._apply_norm(self.apply_rotation(x))
        rotated_input = self.apply_rotation(x + residual)
        normalized = self._apply_norm(rotated_input)
        return normalized, rotated_input


__all__ = ["PreRotationRMSNorm"]
