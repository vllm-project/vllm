# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Composite Hadamard rotation module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.kernels.rotation.composite_hadamard import (
    composite_hadamard,
    composite_hadamard_torch,
)
from vllm.model_executor.kernels.rotation.composite_hadamard.utils import (
    resolve_hadamard_dim,
)
from vllm.platforms import current_platform

from .base import RotationTransform


class CompositeHadamardRotation(RotationTransform):
    r"""Apply ``D(hk \otimes H_P) / sqrt(N)`` or its transpose.

    Here ``D = diag(sign)``, ``N = K * P``, and ``P`` is a power of two.
    The checkpoint stores the unnormalized ``hk = H_K.T``.
    """

    def __init__(
        self,
        hidden_size: int,
        sign: torch.Tensor | None = None,
        hk: torch.Tensor | None = None,
        *,
        k: int | None = None,
        p: int | None = None,
    ) -> None:
        super().__init__(hidden_size)
        k, power_of_two_part = resolve_hadamard_dim(hidden_size, k=k, p=p)
        if (sign is None) != (hk is None):
            raise ValueError("sign and hk must both be provided or both be absent")

        if sign is None:
            # Invalid sentinels make missing checkpoint tensors fail at bind.
            sign = torch.zeros(hidden_size, dtype=torch.int8)
            hk = torch.full((k, k), float("nan"), dtype=torch.float32)
        else:
            if sign.shape != (hidden_size,) or sign.dtype != torch.int8:
                raise ValueError("sign must be a [hidden_size] int8 tensor")
            if hk is None or hk.shape != (k, k):
                raise ValueError("hk must be a [K, K] tensor")

        self.sign = nn.Parameter(
            sign.detach().to(dtype=torch.int8).contiguous(), requires_grad=False
        )
        self.hk = nn.Parameter(
            hk.detach().to(dtype=torch.float32).contiguous(), requires_grad=False
        )
        self.k = k
        self.p = power_of_two_part

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CompositeHadamardRotation:
        """Build a rotation from one ``rotation_config.rotations`` entry."""
        return cls(
            hidden_size=config["hidden_size"],
            k=config["hadamard_k"],
            p=config.get("hadamard_p"),
        )

    def validate_loaded(self, tag: str) -> None:
        sign_valid = bool(torch.all((self.sign == -1) | (self.sign == 1)).item())
        hk_valid = bool(torch.isfinite(self.hk).all().item())
        if sign_valid and hk_valid:
            return
        missing = []
        if not sign_valid:
            missing.append("sign")
        if not hk_valid:
            missing.append("hk")
        raise ValueError(
            "Rotation checkpoint parameters were not loaded for "
            f"tag {tag}: {', '.join(missing)}"
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden_size={self.hidden_size}, got {x.shape[-1]}"
            )

    def forward_native(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        self._validate_input(x)
        return composite_hadamard_torch(
            x,
            self.sign,
            self.hk,
            inverse=inverse,
            k=self.k,
            p=self.p,
        )

    def forward_cuda(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        self._validate_input(x)
        return composite_hadamard(
            x,
            self.sign.to(device=x.device),
            self.hk.to(device=x.device, dtype=torch.float32),
            inverse=inverse,
            k=self.k,
            p=self.p,
        )

    def forward(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        if x.device.type == "cpu":
            return self.forward_native(x, inverse=inverse)
        if x.device.type == "cuda":
            return self.forward_cuda(x, inverse=inverse)
        raise NotImplementedError(
            f"composite_hadamard does not support device {x.device.type!r}"
        )

    def register_warmup(self, *, inverse: bool, dtype: torch.dtype) -> None:
        if not current_platform.is_cuda() or dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return
        from vllm.model_executor.kernels.rotation.composite_hadamard.warmup import (
            COMPOSITE_HADAMARD_EPILOGUE,
        )

        COMPOSITE_HADAMARD_EPILOGUE.register_warmup(
            k=self.k,
            p=self.p,
            inverse=inverse,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, K={self.k}, P={self.p}"


__all__ = ["CompositeHadamardRotation"]
