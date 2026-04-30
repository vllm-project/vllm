# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from typing_extensions import Self

from vllm import envs


@dataclass
class Config:
    input_dtype: torch.dtype
    weight_shape: tuple[int, int]
    batch_invariant: bool = False


class Kernel(ABC):
    @classmethod
    def try_select(
        cls,
        config: Config,
        compute_capability: int | None = None,
    ) -> tuple[type[Self] | None, str | None]:
        name = cls.get_name()
        if name in envs.VLLM_DISABLED_KERNELS:
            return None, f"{name} is disabled"
        ok, reason = cls.is_supported(compute_capability)
        if not ok:
            return None, reason
        ok, reason = cls.can_implement(config)
        if not ok:
            return None, reason
        return cls, None

    @classmethod
    def get_name(cls) -> str:
        module = cls.__module__.removeprefix("vllm.model_executor.kernels.")
        return f"{module}.{cls.__name__}"

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]: ...

    @classmethod
    def can_implement(cls, config: Config) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
