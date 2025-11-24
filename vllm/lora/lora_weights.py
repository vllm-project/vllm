# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence as GenericSequence
from typing import Optional

import torch
import torch.types

from vllm.lora.peft_helper import PEFTHelper
from vllm.utils.platform_utils import is_pin_memory_available


class LoRALayerWeights:
    """LoRA weights for a layer composed of two low rank matrixes."""

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alpha: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scaling: float | None = None,
    ) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_a = lora_a
        self.lora_b = lora_b

        if scaling is None:
            self.scaling = self.lora_alpha / self.rank
        else:
            self.scaling = scaling

    def optimize(self) -> "LoRALayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        if self.scaling == 1:
            return self
        self.lora_b *= self.scaling
        self.scaling = 1
        return self

    @property
    def input_dim(self) -> int:
        return self.lora_a.shape[1]

    @property
    def output_dim(self) -> int:
        return self.lora_b.shape[0]

    @property
    def is_packed(self) -> bool:
        return False

    @classmethod
    def from_config(
        cls,
        module_name: str,
        peft_helper: PEFTHelper,
    ) -> "LoRALayerWeights":
        # lora_a and lora_b are set to None for config-based construction
        return cls(
            module_name,
            peft_helper.r,
            peft_helper.lora_alpha,
            None,
            None,
            peft_helper.vllm_lora_scaling_factor,
        )

    @classmethod
    def create_dummy_lora_weights(
        cls,
        module_name: str,
        input_dim: int,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.types.Device,
    ) -> "LoRALayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        lora_a = torch.zeros(
            [rank, input_dim], dtype=dtype, device=device, pin_memory=pin_memory
        )
        lora_b = torch.zeros(
            [output_dim, rank], dtype=dtype, device=device, pin_memory=pin_memory
        )

        return cls(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=lora_a,
            lora_b=lora_b,
        )


class PackedLoRALayerWeights(LoRALayerWeights):
    """LoRA used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alphas: list[int | None],
        lora_a: list[torch.Tensor | None],
        lora_b: list[torch.Tensor | None],
        scaling: list[float] | None = None,
    ) -> None:
        super().__init__(
            module_name=module_name,
            rank=rank,
            lora_alpha=0,
            lora_a=lora_a,
            lora_b=lora_b,
            scaling=scaling,  # type: ignore
        )
        self.lora_alphas = lora_alphas
        if scaling is None:
            self.scaling = [  # type: ignore
                lora_alpha / self.rank  # type: ignore # noqa
                for lora_alpha in self.lora_alphas
            ]

    @classmethod
    def pack(
        cls, loras: GenericSequence[Optional["LoRALayerWeights"]]
    ) -> "PackedLoRALayerWeights":
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        first_lora = next(lora for lora in loras if lora is not None)
        for lora in loras:
            if lora is None:
                continue
            lora.optimize()
        rank = first_lora.rank
        module_name = first_lora.module_name
        obj = cls(
            module_name,
            rank,
            [lora.lora_alpha if lora is not None else None for lora in loras],
            [lora.lora_a if lora is not None else None for lora in loras],
            [lora.lora_b if lora is not None else None for lora in loras],
            scaling=[
                1 if lora is not None else None  # type: ignore
                for lora in loras
            ],
        )
        return obj

    def optimize(self) -> "PackedLoRALayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        for i in range(len(self.lora_b)):
            if self.scaling[i] == 1 or self.lora_b[i] is None:  # type: ignore
                continue
            self.lora_b[i] *= self.scaling[i]  # type: ignore
            self.scaling[i] = 1  # type: ignore
        return self

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True
