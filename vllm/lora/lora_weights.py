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

    @classmethod
    def pack_moe(
        cls, loras: GenericSequence[Optional["LoRALayerWeights"]], module_name: str
    ) -> "PackedLoRALayerWeights":
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """

        first_lora = next(lora for lora in loras if lora is not None)
        assert first_lora is not None
        rank = first_lora.rank
        lora_alpha = first_lora.lora_alpha
        assert len(loras) % 3 == 0
        w1_lora_a_lst = []
        w2_lora_a_lst = []
        w3_lora_a_lst = []
        w1_lora_b_lst = []
        w2_lora_b_lst = []
        w3_lora_b_lst = []
        # TODO: Consider the case where some experts don't have LoRA added.
        for eid in range(len(loras) // 3):
            w1_lora = loras[eid * 3]
            w2_lora = loras[eid * 3 + 1]
            w3_lora = loras[eid * 3 + 2]
            assert w1_lora is not None
            assert w2_lora is not None
            assert w3_lora is not None

            w1_lora_a_lst.append(w1_lora.lora_a)
            w2_lora_a_lst.append(w2_lora.lora_a)
            w3_lora_a_lst.append(w3_lora.lora_a)

            w1_lora_b_lst.append(w1_lora.lora_b)
            w2_lora_b_lst.append(w2_lora.lora_b)
            w3_lora_b_lst.append(w3_lora.lora_b)

        w1_lora_a = torch.stack(w1_lora_a_lst, dim=0)  # (num_experts,rank,input_size)
        w2_lora_a = torch.stack(w2_lora_a_lst, dim=0)
        w3_lora_a = torch.stack(w3_lora_a_lst, dim=0)
        w1_lora_b = torch.stack(w1_lora_b_lst, dim=0)  # (num_experts,output_size,rank)
        w2_lora_b = torch.stack(w2_lora_b_lst, dim=0)
        w3_lora_b = torch.stack(w3_lora_b_lst, dim=0)

        obj = cls(
            module_name,
            rank,
            [lora_alpha, lora_alpha, lora_alpha],
            [w1_lora_a, w2_lora_a, w3_lora_a],
            [w1_lora_b, w2_lora_b, w3_lora_b],
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


class SharedMoELoRAWeights:
    """Pre-packed MoE LoRA weights with shared outer weights.

    For w1 (gate) and w3 (up): lora_A is shared, lora_B is per-expert
    For w2 (down): lora_A is per-expert, lora_B is shared

    This class holds weights loaded from compact checkpoint format and
    converts them to PackedLoRALayerWeights for use with set_lora().
    """

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alpha: int,
        # Shared weights (no expert dim): shape (rank, dim) or (dim, rank)
        w1_lora_a_shared: torch.Tensor,  # (rank, hidden_size)
        w3_lora_a_shared: torch.Tensor,  # (rank, hidden_size)
        w2_lora_b_shared: torch.Tensor,  # (hidden_size, rank)
        # Per-expert weights: shape (num_experts, ...)
        w1_lora_b: torch.Tensor,  # (num_experts, intermediate_size, rank)
        w3_lora_b: torch.Tensor,  # (num_experts, intermediate_size, rank)
        w2_lora_a: torch.Tensor,  # (num_experts, rank, intermediate_size)
    ) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.w1_lora_a_shared = w1_lora_a_shared
        self.w3_lora_a_shared = w3_lora_a_shared
        self.w2_lora_b_shared = w2_lora_b_shared
        self.w1_lora_b = w1_lora_b
        self.w3_lora_b = w3_lora_b
        self.w2_lora_a = w2_lora_a

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.rank

    def to_packed_format(self) -> "PackedLoRALayerWeights":
        """Convert to PackedLoRALayerWeights format for set_lora().

        Returns weights as lists [w1, w2, w3] where shared weights
        have shape (rank, dim) instead of (num_experts, rank, dim).
        The FusedMoEWithSharedOuterLoRA.set_lora() method handles
        the expansion via stride=0.
        """
        # lora_a: [w1_a, w2_a, w3_a]
        # lora_b: [w1_b, w2_b, w3_b]
        # Shared weights keep their compact shape - set_lora() handles expansion
        return PackedLoRALayerWeights(
            self.module_name,
            self.rank,
            [self.lora_alpha, self.lora_alpha, self.lora_alpha],
            [self.w1_lora_a_shared, self.w2_lora_a, self.w3_lora_a_shared],
            [self.w1_lora_b, self.w2_lora_b_shared, self.w3_lora_b],
        )
