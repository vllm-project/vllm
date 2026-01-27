# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

from typing import Any

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config
@dataclass
class OffloadConfig:
    """Configuration for model weight offloading to CPU.

    This controls how model parameters are offloaded to CPU memory to reduce
    GPU memory usage, at the cost of additional CPU-GPU transfers during
    inference.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Advanced CPU offloading (V2): Group every N layers together. Offload last
    `offload_num_in_group` layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Advanced CPU offloading (V2): Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Advanced CPU offloading (V2): Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        if self.offload_group_size > 0:
            if self.offload_num_in_group > self.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.offload_num_in_group}) must be "
                    f"<= offload_group_size ({self.offload_group_size})"
                )
            if self.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step ({self.offload_prefetch_step}) must be "
                    f">= 1 when V2 offloading is enabled (offload_group_size > 0)"
                )
        return self

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # Offload settings don't affect the computation graph structure,
        # only the memory layout and transfer patterns.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
