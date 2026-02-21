# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config


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

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
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
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because OffloaderV2 patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        # OffloaderV2 (offload_group_size > 0) patches module forwards
        # and inserts custom ops (wait_prefetch, start_prefetch) into the
        # computation graph, so all offload settings must be part of the
        # cache key to avoid stale compilation cache hits.
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
