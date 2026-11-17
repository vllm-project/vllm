# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from typing import Self

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    enable_return_indexer_topk: bool = False
    """Capture and return sparse-attention index selections."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or `None` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    object_key_prefix: str | None = None
    """First segment of the keys of the shared objects that deliver the
    auxiliary outputs of requests with `SamplingParams.aux_session`. Must not
    be reused: choose a new one for every engine start."""

    mooncake_config_path: str | None = None
    """Mooncake configuration file of the store that holds the objects of
    `object_key_prefix`, in the format `MOONCAKE_CONFIG_PATH` names; `None`
    reads `MOONCAKE_CONFIG_PATH`. Use a master and pool separate from any
    Mooncake KV store: a KV store evicts and can be reset. Requires
    `object_key_prefix`."""

    object_hard_pin: bool = False
    """Write objects hard-pinned, so the store never evicts them: for a
    memory-only store. By default objects are not pinned, for a store whose
    master offloads to SSD on eviction: memory is a cache and the SSD tier keeps
    every object. Without that offload, an object that is not pinned is lost
    when memory evicts it."""

    object_put_stall_timeout_s: float = Field(default=600.0, gt=0)
    """Seconds object writes may be pending without a successful put before
    the engine stops. A write the store refuses for lack of memory is retried
    every 0.2 s: it may fit once the store has offloaded to SSD or a
    release has freed memory."""

    @model_validator(mode="after")
    def _check_object_key_prefix(self) -> Self:
        prefix = self.object_key_prefix
        if prefix is None and self.mooncake_config_path is not None:
            raise ValueError("mooncake_config_path requires object_key_prefix.")
        if prefix is not None and not (prefix and "/" not in prefix):
            raise ValueError("object_key_prefix must be non-empty and without '/'.")
        return self

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts or self.enable_return_indexer_topk

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        ignored_factors = {
            "max_bytes",
            "mooncake_config_path",
            "object_hard_pin",
            "object_key_prefix",
            "object_put_stall_timeout_s",
        }
        return get_compile_factors(self, ignored_factors)
