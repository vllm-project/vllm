# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import model_validator

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config
class ShutdownConfig:
    """Configuration for shutdown behavior."""

    mode: Literal["abort", "wait"] = "abort"
    """Shutdown mode:
    - 'abort': Immediately abort all in-flight requests on SIGTERM
    - 'wait': Wait for in-flight requests to complete before shutdown
    """

    wait_timeout: int = 120
    """Maximum seconds to wait for in-flight requests in 'wait' mode."""

    @model_validator(mode="after")
    def validate_config(self) -> "ShutdownConfig":
        if self.mode not in ("abort", "wait"):
            raise ValueError(f"Invalid shutdown mode: {self.mode}")
        if self.wait_timeout < 0:
            raise ValueError(f"Invalid wait timeout: {self.wait_timeout}")
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
        # Shutdown config doesn't affect the computation graph
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
