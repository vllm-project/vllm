# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator configs that only the V1 model runner needs.

These cover speculative methods that the V2 model runner doesn't implement.
They will be removed together with the V1 model runner.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

logger = init_logger(__name__)


class UnsupportedSpeculatorConfig(SpeculatorConfig):
    """A speculative method that the V2 model runner doesn't implement."""


class SuffixSpeculatorConfig(UnsupportedSpeculatorConfig):
    base_method = "suffix"

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        return "suffix"

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        spec._validate_suffix_decoding()


class V1CustomProposerConfig(UnsupportedSpeculatorConfig):
    """A V1 custom proposer, loaded from the module path in `model`."""

    base_method = "custom_class"

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        raise ValueError(
            "method='custom_class' requires 'model' to contain the custom "
            "proposer module path (e.g., 'my_module.MyProposer')."
        )

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        logger.warning_once(
            "Using a custom class-based proposer backend. This is an "
            "experimental feature and the proposer interface is subject to "
            "breaking changes in future vLLM releases."
        )
        super().configure(spec)
