# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any, Literal

from pydantic.dataclasses import dataclass

from vllm.config.utils import config

StructuredOutputsBackend = Literal["auto", "xgrammar", "guidance", "outlines",
                                   "lm-format-enforcer"]


@config
@dataclass
class StructuredOutputsConfig:
    """Dataclass which contains structured outputs config for the engine."""

    backend: StructuredOutputsBackend = "auto"
    """Which engine will be used for structured outputs (e.g. JSON schema,
    regex, etc) by default. With "auto", we will make opinionated choices
    based on request contents and what the backend libraries currently support,
    so the behavior is subject to change in each release."""
    disable_fallback: bool = False
    """If `True`, vLLM will not fallback to a different backend on error."""
    disable_any_whitespace: bool = False
    """If `True`, the model will not generate any whitespace during structured
    outputs. This is only supported for xgrammar and guidance backends."""
    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""
    reasoning_parser: str = ""
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""

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
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if (self.disable_any_whitespace
                and self.backend not in ("xgrammar", "guidance")):
            raise ValueError("disable_any_whitespace is only supported for "
                             "xgrammar and guidance backends.")
        if (self.disable_additional_properties and self.backend != "guidance"):
            raise ValueError("disable_additional_properties is only supported "
                             "for the guidance backend.")
