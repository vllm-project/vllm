# SPDX-License-Identifier: Apache-2.0
import hashlib
from typing import Any, Literal, cast, get_args

from pydantic.dataclasses import dataclass
from typing_extensions import deprecated

import vllm.envs as envs
from vllm.config.utils import config

GuidedDecodingBackendV0 = Literal["auto", "outlines", "lm-format-enforcer",
                                  "xgrammar", "guidance"]
GuidedDecodingBackendV1 = Literal["auto", "xgrammar", "guidance"]
GuidedDecodingBackend = Literal[GuidedDecodingBackendV0,
                                GuidedDecodingBackendV1]


@config
@dataclass
class DecodingConfig:
    """Dataclass which contains the decoding strategy of the engine."""

    @property
    @deprecated(
        "`guided_decoding_backend` is deprecated and has been renamed to "
        "`backend`. This will be removed in v0.10.0. Please use the "
        "`backend` argument instead.")
    def guided_decoding_backend(self) -> GuidedDecodingBackend:
        return self.backend

    @guided_decoding_backend.setter
    def guided_decoding_backend(self, value: GuidedDecodingBackend):
        self.backend = value

    backend: GuidedDecodingBackend = "auto" if envs.VLLM_USE_V1 else "xgrammar"
    """Which engine will be used for guided decoding (JSON schema / regex etc)
    by default. With "auto", we will make opinionated choices based on request
    contents and what the backend libraries currently support, so the behavior
    is subject to change in each release."""

    disable_fallback: bool = False
    """If `True`, vLLM will not fallback to a different backend on error."""

    disable_any_whitespace: bool = False
    """If `True`, the model will not generate any whitespace during guided
    decoding. This is only supported for xgrammar and guidance backends."""

    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""

    reasoning_backend: str = ""
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
        if ":" in self.backend:
            self._extract_backend_options()

        if envs.VLLM_USE_V1:
            valid_guided_backends = get_args(GuidedDecodingBackendV1)
        else:
            valid_guided_backends = get_args(GuidedDecodingBackendV0)
        if self.backend not in valid_guided_backends:
            raise ValueError(f"Invalid backend '{self.backend}',"
                             f" must be one of {valid_guided_backends}")
        if (self.disable_any_whitespace
                and self.backend not in ("xgrammar", "guidance")):
            raise ValueError("disable_any_whitespace is only supported for "
                             "xgrammar and guidance backends.")
        if (self.disable_additional_properties and self.backend != "guidance"):
            raise ValueError("disable_additional_properties is only supported "
                             "for the guidance backend.")

    @deprecated(
        "Passing guided decoding backend options inside backend in the format "
        "'backend:...' is deprecated. This will be removed in v0.10.0. Please "
        "use the dedicated arguments '--disable-fallback', "
        "'--disable-any-whitespace' and '--disable-additional-properties' "
        "instead.")
    def _extract_backend_options(self):
        """Extract backend options from the backend string."""
        backend, options = self.backend.split(":")
        self.backend = cast(GuidedDecodingBackend, backend)
        options_set = set(options.strip().split(","))
        if "no-fallback" in options_set:
            self.disable_fallback = True
        if "disable-any-whitespace" in options_set:
            self.disable_any_whitespace = True
        if "no-additional-properties" in options_set:
            self.disable_additional_properties = True
