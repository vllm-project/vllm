# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming parameters for token streaming during text generation."""
from typing import Annotated

import msgspec


class StreamValidationError(ValueError):
    pass


class StreamDefaults:
    STREAM_N_DEFAULT = 1


class StreamLimits:
    STREAM_N_MIN = 1
    STREAM_N_MAX = 1024


class StreamingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        dict=True):  # type: ignore[call-arg]
    """Streaming parameters for token streaming during text generation.

    Args:
        stream_n: Number of tokens to stream at a time. Must be an integer >= 1.
            Defaults to 1.
    """

    stream_n: Annotated[int, msgspec.Meta(
        ge=StreamLimits.STREAM_N_MIN)] = (StreamDefaults.STREAM_N_DEFAULT)

    def __post_init__(self) -> None:
        if self.stream_n is None:
            self.stream_n = StreamDefaults.STREAM_N_DEFAULT
        if not isinstance(self.stream_n, int):
            raise StreamValidationError(
                f"stream_n must be an integer, got {type(self.stream_n)}.")
        if not (StreamLimits.STREAM_N_MIN <= self.stream_n <=
                StreamLimits.STREAM_N_MAX):
            raise StreamValidationError(
                f"stream_n must be between {StreamLimits.STREAM_N_MIN} and "
                f"{StreamLimits.STREAM_N_MAX}, got {self.stream_n}.")

    def __repr__(self) -> str:
        return f"StreamingParams(stream_n={self.stream_n})"