# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport TP all-reduce with epilogues written as EFC functions."""

from .efc import (
    DeviceScalar,
    Group128Scale,
    GroupScale,
    NVFP4Scale,
    PerToken,
    Phase,
    Row,
    Scalar,
    Streams,
    Weight,
)
from .kernel import FusedAllReduce, LamportAllReduce

__all__ = [
    "DeviceScalar",
    "FusedAllReduce",
    "Group128Scale",
    "GroupScale",
    "LamportAllReduce",
    "NVFP4Scale",
    "PerToken",
    "Phase",
    "Row",
    "Scalar",
    "Streams",
    "Weight",
]
