# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono layer's kernel sources, in every build's JIT cache key
(``plan.key_tuple``): an edit to this package or to the shared AMD primitives
rebuilds."""

from pathlib import Path

from vllm.models.common.specialized_decode.amd.mono.plan import SHARED, source_digest

SOURCES = source_digest(Path(__file__).resolve().parent, SHARED)
