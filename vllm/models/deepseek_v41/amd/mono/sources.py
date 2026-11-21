# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono layer's kernel sources, in every build's JIT cache key
(`common.plan.key_tuple`): an edit anywhere in this package rebuilds."""

from vllm.models.deepseek_v41.amd.mono.common.grid import BLOCKS
from vllm.models.deepseek_v41.amd.mono.common.plan import source_digest

# The grid is a build parameter of every kernel, not a source edit: in the key.
SOURCES = source_digest(".") + f"-b{BLOCKS}"
