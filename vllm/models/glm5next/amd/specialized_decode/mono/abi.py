# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-visible layout every MonoKernel shares with the host runtime.

Error words: one int32 per wait stage per kernel runtime, sticky. An expired wait
stores 1 into its stage's word. With TP peers it also stores 1 into its own rank's
flag in the output rank's flag array, then a release fence, then any further data.
"""

from __future__ import annotations

from dataclasses import dataclass

MONO_ABI_VERSION = 1
ERROR_WORD_BYTES = 4
FLAG_BYTES = 4


@dataclass(frozen=True)
class FaultSpec:
    """Test-only: stall ``rank`` for ``us`` microseconds at kernel step ``step``.
    The values are closure constants of the kernel, which FlyDSL folds into its
    disk-cache key; ``us > 0`` keeps that key apart from a build without a fault
    (0, 0, 0), so a later run never loads a fault build."""

    rank: int
    step: int
    us: int

    def __post_init__(self) -> None:
        if self.us <= 0:
            raise ValueError(f"a fault stall needs us > 0, got {self.us}")
