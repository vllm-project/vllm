# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-visible layout every MonoKernel shares with the host runtime.

Error words: one int32 per wait stage per kernel runtime, sticky. An expired wait
stores 1 into its stage's word. With TP peers it also stores 1 into its own rank's
flag in the output rank's flag array, then a release fence, then any further data.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm import envs

MONO_ABI_VERSION = 1
ERROR_WORD_BYTES = 4
FLAG_BYTES = 4
FAULT_INJECT_ENV = "VLLM_MONO_FAULT_INJECT"
UNSAFE_WARN_ENV = "VLLM_MONO_UNSAFE_WARN"


@dataclass(frozen=True)
class FaultSpec:
    """Test-only: stall ``rank`` for ``us`` microseconds at kernel step ``step``."""

    rank: int
    step: int
    us: int


def fault_spec() -> FaultSpec | None:
    """The build-time fault hook, from ``VLLM_MONO_FAULT_INJECT="rank=R,step=N,us=D"``.
    Unset: kernels generate no fault code. The values are closure constants of the
    kernel, which FlyDSL folds into its disk-cache key; ``D > 0`` keeps that key
    apart from the unset (0, 0, 0), so a later run never loads a fault build."""
    raw = envs.VLLM_MONO_FAULT_INJECT
    if not raw:
        return None
    fields = dict(kv.split("=", 1) for kv in raw.split(","))
    try:
        spec = FaultSpec(int(fields["rank"]), int(fields["step"]), int(fields["us"]))
    except (KeyError, ValueError) as error:
        raise ValueError(
            f'{FAULT_INJECT_ENV} must look like "rank=R,step=N,us=D", got {raw!r}'
        ) from error
    if spec.us <= 0:
        raise ValueError(f"{FAULT_INJECT_ENV} needs us > 0, got {raw!r}")
    return spec


def unsafe_warn() -> bool:
    return envs.VLLM_MONO_UNSAFE_WARN
