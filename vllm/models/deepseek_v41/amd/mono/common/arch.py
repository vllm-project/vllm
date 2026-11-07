# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GPU architecture the mono kernels are compiled for.

FlyDSL compiles for the architecture in the ARCH environment variable, and
otherwise for the GPU it finds. The kernels were written for gfx950 (MI355X).
gfx942 (MI300X, MI325X) lacks several instructions they use: the scaled
f8f6f4 MFMA, the 16x16x32 bf16 MFMA, v_permlane16_swap / v_permlane32_swap
and the OCP FP8 conversions. Its FP8 format is FNUZ (bias 8, no negative
zero, 0x80 is NaN) instead of OCP. The helpers that differ test ``GFX942``
and take another code path there.

The kernels place one resident CTA per compute unit, so the grid width
``BLOCKS`` is the GPU's compute-unit count. MI355X and MI350X (gfx950) and the
gfx942 parts have 256 CUs; MI350P (gfx950) has 128. ``resolve_tiling`` picks
the tiling for a ``(arch, compute units)`` pair and the runner hands the device
modules the chosen ``BLOCKS`` through ``MONO_BLOCKS_ENV`` before FlyDSL traces
them (they cannot import this host module).
"""

import os

from flydsl.runtime.device import get_rocm_arch

MONO_BLOCKS_ENV = "VLLM_DSV41_MONO_BLOCKS"

# The compute-unit counts the tiling is validated for. gfx942 (MI300X, MI325X)
# and the 256-CU gfx950 parts (MI355X, MI350X) run the original 256-CTA grid.
# A grid wider than the CUs deadlocks the all-resident model and a narrower one
# leaves CUs idle, so any other count is refused.
_SUPPORTED_CUS = {
    "gfx942": (256,),
    "gfx950": (256,),
}

# MI350P (gfx950, 128 CUs, 4 XCDs) needs a 128-CTA grid, but the MoE down stage
# and the attention wo_b stage fix their task counts from the model width
# (DOWN_TASKS = HIDDEN // ROWS = 320, and d.wob_tasks likewise), and the device
# loops emit one task a CTA plus an optional second (``BLOCKS < tasks <=
# 2 * BLOCKS``). At 128 CTAs a 320-task stage needs three passes a CTA, which
# the traced loops do not emit, so the build asserts fire. Enabling 128 CTAs is
# a task-decomposition rewrite (ceil(tasks / BLOCKS) passes in the down, wob and
# woa stages, with the mailbox slot counts and epoch marks re-derived) that must
# be validated on the hardware. ``resolve_tiling`` names this when asked for it.
_NEEDS_RETUNE = {
    ("gfx950", 128): (
        "MI350P (128 CUs) needs the MoE down and attention wo_b/wo_a task "
        "decomposition re-derived for a 128-CTA grid (DOWN_TASKS=320, "
        "d.wob_tasks exceed 2*128); this is unimplemented and must be tuned on "
        "the hardware"
    ),
}

# The workgroup width stays 512 threads (8 waves of 64) on every part; only the
# grid width tracks the compute-unit count.
_THREADS = 512
_WAVES = _THREADS // 64


def target_arch() -> str:
    """The architecture FlyDSL compiles for, as FlyDSL's ROCm backend picks it."""
    arch = os.environ.get("ARCH") or get_rocm_arch()
    return arch.lower().split(":")[0]


def resolve_tiling(arch: str, cus: int) -> tuple[int, int, int]:
    """The ``(BLOCKS, THREADS, WAVES)`` tiling for a GPU.

    Args:
        arch: The gfx architecture, lowercased and without a feature suffix,
            as ``target_arch`` returns it.
        cus: The GPU's compute-unit count (``BLOCKS``: one resident CTA each).

    Returns:
        The grid width, workgroup width, and wave count the kernels build with.

    Raises:
        ValueError: The architecture, or its compute-unit count, is one the
            tiling is not validated for. A count with a known but unimplemented
            tiling (MI350P) names the work it needs.
    """
    family = arch.lower().split(":")[0]
    retune = _NEEDS_RETUNE.get((family, cus))
    if retune is not None:
        raise ValueError(retune)
    supported = _SUPPORTED_CUS.get(family)
    if supported is None:
        raise ValueError(
            f"the mono kernels have no tiling for {arch}; "
            f"validated architectures are {sorted(_SUPPORTED_CUS)}"
        )
    if cus not in supported:
        raise ValueError(
            f"the mono kernels have no tiling for {arch} with {cus} compute "
            f"units; validated counts are {sorted(supported)}"
        )
    return cus, _THREADS, _WAVES


def tiling() -> tuple[int, int, int]:
    """The tiling for this process: the runner's ``MONO_BLOCKS_ENV`` grid width,
    else the one ``resolve_tiling`` picks from the GPU. The device modules read
    the env the runner sets, since they cannot import the host platform."""
    blocks = os.environ.get(MONO_BLOCKS_ENV)
    if blocks is not None:
        return int(blocks), _THREADS, _WAVES
    from vllm.platforms import current_platform

    cus = current_platform.num_compute_units()
    return resolve_tiling(target_arch(), cus)


GFX942 = target_arch().startswith("gfx942")
