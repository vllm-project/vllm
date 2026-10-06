# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch configs of the SketchSSM Mamba-2 Triton kernels.

Mirrors ``replayssm_config.py``: a hard-coded config per kernel, plus
``_overrides`` for benchmarks and config sweeps. The configs were chosen on an
RTX PRO 6000 Blackwell (sm_120); architectures that tune differently get their
own branch here.
"""

# Per-kernel overrides keyed by the kernel name passed to get_sketchssm_config.
_overrides: dict[str, tuple] = {}


def get_sketchssm_config(kernel: str) -> tuple:
    """Return the launch config for ``kernel`` (override > default).

    Args:
        kernel: ``"mamba2_decode"``: ``(BLOCK_M, BLOCK_J, num_warps)`` of the
            non-flush decode; ``"mamba2_flush"``: ``(BLOCK_M, num_warps, rows
            per program, BLOCK_V)`` of the flush with its sketch rebuild;
            ``"mamba2_build"``: ``(BLOCK_V, num_warps)`` of the sketch build
            after a prefill.

    """
    if kernel in _overrides:
        return _overrides[kernel]
    if kernel == "mamba2_decode":
        return 128, 16, 1
    if kernel == "mamba2_flush":
        return 16, 4, 4, 32
    if kernel == "mamba2_build":
        return 32, 2
    raise ValueError(f"unknown SketchSSM kernel config key: {kernel}")
