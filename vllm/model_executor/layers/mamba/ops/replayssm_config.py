# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch-config selection for the ReplaySSM decode kernels (Mamba2
output_only and GDN).

Mirrors `mamba_ssm.py`: a hard-coded heuristic per kernel, plus an
`override` context manager for benchmarks/tests/config sweeps. Hardware is
auto-detected (Blackwell vs not) so call sites need not thread it through.
"""

import functools

from vllm.platforms import current_platform
from vllm.triton_utils import triton


@functools.cache
def _is_blackwell() -> bool:
    try:
        return current_platform.is_device_capability_family(100)
    except Exception:
        return False


# Per-kernel overrides keyed by the kernel name passed to get_replayssm_config.
_overrides: dict[str, tuple] = {}


def _dstate_tile(dstate: int, tile: int) -> int:
    return max(16, min(tile, triton.next_power_of_2(dstate)))


def _mamba2_output_only(dstate, L, is_blackwell):
    # (block_size_m, num_warps, nf_dstate_tile, fl_dstate_tile, num_stages);
    # decoupled dstate tiling, serving-batch optimum, dtype-independent per device.
    if is_blackwell:
        return 64, 1, _dstate_tile(dstate, 32), _dstate_tile(dstate, 64), 2
    return 16, 1, _dstate_tile(dstate, 64), _dstate_tile(dstate, 128), 2


# Longer GDN rings spill registers and replay more than the skipped state
# store saves: on B200 the decode step is slower than the baseline from L = 64.
GDN_MAX_BUFFER_LEN = 32


def _gdn_decode(is_blackwell):
    # (block_v, num_warps, num_stages, nk); serving-batch optimum per device.
    if is_blackwell:
        return 128, 1, 3, 4
    return 64, 1, 3, 2


# GDN spec configs are (block_v, num_warps, nk, num_stages), swept on
# B300 at the Qwen3.5 geometry. The head_k_dim gate matters: these tuples
# regress badly at K=64, which the default covers instead.
_GDN_SPEC_VERIFY_BLACKWELL = {
    2: (128, 2, 4, 3),
    4: (128, 2, 4, 3),
    6: (128, 2, 8, 4),
    8: (128, 2, 8, 4),
}
_GDN_SPEC_FLUSH_BLACKWELL = {
    2: (64, 1, 8, 4),
    4: (64, 1, 8, 4),
    6: (64, 1, 8, 2),
    8: (64, 1, 8, 2),
}


def _gdn_spec_default(max_spec_len):
    return 64, 1, (4 if max_spec_len >= 6 else 2), 2


def _gdn_spec_verify(max_spec_len, head_k_dim, is_blackwell):
    if is_blackwell and head_k_dim == 128:
        return _GDN_SPEC_VERIFY_BLACKWELL.get(
            max_spec_len, _gdn_spec_default(max_spec_len)
        )
    return _gdn_spec_default(max_spec_len)


def _gdn_spec_flush(max_spec_len, head_k_dim, is_blackwell):
    if is_blackwell and head_k_dim == 128:
        return _GDN_SPEC_FLUSH_BLACKWELL.get(
            max_spec_len, _gdn_spec_default(max_spec_len)
        )
    return _gdn_spec_default(max_spec_len)


def get_replayssm_config(kernel: str, **shape) -> tuple:
    """Return the launch config for `kernel` (override > tuned default).

    Hardware is auto-detected.
    """
    if kernel in _overrides:
        return _overrides[kernel]
    bw = _is_blackwell()
    if kernel == "mamba2_output_only":
        return _mamba2_output_only(shape["dstate"], shape.get("L", 16), bw)
    if kernel == "gdn_decode":
        return _gdn_decode(bw)
    if kernel == "gdn_spec_verify":
        return _gdn_spec_verify(shape["max_spec_len"], shape["head_k_dim"], bw)
    if kernel == "gdn_spec_flush":
        return _gdn_spec_flush(shape["max_spec_len"], shape["head_k_dim"], bw)
    raise ValueError(f"unknown ReplaySSM kernel config key: {kernel}")
