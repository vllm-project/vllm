# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch-config selection for the ReplaySSM decode kernels.

Mirrors ``mamba_ssm.py``: a hard-coded heuristic per kernel, plus an
``override`` context manager for benchmarks/tests/config sweeps. Hardware is
auto-detected (Blackwell vs not) so call sites need not thread it through.

Config tuples are tuned from the ablation bestcfg sweeps at batch >= 256 in the
deployed spec regime (spec window T >= 4); use ``override_replayssm_config`` to
pin a different config.
"""

import functools
from contextlib import contextmanager

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


@contextmanager
def override_replayssm_config(kernel: str, config: tuple):
    """Pin ``kernel``'s launch config for the duration of the context."""
    prev = _overrides.get(kernel)
    _overrides[kernel] = config
    try:
        yield
    finally:
        if prev is None:
            _overrides.pop(kernel, None)
        else:
            _overrides[kernel] = prev


def _dstate_tile(dstate: int, tile: int) -> int:
    return max(16, min(tile, triton.next_power_of_2(dstate)))


def _mamba2_spec_verify(dstate, base_block, max_spec_len, is_blackwell):
    # (block_size_m, num_warps, dstate_tile, num_stages)
    bsm = 64 if (is_blackwell or base_block <= 16) else 32
    return bsm, 2, _dstate_tile(dstate, 64), 2


def _mamba2_spec_flush(dstate, base_block, max_spec_len, is_blackwell):
    if base_block <= 16:
        return 32, 2, _dstate_tile(dstate, 64), 2
    if is_blackwell:
        return 64, 2, _dstate_tile(dstate, 128), 2
    return 32, 1, _dstate_tile(dstate, 128), 2


def _mamba2_output_only(dstate, L, is_blackwell):
    # (block_size_m, num_warps, nf_dstate_tile, fl_dstate_tile, num_stages);
    # decoupled dstate tiling, serving-batch optimum, dtype-independent per device.
    if is_blackwell:
        return 64, 1, _dstate_tile(dstate, 32), _dstate_tile(dstate, 64), 2
    return 16, 1, _dstate_tile(dstate, 64), _dstate_tile(dstate, 128), 2


# (block_v, num_warps, nk, num_stages); verify and flush are tuned
# independently per spec window on Blackwell. Non-Blackwell and untuned T
# fall back to the shared default.
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


def _l_bucket(cache_len: int) -> int:
    """Map an arbitrary buffer length to the nearest tuned bucket:
    L<=8 -> 8, 8<L<=16 -> 16, L>16 -> 32. Default callers pass L=16."""
    if cache_len <= 8:
        return 8
    if cache_len <= 16:
        return 16
    return 32


# state_and_output stays un-tiled (retained only for precision experiments); the
# output_only route uses the decoupled dstate-tiled config in _mamba2_output_only.
_STATE_AND_OUTPUT_BY_L = {8: (32, 1), 16: (32, 1), 32: (32, 1)}


def _gdn_decode(L, is_blackwell):
    if is_blackwell:
        return 128, 1, 3, 4
    return 64, 1, 3, 2


def get_replayssm_config(kernel: str, **shape) -> tuple:
    """Return the launch config for ``kernel`` (override > tuned default).

    kernel: one of "mamba2_spec_verify", "mamba2_spec_flush",
    "mamba2_output_only", "mamba2_state_and_output", "gdn_decode",
    "gdn_spec_verify", "gdn_spec_flush". ``shape`` carries the keying dims
    (dstate / base_block / max_spec_len for spec; L for standard decode,
    default 16), hardware is auto-detected.
    """
    if kernel in _overrides:
        return _overrides[kernel]
    bw = _is_blackwell()
    if kernel == "mamba2_spec_verify":
        return _mamba2_spec_verify(
            shape["dstate"], shape["base_block"], shape["max_spec_len"], bw
        )
    if kernel == "mamba2_spec_flush":
        return _mamba2_spec_flush(
            shape["dstate"], shape["base_block"], shape["max_spec_len"], bw
        )
    if kernel == "mamba2_output_only":
        return _mamba2_output_only(shape["dstate"], shape.get("L", 16), bw)
    if kernel == "mamba2_state_and_output":
        return _STATE_AND_OUTPUT_BY_L[_l_bucket(shape.get("L", 16))]
    if kernel == "gdn_decode":
        return _gdn_decode(shape.get("L", 16), bw)
    if kernel == "gdn_spec_verify":
        return _gdn_spec_verify(shape["max_spec_len"], shape["head_k_dim"], bw)
    if kernel == "gdn_spec_flush":
        return _gdn_spec_flush(shape["max_spec_len"], shape["head_k_dim"], bw)
    raise ValueError(f"unknown ReplaySSM kernel config key: {kernel}")
