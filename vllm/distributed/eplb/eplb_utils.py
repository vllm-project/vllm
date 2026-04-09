# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for EPLB (Expert Parallel Load Balancing)."""

import functools
import os
import sys

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def override_envs_for_eplb(parallel_config: ParallelConfig) -> None:
    """
    Override environment variables for EPLB when specific conditions are met.

    Args:
        parallel_config: The parallel configuration object.
    """
    is_data_parallel = parallel_config.data_parallel_size > 1
    is_eplb_enabled = parallel_config.enable_eplb
    async_eplb = parallel_config.eplb_config.use_async
    is_deepep_ll = parallel_config.all2all_backend == "deepep_low_latency"
    is_nccl_based_eplb_communicator = parallel_config.eplb_config.communicator in (
        "torch_nccl",
        "pynccl",
    )

    # Override NCCL_MAX_CTAS to avoid hangs when using async EPLB with the
    # DeepEP low-latency backend.
    #
    # The hang happens when two ranks interleave kernel launches differently
    # between NCCL collectives (used by async EPLB weight exchange) and DeepEP
    # low-latency (LL) kernels. DeepEP LL uses a cooperative launch and tries
    # to reserve a large fraction of the GPU's SMs; if those SMs are currently
    # occupied by NCCL, the DeepEP LL launch blocks until enough SMs are
    # freed.
    #
    # If rank A enters DeepEP LL in main thread while rank B is still executing
    # NCCL in async thread, rank A can block waiting for SMs, while rank B can
    # block inside NCCL waiting for rank A to participate in the collective.
    # This circular wait causes a deadlock.
    # Limiting NCCL occupancy via NCCL_MAX_CTAS leaves space for the DeepEP
    # cooperative kernel to launch and complete, breaking the deadlock.
    # See: https://github.com/deepseek-ai/DeepEP/issues/496
    if (
        is_data_parallel
        and is_eplb_enabled
        and is_deepep_ll
        and async_eplb
        and is_nccl_based_eplb_communicator
    ):
        current_value_str = os.getenv("NCCL_MAX_CTAS")

        if current_value_str and current_value_str.isdigit():
            return

        override_value = 8
        os.environ["NCCL_MAX_CTAS"] = str(override_value)
        logger.info_once(
            f"EPLB: Setting NCCL_MAX_CTAS={override_value} "
            "for expert parallel with NCCL-based EPLB communicator and "
            "deepep_low_latency backend",
            scope="global",
        )


# ---------------------------------------------------------------------------
# Formatting helpers for EPLB dump output
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _use_heat_color() -> bool:
    return (
        not os.environ.get("NO_COLOR", "")
        and os.environ.get("TERM", "") != "dumb"
        and sys.stderr.isatty()
    )


def heat_cell(text: str, val: float, vmin: float, vmax: float) -> str:
    """Wrap *text* in green-to-red ANSI color based on *val* in [vmin, vmax]."""
    if not _use_heat_color() or vmin >= vmax:
        return text
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    r, g = int(220 * t), int(220 * (1 - t))
    return f"\033[38;2;{r};{g};30m{text}\033[0m"


def human_tokens(n: float) -> str:
    """1234 -> '1234', 12345 -> '12k', 1234567 -> '1235k'."""
    v = int(round(n))
    return str(v) if v < 10_000 else f"{round(v / 1000)}k"


def compact_int_list(items: list) -> str:
    """Format mixed str/int list with run compression: [shared, 0..63, 123]."""
    if not items:
        return "[]"
    parts: list[str] = []
    rs: int | None = None
    re: int | None = None

    def _flush() -> None:
        if rs is not None:
            parts.append(str(rs) if rs == re else f"{rs}..{re}")

    for item in items:
        if isinstance(item, str):
            _flush()
            rs = re = None
            parts.append(item)
        else:
            x = int(item)
            if rs is None or re is None:
                rs = re = x
            elif x == re + 1:
                re = x
            elif x != re:
                _flush()
                rs = re = x
    _flush()
    return "[" + ", ".join(parts) + "]"
