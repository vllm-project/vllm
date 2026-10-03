# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import re
from collections.abc import Callable, MutableMapping, Sequence

logger = logging.getLogger(__name__)

_BATTLEMAGE_ARC_RE = re.compile(r"\barc(?:\(tm\))?(?: pro)? b\d{2,3}\b")


def is_intel_battlemage(device_name: str) -> bool:
    normalized = device_name.casefold()
    return "battlemage" in normalized or bool(_BATTLEMAGE_ARC_RE.search(normalized))


def has_multi_gpu_battlemage(device_names: Sequence[str]) -> bool:
    return len(device_names) > 1 and all(
        is_intel_battlemage(device_name) for device_name in device_names
    )


def maybe_apply_battlemage_xccl_workaround(
    *,
    get_device_count: Callable[[], int],
    get_device_name: Callable[[int], str],
    environ: MutableMapping[str, str],
) -> bool:
    """Apply a oneCCL IPC-handle workaround for multi-GPU Battlemage.

    Battlemage/Xe2 multi-GPU XCCL runs can hit stale cached Level Zero IPC
    handles when buffers are reallocated between collectives. Re-opening the
    IPC handle per collective avoids that failure mode.
    """
    if "CCL_ZE_CACHE_OPEN_IPC_HANDLES" in environ:
        return False

    device_count = get_device_count()
    if device_count <= 1:
        return False

    device_names = [get_device_name(device_id) for device_id in range(device_count)]
    if not has_multi_gpu_battlemage(device_names):
        return False

    environ["CCL_ZE_CACHE_OPEN_IPC_HANDLES"] = "0"
    logger.info(
        "XPU platform: detected multi-GPU Intel Battlemage/Xe2 setup; "
        "setting CCL_ZE_CACHE_OPEN_IPC_HANDLES=0."
    )
    return True
