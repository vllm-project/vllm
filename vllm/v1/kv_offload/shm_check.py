# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def check_local_engines_offload_shm(
    vllm_config: "VllmConfig", local_engine_count: int
) -> None:
    """Check that the CPU KV offload buffers of the engines about to start on
    this host fit in /dev/shm together.

    Each engine checks free space only for its own region, and ftruncate
    reserves nothing, so engines started together all see the same free space
    and can exhaust it while pre-faulting. A single engine is left to its own
    check, and specs that keep their buffer out of /dev/shm are skipped.
    """
    if local_engine_count < 2:
        return
    ktc = vllm_config.kv_transfer_config
    if ktc is None or ktc.kv_connector != "OffloadingConnector":
        return
    extra_config = ktc.kv_connector_extra_config
    cpu_bytes = int(extra_config.get("cpu_bytes_to_use") or 0)
    if not cpu_bytes:
        return

    from vllm.distributed.device_communicators.shm_broadcast import (
        check_shm_free_space,
    )
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    spec_cls = OffloadingSpecFactory.get_spec_cls(extra_config)
    uses_shared_region = getattr(spec_cls, "_uses_shared_region", None)
    if uses_shared_region is None or not uses_shared_region():
        return
    check_shm_free_space(
        cpu_bytes * local_engine_count,
        allocation_name=(
            f"CPU KV offload buffers of {local_engine_count} data-parallel "
            f"engines ({cpu_bytes / (1 << 30):.1f} GiB each, "
            "set by --kv-offloading-size)"
        ),
    )
