# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def check_local_engines_offload_shm(
    vllm_config: "VllmConfig", local_engine_count: int
) -> None:
    """Warn when the CPU KV offload buffers of the engines about to start on
    this host may not fit in /dev/shm together.

    Each engine checks free space only for its own region, and ftruncate
    reserves nothing, so engines started together all see the same free space
    and can exhaust it while pre-faulting. A single engine is left to its own
    check, and specs that keep their buffer out of /dev/shm are skipped.

    The size this can compute is an upper bound: it is the configured budget,
    while CPUOffloadingSpec rounds each engine's region down to whole chunks,
    and the chunk size is not known until an engine has its KV cache spec. So
    this logs a warning instead of refusing the launch, like the cgroup
    preflight that check_shm_free_space runs.
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
    from vllm.utils.cpu_resource_utils import check_cgroup_memory_available
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    spec_cls = OffloadingSpecFactory.get_spec_cls(extra_config)
    uses_shared_region = getattr(spec_cls, "_uses_shared_region", None)
    if uses_shared_region is None or not uses_shared_region():
        return
    required_bytes = cpu_bytes * local_engine_count
    allocation_name = (
        f"CPU KV offload buffers of {local_engine_count} data-parallel "
        f"engines ({cpu_bytes / (1 << 30):.1f} GiB each, "
        "set by --kv-offloading-size)"
    )
    try:
        check_shm_free_space(required_bytes, allocation_name=allocation_name)
    except RuntimeError as err:
        logger.warning(
            "%s Each engine rounds its region down to whole chunks, so each "
            "region may be up to one chunk smaller than its budget. The "
            "engines will still start; if /dev/shm runs out, the engine that "
            "cannot allocate its region reports it.",
            err,
        )
        # check_shm_free_space stops before its own cgroup preflight when it
        # raises, so log that headroom here, where it is most useful.
        check_cgroup_memory_available(required_bytes, allocation_name)
