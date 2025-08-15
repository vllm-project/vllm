# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.device_communicators.shm_object_storage import (
    SingleWriterShmObjectStorage)
from vllm.v1.core.sched.output import SchedulerOutput


def get_and_update_mm_cache(
    object_storage: SingleWriterShmObjectStorage,
    args: tuple,
) -> None:
    """Check if the first argument is a SchedulerOutput and update
        MultiModalKwargs from the object storage if needed."""
    if args and isinstance(args[0], SchedulerOutput):
        scheduler_output = args[0]
        for request_data in scheduler_output.scheduled_new_reqs:
            for i in range(len(request_data.mm_kwargs)):
                mm_input = request_data.mm_kwargs[i]
                if "address" in mm_input:
                    address, monotonic_id = \
                        mm_input["address"].data, mm_input["monotonic_id"].data
                    request_data.mm_kwargs[i] = \
                        object_storage.get(address, monotonic_id)
