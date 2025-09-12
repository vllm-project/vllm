# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.multimodal.cache import ShmObjectStoreReceiverCache
from vllm.v1.core.sched.output import SchedulerOutput


def get_and_update_mm_cache(
    receiver_cache: ShmObjectStoreReceiverCache,
    args: tuple[SchedulerOutput],
) -> None:
    """
    For each MultiModalKwargsItem in SchedulerOutput, fetch from shared memory
    cache as needed.

    Args:
        receiver_cache: The receiver cache to update.
        args: According to the collective_rpc call of execute_model method in
            executor, args is a tuple of only one SchedulerOutput element.
    """
    scheduler_output = args[0]
    for request_data in scheduler_output.scheduled_new_reqs:
        request_data.mm_features = receiver_cache.get_and_update_features(
            request_data.mm_features)
