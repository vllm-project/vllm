# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for vLLM v1 workers."""

from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_pp_group, get_tp_group)
from vllm.utils import decorate_logs, set_process_title


def setup_worker_process_title_and_logging(enable_ep: bool = False) -> None:
    """Set up worker process title and log prefix based on parallelism ranks.
    
    This function fetches parallelism info from
    vllm.distributed.parallel_state and constructs
    process title and logging prefix for debugging.
    
    Args:
        enable_ep: Whether expert parallelism is enabled
    """
    dp_size = get_dp_group().world_size
    dp_rank = get_dp_group().rank_in_group
    pp_size = get_pp_group().world_size
    pp_rank = get_pp_group().rank_in_group
    tp_size = get_tp_group().world_size
    tp_rank = get_tp_group().rank_in_group

    process_name = "Worker"
    if dp_size > 1:
        process_name += f"_DP{dp_rank}"
    if pp_size > 1:
        process_name += f"_PP{pp_rank}"
    if tp_size > 1:
        process_name += f"_TP{tp_rank}"
    if enable_ep:
        ep_rank = get_ep_group().rank_in_group
        process_name += f"_EP{ep_rank}"

    set_process_title(name=process_name)
    decorate_logs(process_name)
