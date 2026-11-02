# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    fill_remote_pod_from_config,
    fold_local_rank,
    pod_index,
)

_EXTRA = {"remote_hosts": ["10.0.0.1", "10.0.0.2"], "remote_dp_size_local": 8}


def test_child_rank_uses_the_second_host():
    params = {"remote_dp_rank": 12}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params["remote_hosts"][pod_index(12, 8)] == "10.0.0.2"
    assert fold_local_rank(12, params["remote_dp_size_local"]) == 4
    assert params["is_request_leader"] is True


def test_router_hosts_are_kept():
    params = {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 12}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params == {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 12}
