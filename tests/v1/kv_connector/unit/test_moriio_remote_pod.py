# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    fill_remote_pod_from_config,
)

_EXTRA = {"remote_hosts": ["10.0.0.1", "10.0.0.2"], "remote_dp_size_local": 8}


def test_serve_config_hosts_fill_the_request():
    params = {}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params["remote_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert params["remote_dp_size_local"] == 8
    assert params["is_request_leader"] is True


def test_router_hosts_are_kept():
    params = {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 12}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params == {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 12}
