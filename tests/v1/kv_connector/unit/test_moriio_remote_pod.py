# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    fill_remote_pod_from_config,
)


def test_fill_remote_pod_from_serve_config_when_router_omits_hosts():
    params = {"remote_dp_rank": 12}
    extra = {
        "remote_hosts": ["10.0.0.1", "10.0.0.2"],
        "remote_dp_size_local": 8,
    }
    fill_remote_pod_from_config(params, extra, is_producer=False)
    assert params["remote_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert params["remote_dp_size_local"] == 8
    assert params["is_request_leader"] is True


def test_router_remote_hosts_are_left_alone():
    params = {"remote_hosts": ["10.0.0.9"], "remote_dp_rank": 1}
    extra = {"remote_hosts": ["10.0.0.1"], "remote_dp_size_local": 8}
    fill_remote_pod_from_config(params, extra, is_producer=False)
    assert params["remote_hosts"] == ["10.0.0.9"]
    assert "is_request_leader" not in params
