# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared producer prompt-normalization contract for P/D connectors."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

from .test_mooncake_connector_hybrid_mamba import make_hybrid_gdn_kv_cache_config
from .test_moriio_hma_scheduler import restore_moriio_role as restore_moriio_role
from .utils import create_vllm_config


@pytest.mark.cpu_test
@pytest.mark.usefixtures("restore_moriio_role")
@pytest.mark.parametrize(
    "connector_name,extra_config",
    [
        ("NixlConnector", {}),
        ("NixlPushConnector", {}),
        ("MooncakeConnector", {}),
        ("MoRIIOConnector", {"read_mode": True, "host": "127.0.0.1"}),
        ("MoRIIOConnector", {"read_mode": False, "host": "127.0.0.1"}),
    ],
)
@pytest.mark.parametrize("client_marker", [None, False, True, "already truncated"])
def test_prefill_truncation_is_request_owned(
    connector_name, extra_config, client_marker
):
    """Client metadata and parallel requests cannot suppress the producer cut."""
    config = create_vllm_config(
        kv_connector=connector_name,
        kv_role="kv_producer",
        kv_connector_extra_config=extra_config,
    )
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    connector = KVConnectorFactory.create_connector(
        config,
        KVConnectorRole.SCHEDULER,
        make_hybrid_gdn_kv_cache_config(config.cache_config.block_size),
    )
    params = {"do_remote_decode": True, "_p_side_truncated": client_marker}
    sampling_params = SamplingParams(
        max_tokens=128, extra_args={"kv_transfer_params": params}
    )
    requests = [
        Request(
            request_id=f"prefill-{i}",
            prompt_token_ids=list(range(65)),
            sampling_params=sampling_params,
            pooling_params=None,
        )
        for i in range(2)
    ]

    for request in requests:
        connector.on_new_request(request)
        assert request.prompt_token_ids == list(range(64))
        assert request.num_prompt_tokens == 64
        assert request.max_tokens == 1
        connector.on_new_request(request)
        assert request.prompt_token_ids == list(range(64))
        assert request.num_prompt_tokens == 64
