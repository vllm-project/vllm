# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import httpx

from vllm.config import VllmConfig
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)
from vllm.outputs import RequestOutput
from vllm.v1.request import Request


class PrefillLocalDecodeRemoteManager(DisaggregatedRequestManager):
    priority = 1

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.prefill_clients = {}

    async def dispatch_request(self, request: Request,
                               local_output: Optional[RequestOutput],
                               client: httpx.AsyncClient,
                               local_executed: bool):
        kv_params = request.kv_transfer_params
        assert kv_params is not None
        if kv_params.get("do_remote_decode", False):
            # Non-disaggregated request
            return False, None

        if local_executed:
            # Local Prefill completed: open a new connection to the remote
            # decode server and stream response back to the client
            assert local_output.kv_transfer_params is not None
            # Contains P->D transfer params
            request.kv_transfer_params = local_output.kv_transfer_params

            # TODO deffered decode selection here
            host = request.kv_transfer_params["remote_host"]
            port = request.kv_transfer_params["remote_port"]
            assert host and port

            # TODO return streaming generator
            response = await client.post("", json=request)
            response.raise_for_status()
            return True, response.json()

        # Let local prefill run through, but make sure streaming is disabled
        # TODO save to figure out if streaming is needed
        request.stream = False
        request.max_tokens = 1
        request.max_completion_tokens = 1
        request.stream_options = None

        return True, None
