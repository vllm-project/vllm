# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import httpx

from vllm.config import VllmConfig
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)
from vllm.v1.request import Request


class DecodeLocalPrefillRemoteManager(DisaggregatedRequestManager):
    priority = 1

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.prefill_clients = {}

    def dispatch_request(self, request: Request,
                         shared_http_clients: dict[str, httpx.AsyncClient],
                         local_executed: bool):
        kv_params = request.kv_transfer_params
        assert kv_params is not None
        if kv_params.get("do_remote_prefill", False):
            # Non-disaggregated request
            return False, None

        if not local_executed:
            # Send request to prefill server right away before local Decode
            # Keep connection open so that a single disconnect allows the whole
            # chain to be closed and cleaned up.
            pass
            return True, None

        # Let local decode run through
        # TODO add the computed token to the prompt
        return True, None
