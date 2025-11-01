# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import AbstractContextManager, nullcontext
from typing import Optional

import httpx

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.disaggregated import GenerationResponseT
from vllm.distributed.disaggregated.factory import (
    DisaggregatedRequestManagerFactory)
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)
from vllm.outputs import RequestOutput
from vllm.v1.request import Request


class DisaggregatedServerMixin:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.client = httpx.AsyncClient()
        # TODO Logic for enabling might be more complex and platform-dependent
        self._enabled = self.vllm_config.kv_transfer_config is not None
        self.managers = list[DisaggregatedRequestManager]()
        # TODO allow different API keys for different remotes
        self._api_key = envs.VLLM_API_KEY

    def maybe_setup_disaggregated_server(self):
        if not self._enabled:
            return

        # Initialize managers, ordered by priority for dispatching
        self.managers = DisaggregatedRequestManagerFactory.\
            create_request_managers(self.vllm_config)

    async def _maybe_run_disaggregated(
        self,
        request: Request,
        local_output: Optional[RequestOutput],
        local_executed: bool,
    ) -> AbstractContextManager[Optional[GenerationResponseT]]:
        if not self._enabled or request.kv_transfer_params is None:
            return nullcontext()

        # Setup shared client
        if kv_params := request.kv_transfer_params:
            host = kv_params.get("remote_host")
            port = kv_params.get("remote_port")
            if host and port:
                # TODO set other endpoints
                self.client.base_url = f"http://{host}:{port}/v1/chat/completions"
                self.client.headers.update({
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Request-Id": request.request_id
                })

        # Dispatch to Manager with highest priority
        response = None
        for manager in self.managers:
            success, response = await manager.dispatch_request(
                request,
                local_output,
                self.client,
                local_executed=local_executed)
            if success:
                break

        return response

    async def maybe_run_disaggregated_before_local(
        self,
        request: Request,
    ) -> Optional[GenerationResponseT]:
        return await self._maybe_run_disaggregated(request, None, False)

    async def maybe_run_disaggregated_after_local(
        self,
        request: Request,
        local_output: RequestOutput,
    ) -> AbstractContextManager[Optional[GenerationResponseT]]:
        return await self._maybe_run_disaggregated(request, local_output, True)

    def shutdown(self):
        self.client.aclose()
