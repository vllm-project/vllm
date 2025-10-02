# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Optional

import httpx

from vllm.config import VllmConfig
from vllm.distributed.disaggregated import GenerationResponseT
from vllm.distributed.disaggregated.factory import (
    DisaggregatedRequestManagerFactory)
from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)
from vllm.v1.request import Request


class DisaggregatedServerMixin:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self._clients = dict[str, httpx.AsyncClient]()
        # TODO Logic for enabling might be more complex and platform-dependent
        self._enabled = self.vllm_config.kv_transfer_config is not None
        self.managers = list[DisaggregatedRequestManager]()

    def maybe_setup_disaggregated_server(self):
        if not self._enabled:
            return

        # Initialize managers, ordered by priority for dispatching
        self.managers = DisaggregatedRequestManagerFactory.\
            create_request_managers(self.vllm_config)

    @contextmanager
    def maybe_get_disaggregated_server_output(
        self,
        request: Request,
    ) -> AbstractContextManager[Optional[GenerationResponseT]]:
        if not self._enabled:
            return nullcontext()

        # Dispatch to Manager with highest priority
        for manager in self.managers:
            success, response = manager.dispatch_request(
                request, self._clients)
            if success:
                return response

        # No manager picked up the request
        return nullcontext(None)

    def shutdown(self):
        for client in self._clients.values():
            client.aclose()
        self._clients.clear()
