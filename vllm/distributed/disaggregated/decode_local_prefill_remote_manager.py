# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import httpx

from vllm.distributed.disaggregated.request_manager import (
    DisaggregatedRequestManager)
from vllm.v1.request import Request


class DecodeLocalPrefillRemoteManager(DisaggregatedRequestManager):

    def __init__(self):
        super().__init__()
        self.prefill_clients = {}

    def dispatch_request(self, request: Request, client: httpx.AsyncClient):
        pass
