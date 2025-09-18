# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import httpx

from vllm.v1.request import Request


class DisaggregatedRequestManager(ABC):

    @abstractmethod
    def dispatch_request(self, request: Request,
                         prefill_client: httpx.AsyncClient,
                         decode_client: httpx.AsyncClient):
        # TODO switch to GenerateRequest when available
        # TODO where is this client coming from?
        pass