# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import httpx

from vllm.distributed.disaggregated import GenerationResponseT
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DisaggregatedRequestManager(ABC):
    priority: int = 0

    def __init__(self, vllm_config: "VllmConfig"):
        self._vllm_config = vllm_config

    @abstractmethod
    def dispatch_request(
        self, request: Request, shared_http_clients: dict[str,
                                                          httpx.AsyncClient]
    ) -> tuple[bool, Optional[GenerationResponseT]]:
        # First flag to indicate if the request is successfully dispatched
        # TODO switch to GenerateRequest when available
        # TODO where is this client coming from?
        pass
