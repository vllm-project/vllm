# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import httpx

from vllm.distributed.disaggregated import GenerationResponseT
from vllm.outputs import RequestOutput
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DisaggregatedRequestManager(ABC):
    priority: int = 0

    def __init__(self, vllm_config: "VllmConfig"):
        self._vllm_config = vllm_config

    @abstractmethod
    async def dispatch_request(
        self,
        request: Request,
        local_output: Optional[RequestOutput],
        client: httpx.AsyncClient,
        local_executed: bool = False
    ) -> tuple[bool, Optional[GenerationResponseT]]:
        # First flag to indicate if the request is successfully dispatched
        pass
