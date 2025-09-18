# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncGenerator
from contextlib import contextmanager, nullcontext
from typing import AbstractContextManager, Optional, Union

import httpx

from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import (ChatCompletionResponse,
                                              ErrorResponse)
from vllm.v1.request import Request

# TODO generateresponse
GenerationResponseT = Union[AsyncGenerator[str, None], ChatCompletionResponse,
                            ErrorResponse]  # TODO error?


class DisaggregatedServerMixin:

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self._clients = dict[str, httpx.AsyncClient]()
        # TODO Logic for enabling might be more complex and platform-dependent
        self.enabled = self.vllm_config.kv_transfer_config is not None

    @contextmanager
    def maybe_get_disaggregated_server_output(
        self,
        request: Request,
    ) -> AbstractContextManager[Optional[GenerationResponseT]]:
        if not self.enabled:
            return nullcontext()

        # TODO
        # dispatch to Manager
        # get client

    def shutdown(self):
        for client in self._clients.values():
            client.aclose()
        self._clients.clear()
