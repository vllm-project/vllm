# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC
from contextlib import asynccontextmanager

from fastapi import Request

from vllm.config import VllmConfig

from ...pooling.typing import PoolingServeContext
from ..base.io_processor import PoolingIOProcessor


class PoolingServingObservabilityMixin(ABC):
    vllm_config: VllmConfig

    def __init__(self):
        self.observability_config = self.vllm_config.observability_config
        self.is_tracing_enabled = self.observability_config.is_tracing_enabled()
        self.is_tracing_request = self.is_tracing_enabled

    @asynccontextmanager
    async def _maybe_tracing(
        self,
        ctx: PoolingServeContext,
        io_processor: PoolingIOProcessor,
        raw_request: Request | None = None,
    ):
        yield
