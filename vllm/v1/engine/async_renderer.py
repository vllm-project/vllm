# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.engine.protocol import RendererClient
from vllm.plugins.io_processors import get_io_processor
from vllm.renderers import renderer_from_config
from vllm.tokenizers.protocol import TokenizerLike
from vllm.tracing import init_tracer
from vllm.v1.engine.input_processor import InputProcessor


class AsyncRenderer(RendererClient):
    """Standalone RendererClient built directly from a VllmConfig.

    Owns the renderer, io_processor, and input_processor — all CPU-only
    resources.  Does not depend on :class:`AsyncLLM` or any inference engine.
    In a disaggregated deployment this class would be replaced by a remote stub
    that talks to a dedicated renderer process over the network.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer_from_config(vllm_config)
        self.io_processor = get_io_processor(
            vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )
        self.input_processor = InputProcessor(vllm_config, self.renderer)
        self._observability_config = vllm_config.observability_config

        tracing_endpoint = self._observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

    # Client base (liveness) — renderer has no long-running background process

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        raise RuntimeError("AsyncRenderer has no error state")

    async def check_health(self) -> None:
        pass  # no background process to check

    def shutdown(self) -> None:
        self.renderer.shutdown()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return self._observability_config.otlp_traces_endpoint is not None
