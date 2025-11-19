from vllm.config import VllmConfig
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.streaming.engine import StreamingEngineCoreOutputs


class StreamingAsyncMPClient(AsyncMPClient):
    output_cls: type[EngineCoreOutputs] = StreamingEngineCoreOutputs

    @staticmethod
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "StreamingAsyncMPClient":
        client_args = (vllm_config, executor_class, log_stats,
                       client_addresses, client_count, client_index)
        return StreamingAsyncMPClient(*client_args)
