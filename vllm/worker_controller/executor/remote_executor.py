# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

import vllm.envs as envs

logger = init_logger(__name__)


class RemoteExecutor(Executor):
    """
    Executor that communicates with pre-warmed workers via IPC queues.

    Performance optimizations:
    - Uses a thread pool for non-blocking responses instead of per-request threads
    - Minimal serialization overhead (queues handle pickling)
    """

    def __init__(self, vllm_config: VllmConfig, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        # Thread pool for non-blocking response handling (reuse threads)
        self._response_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="remote_executor_response"
        )
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        # Trigger model loading on the assigned workers
        # Use None timeout as loading can take time
        load_result = self.collective_rpc(
            "load_model", args=(self.vllm_config,), timeout=None
        )
        # Store timing information from workers
        if isinstance(load_result, list):
            self.model_load_timings = load_result
        elif isinstance(load_result, dict):
            self.model_load_timings = [load_result]
        else:
            self.model_load_timings = None
        self.output_rank = self._get_output_rank()

    def _get_output_rank(self) -> int:
        return (
            self.parallel_config.world_size - self.parallel_config.tensor_parallel_size
        )

    def shutdown(self):
        if hasattr(self, "_response_pool"):
            self._response_pool.shutdown(wait=False)

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)

    def determine_available_memory(self) -> List[int]:
        return self.collective_rpc("determine_available_memory")

    def determine_kv_cache_specs(self) -> List[Dict[str, Any]]:
        return self.collective_rpc("get_kv_cache_spec")

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # logger.info(f"RemoteExecutor.execute_model called. output_rank={self.output_rank}")
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            unique_reply_rank=self.output_rank,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            unique_reply_rank=self.output_rank,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        )

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def _process_response(
        self,
        res: Any,
        unique_reply_rank: int | None,
        kv_output_aggregator: KVOutputAggregator | None,
    ) -> Any:
        """Process response from workers, extracting the relevant result."""
        if kv_output_aggregator is not None:
            result = kv_output_aggregator.aggregate(
                res, output_rank=unique_reply_rank or 0
            )
            if isinstance(result, Exception):
                raise result
            return result

        if unique_reply_rank is not None and isinstance(res, list) and len(res) >= 1:
            # Extract result from the first rank (for PP=1, TP>=1)
            result = res[0]
            if isinstance(result, Exception):
                raise result
            return result

        if isinstance(res, Exception):
            raise res
        return res

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: Tuple = (),
        kwargs: Dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any | List[Any] | Future[Any | List[Any]]:
        """
        Send RPC to workers and collect response.

        Args:
            method: Method name or callable to execute on workers
            timeout: Timeout in seconds for response
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            non_block: If True, return a Future instead of blocking
            unique_reply_rank: If set, only return result from this rank
            kv_output_aggregator: Optional aggregator for KV cache outputs
        """
        kwargs = kwargs or {}
        req = (method, args, kwargs)
        self.request_queue.put(req)

        if non_block:
            # Use thread pool instead of creating new thread each time
            future = Future()

            def wait_and_process():
                try:
                    res = self.response_queue.get(timeout=timeout if timeout else None)
                    final_res = self._process_response(
                        res, unique_reply_rank, kv_output_aggregator
                    )
                    future.set_result(final_res)
                except Exception as e:
                    logger.error(
                        f"RemoteExecutor error waiting for response for {method}: {e}"
                    )
                    future.set_exception(e)

            self._response_pool.submit(wait_and_process)
            return future
        else:
            try:
                res = self.response_queue.get(timeout=timeout if timeout else None)
                return self._process_response(
                    res, unique_reply_rank, kv_output_aggregator
                )
            except Exception as e:
                logger.error(
                    f"RemoteExecutor error waiting for response for {method}: {e}"
                )
                raise
