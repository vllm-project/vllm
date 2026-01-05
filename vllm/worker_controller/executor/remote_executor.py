# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
import time
import threading
from collections import deque
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Tuple, Dict

import cloudpickle

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.executor.abstract import Executor
from vllm.v1.core.sched.output import SchedulerOutput, GrammarOutput
from vllm.v1.outputs import ModelRunnerOutput, DraftTokenIds
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
import vllm.envs as envs

logger = init_logger(__name__)


class RemoteExecutor(Executor):
    def __init__(self, vllm_config: VllmConfig, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        # Trigger model loading on the assigned workers
        # Use None timeout as loading can take time
        self.collective_rpc("load_model", args=(
            self.vllm_config,), timeout=None)
        self.output_rank = self._get_output_rank()

    def _get_output_rank(self) -> int:
        return self.parallel_config.world_size - self.parallel_config.tensor_parallel_size

    def shutdown(self):
        pass

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
        self.collective_rpc("execute_dummy_batch",
                            unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", unique_reply_rank=self.output_rank)

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: Tuple = (),
        kwargs: Dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | List[Any] | Future[Any | List[Any]]:

        kwargs = kwargs or {}
        req = (method, args, kwargs)
        # logger.info(f"RemoteExecutor sending request: {method}")
        self.request_queue.put(req)

        if non_block:
            future = Future()

            def wait_response():
                try:
                    res = self.response_queue.get(
                        timeout=timeout if timeout else None)
                    # logger.info(f"RemoteExecutor received response for: {method}")
                    final_res = res
                    if kv_output_aggregator is not None:
                        final_res = kv_output_aggregator.aggregate(
                            res, output_rank=unique_reply_rank or 0)
                    elif unique_reply_rank is not None:
                        if isinstance(res, list) and len(res) >= 1:
                            final_res = res[0]
                    # Check if the result is an Exception and raise it
                    if isinstance(final_res, Exception):
                        future.set_exception(final_res)
                    else:
                        future.set_result(final_res)
                except Exception as e:
                    logger.error(
                        f"RemoteExecutor error waiting for response for {method}: {e}")
                    future.set_exception(e)

            t = threading.Thread(target=wait_response, daemon=True)
            t.start()
            return future
        else:
            try:
                res = self.response_queue.get(
                    timeout=timeout if timeout else None)
                # logger.info(f"RemoteExecutor received response for: {method}. Type: {type(res)}")
                if kv_output_aggregator is not None:
                    result = kv_output_aggregator.aggregate(
                        res, output_rank=unique_reply_rank or 0)
                    if isinstance(result, Exception):
                        raise result
                    return result
                if unique_reply_rank is not None:
                    if isinstance(res, list) and len(res) >= 1:
                        # logger.info(f"RemoteExecutor extracting result from list for rank {unique_reply_rank}")
                        # Assuming target_ranks are sorted and correspond to indices
                        # We need to map unique_reply_rank to index?
                        # ProxyExecutor collects from target_ranks.
                        # If unique_reply_rank is the global rank, we need to know its index in target_ranks.
                        # But RemoteExecutor doesn't know target_ranks!

                        # For now, assume PP=1, TP=1, so we want index 0.
                        result = res[0]
                        if isinstance(result, Exception):
                            raise result
                        return result
                # logger.info("RemoteExecutor returning raw response")
                if isinstance(res, Exception):
                    raise res
                return res
            except Exception as e:
                logger.error(
                    f"RemoteExecutor error waiting for response for {method}: {e}")
                raise
