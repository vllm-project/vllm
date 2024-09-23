import multiprocessing as mp
from queue import Empty
from typing import Union

import vllm.envs as envs
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.executor.ray_gpu_executor import RayGPUExecutor
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

logger = init_logger(__name__)


class FastSyncLLM:

    def __init__(
        self,
        engine_args: EngineArgs,
        input_queue: mp.Queue,
        result_queue: mp.Queue,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        self.engine_args = engine_args
        self.request_counter = Counter()

        self.input_queue = input_queue
        self.result_queue = result_queue
        self.finish = False
        self.need_restart = False
        self.llm_engine: LLMEngine

    def _add_request(
        self,
        inputs: PromptType,
        params: Union[SamplingParams, PoolingParams],
        request_id: str,
    ) -> None:
        if isinstance(inputs, list):
            inputs = TokensPrompt(prompt_token_ids=inputs)
        self.llm_engine.add_request(request_id, inputs, params)

    def _poll_requests(self):
        while True:
            if not self.llm_engine.has_unfinished_requests():
                logger.info("No unfinished requests. Waiting...")
                (request_id, prompt, sampling_params) = self.input_queue.get()
                if self.need_restart and isinstance(
                        self.llm_engine.model_executor,
                        MultiprocessingGPUExecutor):
                    logger.info("Restarting worker loops")
                    for worker in self.llm_engine.model_executor.workers:
                        worker.execute_method("start_worker_execution_loop")
                    self.need_restart = False

            else:
                try:
                    (request_id, prompt,
                     sampling_params) = self.input_queue.get_nowait()
                except Empty:
                    break
            self._add_request(prompt, sampling_params, request_id)

    def run_engine(self):
        self.llm_engine = LLMEngine.from_engine_args(
            self.engine_args, usage_context=UsageContext.LLM_CLASS)
        assert not isinstance(
            self.llm_engine.model_executor,
            RayGPUExecutor), "Ray is not supported in sync openai mode"

        self.result_queue.put(("Ready", None, None))
        prompt_lens = {}
        tokens = {}  # type: ignore
        log_interval = 100
        poll_interval = envs.VLLM_SYNC_SERVER_ENGINE_STEPS_BETWEEN_POLLS
        try:
            while True:
                poll_interval -= 1
                if (self.input_queue.qsize() >=
                        envs.VLLM_SYNC_SERVER_ACCUM_REQUESTS
                        or poll_interval <= 0
                        or not self.llm_engine.has_unfinished_requests()):
                    self._poll_requests()
                    poll_interval = \
                        envs.VLLM_SYNC_SERVER_ENGINE_STEPS_BETWEEN_POLLS
                step_outputs = self.llm_engine.step()
                log_interval -= 1
                if log_interval == 0:
                    log_interval = 100
                    logger.info("Step finished. Unfinished requests: %d",
                                self.llm_engine.get_num_unfinished_requests())
                if not self.llm_engine.has_unfinished_requests():
                    logger.info("Broadcast stop")
                    broadcast_tensor_dict({}, src=0)
                    self.need_restart = True
                for output in step_outputs:
                    assert len(output.outputs) == 1  # type: ignore
                    first_out = output.outputs[0]  # type: ignore
                    stats = None
                    result = first_out.text
                    tokens[output.request_id] = tokens.get(
                        output.request_id, 0) + len(first_out.token_ids)
                    if output.prompt_token_ids is not None:
                        prompt_lens[output.request_id] = len(
                            output.prompt_token_ids)
                    if output.finished:
                        assert output.request_id in prompt_lens
                        stats = {
                            "prompt": prompt_lens[output.request_id],
                            "tokens": tokens[output.request_id],
                            "finish_reason": first_out.finish_reason,
                            "stop_reason": first_out.stop_reason,
                        }
                        del prompt_lens[output.request_id]
                    self.result_queue.put_nowait(
                        (output.request_id, result, stats))
        except Exception as e:
            logger.error("Error in run_engine: %s", e)
            raise e
