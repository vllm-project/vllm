import triton_python_backend_utils as pb_utils
import os
import logging
import numpy as np
import threading
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import argparse
import asyncio


class TritonPythonModel:
    def initialize(self, args):
        parser = argparse.ArgumentParser()
        parser = AsyncEngineArgs.add_cli_args(parser)

        model_path = os.path.join(args["model_repository"], args["model_version"])
        args = parser.parse_args()
        args.model = model_path

        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self._loop = asyncio.get_event_loop()
        self._loop_thread = threading.Thread(
            target=self.engine_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._loop_thread.start()

        logging.info("model initialized")

    def engine_loop(self, loop):
        asyncio.set_event_loop(loop)
        self._loop.run_until_complete(self.await_shutdown())

    async def await_shutdown(self):
        while self._shutdown_event.is_set() is False:
            await asyncio.sleep(5)
        logging.info("shutdown complete")

    def execute(self, requests):
        for request in requests:
            self.create_task(self.generate(request))
        return None

    def create_task(self, coro):
        assert (
            self._shutdown_event.is_set() is False
        ), "cannot create tasks after shutdown has been requested"
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def create_response(self, vllm_output):
        text_outputs = [(output.text).encode("utf-8") for output in vllm_output.outputs]
        triton_output_tensor = pb_utils.Tensor(
            "response", np.asarray(text_outputs, dtype=object)
        )
        return pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])

    async def generate(self, request):
        response_sender = request.get_response_sender()

        prompt = pb_utils.get_input_tensor_by_name(request, "prompt")
        max_tokens = pb_utils.get_input_tensor_by_name(request, "max_tokens")

        req = [ele.decode() for ele in prompt.as_numpy()]
        sampling_params = SamplingParams(
            max_tokens=max_tokens.as_numpy().item(),
        )

        last_output = None
        async for output in self.engine.generate(
            req[0], sampling_params, random_uuid()
        ):
            last_output = output

        response_sender.send(self.create_response(last_output))
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        logging.info("model finalized")

        self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join()
            self._loop_thread = None
