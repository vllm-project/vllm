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

        logging.info("model initialized")

    def execute(self, requests):
        for request in requests:
            self.process_request(request)
        return None

    def process_request(self, request):
        thread = threading.Thread(
            target=asyncio.run,
            args=(self.response_thread(request.get_response_sender(), request),),
        )
        thread.daemon = True
        thread.start()

    async def response_thread(self, response_sender, request):
        prompt = pb_utils.get_input_tensor_by_name(request, "prompt")
        max_tokens = pb_utils.get_input_tensor_by_name(request, "max_tokens")

        req = [ele.decode() for ele in prompt.as_numpy()]
        sampling_params = SamplingParams(
            max_tokens=max_tokens.as_numpy().item(),
        )

        results_generator = self.engine.generate(req[0], sampling_params, random_uuid())

        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        assert final_output is not None

        output_tensor = pb_utils.Tensor(
            "response",
            np.array([final_output.outputs[0].text], dtype=object),
        )

        inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
        response_sender.send(inference_response)
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def finalize(self):
        logging.info("model finalized")
