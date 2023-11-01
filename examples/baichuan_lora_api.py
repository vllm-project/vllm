import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
# from vllm.model_executor.lora_utils import add_lora_adapter

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None
sampling_params = SamplingParams(temperature=0,
                                 top_p=1,
                                 best_of=2,
                                 top_k=-1,
                                 max_tokens=100,
                                 use_beam_search=True)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request with specific lora.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    task = request_dict.pop("task", "adapter_1")
    #sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    sampling_params.lora_id = task
    #prepare_batch(["table_rank"], engine.engine.workers[0].model)
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    #prompt = final_output.prompt
    text_outputs = [ output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    """
    run command:
       python3 baichuan_lora_api.py --model [model path] --lora-paths ./baichuan2-13b-adapter1 ./baichuan2-13b-adapter2 --adapter-names adapter1 adapter2 --trust-remote-code [--tensor-parallel-size 2]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)