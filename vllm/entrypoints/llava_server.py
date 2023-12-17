import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from PIL import Image
import requests
import base64
from io import BytesIO

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llava_engine import AsyncLLaVAEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - images: a list of strings, each string is either a url or a base64 encoded image.
    - other fields: the sampling parameters (See `SamplingParams` for details).

    Currently use base64 to send file data. But it is not very efficient.
    Due to the limitation of http, it is not easy to send both file and json body in a post request.
    There are some other ways to do it, but will need to change the request format:
    https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    images = request_dict.pop("images", None)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    # decode images
    if images is None:
        images = []
    elif not isinstance(images, list):
        images = [images]
    _images = []
    for image in images:
        if isinstance(image, str):
            if image.startswith("http"):
                _images.append(Image.open(
                    requests.get(image, stream=True).raw))
            elif image.startswith("data:"):
                _images.append(
                    Image.open(BytesIO(base64.b64decode(image.split(",")[1]))))
            else:
                _images.append(Image.open(BytesIO(base64.b64decode(image))))
    if len(_images) == 0:
        _images = None

    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id,
                                        images=_images)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLaVAEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
