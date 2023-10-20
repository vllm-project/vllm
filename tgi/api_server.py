## TODO: Needs rework add n_beam, response and requests types etc

import argparse, sys
import json
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

# Run without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
# from tgi.model_types import Request

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

def tgi_to_vllm_params(tgi_params):
    """Convert TGI parameters to VLLM parameters."""
    mapping = {
        'do_sample': 'use_beam_search',
        'max_new_tokens': 'max_tokens',
        'repetition_penalty': 'presence_penalty',  # This mapping might not be correct, adjust according to your requirements
        'temperature': 'temperature',
        'top_k': 'top_k',
        'top_p': 'top_p',
        'typical_p': None,  # Not present in VLLM, ignore
        'best_of': 'best_of',
        'watermark': None,  # Not present in VLLM, ignore
        'details': None,  # Not present in VLLM, ignore
        'decoder_input_details': None,  # Not present in VLLM, ignore
        'top_n_tokens': None  # Not present in VLLM, ignore
    }
    return {mapping[key]: value for key, value in tgi_params.items() if mapping[key] is not None}

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    # request_data = await request.json()
    # print(request_data)
    # prompt = request.inputs
    # stream = request.stream
    # parameters = request.parameters.dict() if request.parameters else {}
    request_data = await request.json()
    print(request_data)
    
    # Correct way to access data
    prompt = request_data['inputs']
    # default stream True if not present
    if 'stream' in request_data:
        stream = request_data['stream']
    else:
        stream = True
    parameters = request_data['parameters'] if 'parameters' in request_data else {}

    vllm_params = tgi_to_vllm_params(parameters)
    sampling_params = SamplingParams(**vllm_params)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[str, None]:
        ## TODO: Handle n-beams here 
        # previous_texts = [""] * n beams
        previous_texts = [""]
        previous_num_tokens = [0]
        async for request_output in results_generator:
            for output in request_output.outputs:
                i = output.index
                token_text = output.text[len(previous_texts[i]):]
                # token_text = output.text
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                ## TODO: Retrieve this
                # token = {
                #     "id": output.token_id,
                #     "logprob": output.logprob,
                #     "special": output.is_special,
                #     "text": token_text
                # }
                token = {
                    "text": token_text
                }
                top_tokens = [token]  # Assuming the output token is the top token
                ret = {
                    "details": None,
                    "generated_text": None,
                    "token": token,
                    "top_tokens": top_tokens,
                    "finish_reason":None
                }
                if output.finish_reason is None:
                    yield f"data:{json.dumps(ret)}\n\n"
                else:
                    ret = {
                    "details": None,
                    "generated_text": None,
                    "token": token,
                    "top_tokens": top_tokens,
                    "finish_reason":output.finish_reason
                }
                    yield f"data:{json.dumps(ret)}\n\n"
                    ret = {
                    "details": None,
                    "generated_text": output.text,
                    "token": token,
                    "top_tokens": top_tokens,
                    "finish_reason":output.finish_reason
                }
                    yield f"data:{json.dumps(ret)}\n\n"   
                    
    if stream:
        return StreamingResponse(stream_results(), media_type="text/event-stream")

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
    new_generated = ''.join([output.text for output in final_output.outputs])
    ret = {"text": text_outputs, "new_generated": new_generated}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
