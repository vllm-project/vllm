import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
MODELBASEDIR="/root/.cache/huggingface/hub"

def baichuan_preprocess(prompt):
    """
    <reserved_106>美国下任总统是谁<reserved_107>不知道<reserved_106>解释一下“温故而知新”<reserved_107>
    """
    prompt_str = "<reserved_107>"
    for i, p in enumerate(reversed(prompt)):
        if p['role'] == 'user':
            content = '<reserved_106>' + p['content']
        else:
            content = '<reserved_107>' + p['content']
        prompt_str = content + prompt_str
    return prompt_str

def llama2_preprocess(prompt):
    """
    <s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST] {model_reply_1}</s><s>[INST] {user_message_2} [/INST]
    """
    prompt_str = "[/INST]"
    for i, p in enumerate(reversed(prompt)):
        if p['role'] == 'user':
            content = "<s>[INST]" + p['content']
        else:
            content = "[/INST]" + p['content'] + "</s>"
        prompt_str = content + prompt_str
    return prompt_str

def qwen_preprocess(prompt):
    """
    '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n解释一下“温故而知新”<|im_end|>\n<|im_start|>assistant\n'
    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n美国下任总统是谁<|im_end|>\n<|im_start|>assistant\n作为一个AI语言模型，我不能预测未来事件。美国总统的选举结果将取决于选民的投票和选举结果的计算。<|im_end|>\n<|im_start|>user\n解释一下“温故而知新”<|im_end|>\n<|im_start|>assistant\n
    """
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    nl = "\n"
    prompt_str = f"{nl}{im_start}user{nl}{prompt[-1]['content']}{im_end}{nl}{im_start}assistant{nl}"
    if len(prompt) > 1:
        prompt = prompt[:-1]
        for i, p in enumerate(reversed(prompt)):
            if p['role'] == 'user':
                content = f"{nl}{im_start}user{nl}{p['content']}{im_end}"
            else:
                content = f"{nl}{im_start}assistant{nl}{p['content']}{im_end}"
            prompt_str = content + prompt_str
    return prompt_str

def mixtral_preprocess(prompt):
    prompt_str = "[/INST]"
    for i, p in enumerate(reversed(prompt)):
        if p['role'] == 'user':
            content = '[INST]' + p['content']
        else:
            content = '[/INST]' + p['content']
        prompt_str = content + prompt_str
    prompt_str = '<s>' + prompt_str
    return prompt_str


MODEL_DICT = {
    "Baichuan2-13B":{
        "model_dir":f"{MODELBASEDIR}/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/74391478bce6dc10b6d1ea323aa591273de23fcd",
        "tensor_parallel_size":2,
        "temperature":0.3,
        "max_tokens":2048,
        "top_k":5,
        "top_p":0.85,
        "repetition_penalty":1.05,
        "preprocess_method":baichuan_preprocess
    },
    "GLM3-6B":{
        "model_dir":f"{MODELBASEDIR}/models--THUDM--chatglm3-6b/snapshots/37f2196f481f8989ea443be625d05f97043652ea",
        "tensor_parallel_size":1
    },
    "Llama2-13B":{
        "model_dir":f"{MODELBASEDIR}/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496",
        "tensor_parallel_size":2,
        "temperature":0.6,
        "max_tokens":4096,
        "top_p":0.9,
        "stop":"</s>", 
        "preprocess_method":llama2_preprocess
    },
    "Llama2-70B":{
        "model_dir":f"{MODELBASEDIR}/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2",
        "tensor_parallel_size":8,
        "temperature":0.6,
        "max_tokens":512,
        "top_p":0.9,
        "stop":"</s>", 
        "preprocess_method":llama2_preprocess
    },
    "Qwen-14B":{
        "model_dir":f"{MODELBASEDIR}/models--Qwen--Qwen-14B-Chat/snapshots/cdaff792392504e679496a9f386acf3c1e4333a5",
        "tensor_parallel_size":2,
        "temperature":None,
        "max_tokens":2048,
        "top_k":None,
        "top_p":0.8,
        "repetition_penalty":1.1,
        "stop":"<|im_end|>", 
        "preprocess_method":qwen_preprocess
    },
    "Qwen-72B":{
        "model_dir":f"{MODELBASEDIR}/models--Qwen--Qwen-72B-Chat/snapshots/6eb5569e56644ea662b048e029de9d093e97d4b6",
        "tensor_parallel_size":8,
        "temperature":None,
        "max_tokens":2048,
        "top_k":None,
        "top_p":0.8,
        "repetition_penalty":1.1,
        "stop":"<|im_end|>", 
        "preprocess_method":qwen_preprocess
    },
    "Mixtral-8x7B-v0.1":{
        "model_dir":f"{MODELBASEDIR}/Mixtral-8x7B-Instruct-v0.1",
        "tensor_parallel_size":8,
        "temperature":0.3,
        "max_tokens":2048,
        "top_k":5,
        "top_p":0.85,
        "repetition_penalty":1.05,
        "stop":"</s>", 
        "preprocess_method":mixtral_preprocess
    }  
}

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
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    preprocess_method = MODEL_DICT[args.modeltype].get("preprocess_method", None)
    prefix_pos = request_dict.pop("prefix_pos", None)
    stream = request_dict.pop("stream", False)
    if MODEL_DICT[args.modeltype].get("temperature", None) and request_dict.get("temperature", None) is None:
        request_dict["temperature"] = MODEL_DICT[args.modeltype].get("temperature")
    if MODEL_DICT[args.modeltype].get("max_tokens", None) and request_dict.get("max_tokens", None) is None:
        request_dict["max_tokens"] = MODEL_DICT[args.modeltype].get("max_tokens")
    if MODEL_DICT[args.modeltype].get("top_k", None) and request_dict.get("top_k", None) is None:
        request_dict["top_k"] = MODEL_DICT[args.modeltype].get("top_k")
    if MODEL_DICT[args.modeltype].get("top_p", None) and request_dict.get("top_p", None) is None:
        request_dict["top_p"] = MODEL_DICT[args.modeltype].get("top_p")
    if MODEL_DICT[args.modeltype].get("repetition_penalty", None) and request_dict.get("repetition_penalty", None) is None:
        request_dict["repetition_penalty"] = MODEL_DICT[args.modeltype].get("repetition_penalty")
    if MODEL_DICT[args.modeltype].get("stop", None) and request_dict.get("stop", None) is None:
        request_dict["stop"] = MODEL_DICT[args.modeltype].get("stop")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    if preprocess_method:
        prompt = preprocess_method(prompt)
    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id,
                                        prefix_pos=prefix_pos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            # prompt = request_output.prompt
            text_outputs = [
                output.text for output in request_output.outputs
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

    # assert final_output is not None
    # prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeltype", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model = MODEL_DICT[args.modeltype].get("model_dir", None)
    args.tensor_parallel_size = MODEL_DICT[args.modeltype].get("tensor_parallel_size", 1)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
