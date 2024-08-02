import asyncio
import multiprocessing
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import Mount
from prometheus_client import make_asgi_app

from vllm import FastSyncLLM as LLM
from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.sync_openai.protocol import (CompletionRequest,
                                                   CompletionResponse,
                                                   CompletionResponseChoice,
                                                   UsageInfo)
from vllm.logger import init_logger
from vllm.utils import random_uuid

mp = multiprocessing.get_context(envs.VLLM_WORKER_MULTIPROC_METHOD)

logger = init_logger("api_server.py")


def put_in_queue(queue, item, loop):
    try:
        asyncio.run_coroutine_threadsafe(queue.put(item), loop)
    except Exception as e:
        logger.error("Exception in put_in_queue: %s", e)
        raise e


class BackgroundRunner:

    def __init__(self):
        self.value = 0
        self.engine_args = None
        self.input_queue: multiprocessing.Queue = mp.Queue()
        self.result_queue: multiprocessing.Queue = mp.Queue()
        self.result_queues: Dict[str, asyncio.Queue] = {}
        self.t: threading.Thread = threading.Thread(target=self.thread_proc)
        self.loop = None
        self.llm: LLM
        self.proc: multiprocessing.Process

    def set_engine_args(self, engine_args):
        self.engine_args = engine_args

    def add_result_queue(self, id, queue):
        self.result_queues[id] = queue

    def remove_result_queues(self, ids):
        for id in ids:
            assert id in self.result_queues
            del self.result_queues[id]
        logger.debug("Removed result queue from %d ids. %d remaining",
                     len(ids), len(self.result_queues))

    def thread_proc(self):
        while True:
            req_id, result, stats = self.result_queue.get()
            put_in_queue(self.result_queues[req_id], (req_id, result, stats),
                         self.loop)

    async def run_main(self):
        self.llm = LLM(
            engine_args=self.engine_args,
            input_queue=self.input_queue,
            result_queue=self.result_queue,
        )
        self.loop = asyncio.get_event_loop()
        self.proc = mp.Process(target=self.llm.run_engine)
        self.t.start()
        self.proc.start()

    async def add_request(self, prompt, sampling_params):
        result_queue: asyncio.Queue = asyncio.Queue()
        ids = []
        if isinstance(prompt, str) or (isinstance(prompt, list)
                                       and isinstance(prompt[0], int)):
            prompt = [prompt]
        for p in prompt:
            id = random_uuid()
            self.add_result_queue(id, result_queue)
            self.input_queue.put_nowait((id, p, sampling_params))
            ids.append(id)
        return ids, result_queue


runner = BackgroundRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner.result_queues["Ready"] = asyncio.Queue()
    asyncio.create_task(runner.run_main())
    await runner.result_queues["Ready"].get()
    del runner.result_queues["Ready"]
    yield


app = FastAPI(lifespan=lifespan)

# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


async def completion_generator(model, result_queue, choices, created_time,
                               ids):
    completed = 0
    try:
        while True:
            request_id, token, stats = await result_queue.get()

            choice_idx = choices[request_id]
            res = CompletionResponse(id=request_id,
                                     created=created_time,
                                     model=model,
                                     choices=[
                                         CompletionResponseChoice(
                                             index=choice_idx,
                                             text=token,
                                             logprobs=None,
                                             finish_reason=None,
                                             stop_reason=None)
                                     ],
                                     usage=None)
            if stats is not None:
                res.usage = UsageInfo()
                res.usage.completion_tokens = stats.get("tokens", 0)
                res.usage.prompt_tokens = stats.get("prompt", 0)
                res.usage.total_tokens = (res.usage.completion_tokens +
                                          res.usage.prompt_tokens)
                res.choices[0].finish_reason = stats["finish_reason"]
                res.choices[0].stop_reason = stats["stop_reason"]
                completed += 1
            response_json = res.model_dump_json(exclude_unset=True)
            yield f"data: {response_json}\n\n"
            if completed == len(choices):
                runner.remove_result_queues(ids)
                break

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("Error in completion_generator: %s", e)
    return


@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw_request: Request):
    sampling_params = request.to_sampling_params()
    ids, result_queue = await runner.add_request(request.prompt,
                                                 sampling_params)
    res = CompletionResponse(model=request.model,
                             choices=[],
                             usage=UsageInfo(prompt_tokens=0,
                                             total_tokens=0,
                                             completion_tokens=0))
    choices = {}
    for i, id in enumerate(ids):
        res.choices.append(
            CompletionResponseChoice(index=i,
                                     text="",
                                     finish_reason=None,
                                     stop_reason=None))
        choices[id] = i
    completed = 0
    if request.stream:
        created_time = int(time.time())
        return StreamingResponse(content=completion_generator(
            request.model, result_queue, choices, created_time, ids),
                                 media_type="text/event-stream",
                                 headers={"Access-Control-Allow-Origin": "*"})
    while True:
        request_id, token, stats = await result_queue.get()
        choice_idx = choices[request_id]
        res.choices[choice_idx].text += str(token)
        if stats is not None:
            res.usage.completion_tokens += stats["tokens"]  # type: ignore
            res.usage.prompt_tokens += stats["prompt"]  # type: ignore
            res.choices[choice_idx].finish_reason = stats["finish_reason"]
            res.choices[choice_idx].stop_reason = stats["stop_reason"]
            completed += 1
            if completed == len(ids):
                runner.remove_result_queues(ids)
                break
            continue
    res.usage.total_tokens = (  # type: ignore
        res.usage.completion_tokens + res.usage.prompt_tokens)  # type: ignore
    return res


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    runner.set_engine_args(engine_args)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    uvicorn.run(app, port=args.port, host=args.host)
