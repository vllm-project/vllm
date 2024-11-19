import asyncio
import multiprocessing
import re
import threading
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import Mount
from prometheus_client import make_asgi_app

import vllm
import vllm.envs as envs
from vllm import FastSyncLLM as LLM
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (MultiModalItemTracker,
                                         _parse_chat_message_content,
                                         load_chat_template,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, CompletionRequest,
    CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage,
    ErrorResponse, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.entrypoints.openai.serving_chat import ConversationMessage
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import FlexibleArgumentParser, random_uuid

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
        self.engine_args: EngineArgs
        self.engine_config: VllmConfig
        self.input_queue: multiprocessing.Queue = mp.Queue()
        self.result_queue: multiprocessing.Queue = mp.Queue()
        self.result_queues: Dict[str, asyncio.Queue] = {}
        self.t: threading.Thread = threading.Thread(target=self.thread_proc)
        self.loop = None
        self.llm: LLM
        self.proc: multiprocessing.Process
        self.tokenizer = None
        self.response_role: str
        self.chat_template: Optional[str]
        self.chat_template_content_format = "auto"

    def set_response_role(self, role):
        self.response_role = role

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
        self.proc = mp.Process(  # type: ignore[attr-defined]
            target=self.llm.run_engine)
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
    runner.engine_config = runner.engine_args.create_engine_config()

    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        tokenizer_revision=engine_args.tokenizer_revision,
        trust_remote_code=engine_args.trust_remote_code,
        truncation_side="left")
    runner.tokenizer = tokenizer
    yield


app = FastAPI(lifespan=lifespan)

# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


@app.get("/v1/models")
async def show_available_models():
    models = [
        ModelCard(id=runner.engine_args.model,
                  root=runner.engine_args.model,
                  permission=[ModelPermission()])
    ]
    model_list = ModelList(data=models)
    return JSONResponse(content=model_list.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


async def _check_model(request: Union[CompletionRequest,
                                      ChatCompletionRequest]):
    model = request.model
    if model != runner.engine_args.model:
        return ErrorResponse(message=f"The model {model} does not exist.",
                             type="NotFoundError",
                             code=HTTPStatus.NOT_FOUND)
    return None


async def completion_generator(model, result_queue, choices, created_time,
                               ids):
    completed = 0
    try:
        while True:
            request_id, token, stats = await result_queue.get()

            choice_idx = choices[request_id]
            res = CompletionStreamResponse(id=request_id,
                                           created=created_time,
                                           model=model,
                                           choices=[
                                               CompletionResponseStreamChoice(
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
                res.usage.total_tokens = (
                    res.usage.completion_tokens +  # type: ignore
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
    error_check_ret = await _check_model(request)
    if error_check_ret is not None:
        return JSONResponse(content=error_check_ret.model_dump(),
                            status_code=error_check_ret.code)
    sampling_params = request.to_sampling_params(
        default_max_tokens=runner.engine_config.model_config.max_model_len
        # TODO: gshtras add - len(prompt_inputs["prompt_token_ids"])
    )
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
                                 media_type="text/event-stream")
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


async def chat_completion_generator(model, result_queue, created_time, id):
    try:
        first_token = ChatCompletionStreamResponse(
            id=id,
            created=created_time,
            model=model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role=runner.response_role),
                    logprobs=None,
                    finish_reason=None,
                    stop_reason=None)
            ],
            usage=None)
        response_json = first_token.model_dump_json(exclude_unset=True)
        yield f"data: {response_json}\n\n"

        while True:
            request_id, token, stats = await result_queue.get()
            assert request_id == id

            res = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=token),
                        logprobs=None,
                        finish_reason=None,
                        stop_reason=None)
                ],
                usage=None)
            if stats is not None:
                res.usage = UsageInfo()
                res.usage.completion_tokens = stats.get("tokens", 0)
                res.usage.prompt_tokens = stats.get("prompt", 0)
                res.usage.total_tokens = (
                    res.usage.completion_tokens +  # type: ignore
                    res.usage.prompt_tokens)
                res.choices[0].finish_reason = stats["finish_reason"]
                res.choices[0].stop_reason = stats["stop_reason"]
            response_json = res.model_dump_json(exclude_unset=True)
            yield f"data: {response_json}\n\n"
            if stats is not None:
                runner.remove_result_queues([id])
                break

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("Error in completion_generator: %s", e)
    return


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest,
                           raw_request: Request):
    error_check_ret = await _check_model(request)
    if error_check_ret is not None:
        return JSONResponse(content=error_check_ret.model_dump(),
                            status_code=error_check_ret.code)
    sampling_params = request.to_sampling_params(
        default_max_tokens=runner.engine_config.model_config.max_model_len
        # TODO: gshtras add len(prompt_inputs["prompt_token_ids"])
    )
    conversation: List[ConversationMessage] = []

    res = ChatCompletionResponse(model=request.model,
                                 choices=[],
                                 usage=UsageInfo(prompt_tokens=0,
                                                 total_tokens=0,
                                                 completion_tokens=0))

    mm_tracker = MultiModalItemTracker(runner.engine_config.model_config,
                                       runner.tokenizer)
    chat_template = request.chat_template or runner.chat_template
    content_format = resolve_chat_template_content_format(
        chat_template, runner.chat_template_content_format, runner.tokenizer)
    for msg in request.messages:
        parsed_msg = _parse_chat_message_content(msg, mm_tracker,
                                                 content_format)
        conversation.extend(parsed_msg)

    prompt = runner.tokenizer.apply_chat_template(  # type: ignore
        conversation=conversation,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt,
    )

    ids, result_queue = await runner.add_request(prompt, sampling_params)
    assert len(ids) == 1

    if request.stream:
        created_time = int(time.time())
        return StreamingResponse(content=chat_completion_generator(
            request.model, result_queue, created_time, ids[0]),
                                 media_type="text/event-stream")

    res.choices.append(
        ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role=runner.response_role, content=""),
            finish_reason=None,
            stop_reason=None))

    while True:
        _, token, stats = await result_queue.get()
        assert res.choices[0].message.content is not None
        res.choices[0].message.content += str(token)
        if stats is not None:
            res.usage.completion_tokens += stats["tokens"]  # type: ignore
            res.usage.prompt_tokens += stats["prompt"]  # type: ignore
            res.choices[0].finish_reason = stats["finish_reason"]
            res.choices[0].stop_reason = stats["stop_reason"]
            runner.remove_result_queues(ids)
            break
    res.usage.total_tokens = (  # type: ignore
        res.usage.completion_tokens + res.usage.prompt_tokens)  # type: ignore
    return res


def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    runner.set_engine_args(engine_args)
    runner.set_response_role(args.response_role)
    runner.chat_template = load_chat_template(args.chat_template)
    runner.chat_template_content_format = args.chat_template_content_format

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    uvicorn.run(app, port=args.port, host=args.host)
