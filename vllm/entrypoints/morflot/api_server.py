import asyncio
from contextlib import asynccontextmanager
import random
import re
import string
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import Mount
from prometheus_client import make_asgi_app
import torch
import time
from typing import Annotated, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from vllm import MorflotLLM as LLM
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


class BackgroundRunner:

    def __init__(self):
        self.value = 0
        self.engine_args = None
    
    def set_engine_args(self, engine_args):
        self.engine_args = engine_args

    async def run_main(self):
        self.input_queue = asyncio.Queue()
        self.llm = LLM(engine_args=self.engine_args,
            input_queue=self.input_queue,
        )

        await self.llm.run_engine()

    async def add_request(self, prompt, sampling_params):
        result_queue = asyncio.Queue()
        ids = []
        if isinstance(prompt, str) or (isinstance(prompt, list)
                                       and isinstance(prompt[0], int)):
            id = random_uuid()
            self.input_queue.put_nowait(
                (id, prompt, sampling_params, result_queue))
            ids.append(id)
        else:
            for p in prompt:
                id = random_uuid()
                self.input_queue.put_nowait(
                    (id, p, sampling_params, result_queue))
                ids.append(id)

        return ids, result_queue


runner = BackgroundRunner()


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class CompletionLogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None,
                                ge=torch.iinfo(torch.long).min,
                                le=torch.iinfo(torch.long).max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: Optional[bool] = False
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    include_stop_str_in_output: Optional[bool] = Field(
        default=False,
        description=(
            "Whether to include the stop string in the output. "
            "This is only applied when the stop or stop_token_ids is set."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-completion-extra-params

    def to_sampling_params(self):
        echo_without_generation = self.echo and self.max_tokens == 0

        logits_processors = None
        if self.logit_bias:

            def logit_bias_logits_processor(
                    token_ids: List[int],
                    logits: torch.Tensor) -> torch.Tensor:
                assert self.logit_bias is not None
                for token_id, bias in self.logit_bias.items():
                    # Clamp the bias between -100 and 100 per OpenAI API spec
                    bias = min(100, max(-100, bias))
                    logits[int(token_id)] += bias
                return logits

            logits_processors = [logit_bias_logits_processor]

        return SamplingParams(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            logprobs=self.logprobs,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            prompt_logprobs=self.logprobs if self.echo else None,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=(self.spaces_between_special_tokens),
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            logits_processors=logits_processors,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(runner.run_main())
    yield

app = FastAPI(lifespan=lifespan)

# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


async def completion_generator(model, result_queue, choices, created_time):
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
                res.usage.completion_tokens = stats["tokens"]
                res.usage.prompt_tokens = stats["prompt"]
                res.usage.total_tokens = res.usage.completion_tokens + res.usage.prompt_tokens
                res.choices[0].finish_reason = stats["finish_reason"]
                res.choices[0].stop_reason = stats["stop_reason"]
                completed += 1
            response_json = res.model_dump_json(exclude_unset=True)
            yield f"data: {response_json}\n\n"
            if completed == len(choices):
                break

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in completion_generator: {e}")
    return


"""
    async for request_id, token, stats in result_queue:
        choice_idx = choices[request_id]
        res = CompletionResponse(model=model, choices=CompletionResponseChoice(index=choice_idx, text=token), usage=None)
        if stats is not None:
            res.usage.completion_tokens = stats["tokens"]
            res.usage.prompt_tokens = stats["prompt"]
            res.usage.total_tokens = res.usage.completion_tokens + res.usage.prompt_tokens
            res.choices[choice_idx].finish_reason = stats["finish_reason"]
            res.choices[choice_idx].stop_reason = stats["stop_reason"]
            completed += 1
            if completed == len(choices):
                break
        response_json = res.model_dump_json(exclude_unset=True)
        yield f"data: {response_json}\n\n"
    yield "data: [DONE]\n\n"
"""


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
            request.model, result_queue, choices, created_time),
                                 media_type="text/event-stream")
    while True:
        request_id, token, stats = await result_queue.get()
        choice_idx = choices[request_id]
        res.choices[choice_idx].text += str(token)
        if stats is not None:
            res.usage.completion_tokens += stats["tokens"]
            res.usage.prompt_tokens += stats["prompt"]
            res.choices[choice_idx].finish_reason = stats["finish_reason"]
            res.choices[choice_idx].stop_reason = stats["stop_reason"]
            completed += 1
            if completed == len(ids):
                break
            continue
    res.usage.total_tokens = res.usage.completion_tokens + res.usage.prompt_tokens
    return res

@app.get("/flush")
async def flush():
    from rpdTracerControl import rpdTracerControl
    rpd = rpdTracerControl()
    rpd.stop()
    rpd.flush()
    ver = {"res": "OK"}
    return JSONResponse(content=ver)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    engine_args = EngineArgs.from_cli_args(args)
    runner.set_engine_args(engine_args)
    #from rpdTracerControl import rpdTracerControl
    #rpd = rpdTracerControl()
    #rpd.setPythonTrace(True)
    #rpd.start()
    uvicorn.run(app, port=args.port, host=args.host)
