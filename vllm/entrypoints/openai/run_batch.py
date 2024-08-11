import asyncio
from io import StringIO
from typing import Awaitable, Callable, List

import aiohttp

from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
# yapf: disable
from vllm.entrypoints.openai.protocol import (BatchRequestInput,
                                              BatchRequestOutput,
                                              BatchResponseData,
                                              ChatCompletionResponse,
                                              EmbeddingResponse, ErrorResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible batch runner.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help=
        "The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.")
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help="The path or url to a single output file. Currently supports "
        "local file paths, or web (http or https) urls. If a URL is specified,"
        " the file should be available via HTTP PUT.")
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=True`.")

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')

    return parser.parse_args()


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            return f.read()


async def write_file(path_or_url: str, data: str) -> None:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.put(path_or_url, data=data.encode("utf-8")):
            pass
    else:
        # We should make this async, but as long as this is always run as a
        # standalone program, blocking the event loop won't effect performance
        # in this particular case.
        with open(path_or_url, "w", encoding="utf-8") as f:
            f.write(data)


async def run_request(serving_engine_func: Callable,
                      request: BatchRequestInput) -> BatchRequestOutput:
    response = await serving_engine_func(request.body)

    if isinstance(response, (ChatCompletionResponse, EmbeddingResponse)):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                body=response, request_id=f"vllm-batch-{random_uuid()}"),
            error=None,
        )
    elif isinstance(response, ErrorResponse):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                status_code=response.code,
                request_id=f"vllm-batch-{random_uuid()}"),
            error=response,
        )
    else:
        raise ValueError("Request must not be sent in stream mode")

    return batch_output


async def main(args):
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_BATCH_RUNNER)

    # When using single vLLM without engine_use_ray
    model_config = await engine.get_model_config()

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    # Create the openai serving objects.
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
        lora_modules=None,
        prompt_adapters=None,
        request_logger=request_logger,
        chat_template=None,
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        served_model_names,
        request_logger=request_logger,
    )

    # Submit all requests in the file to the engine "concurrently".
    response_futures: List[Awaitable[BatchRequestOutput]] = []
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        # Skip empty lines.
        request_json = request_json.strip()
        if not request_json:
            continue

        request = BatchRequestInput.model_validate_json(request_json)

        # Determine the type of request and run it.
        if request.url == "/v1/chat/completions":
            response_futures.append(
                run_request(openai_serving_chat.create_chat_completion,
                            request))
        elif request.url == "/v1/embeddings":
            response_futures.append(
                run_request(openai_serving_embedding.create_embedding,
                            request))
        else:
            raise ValueError("Only /v1/chat/completions and /v1/embeddings are"
                             "supported in the batch endpoint.")

    responses = await asyncio.gather(*response_futures)

    output_buffer = StringIO()
    for response in responses:
        print(response.model_dump_json(), file=output_buffer)

    output_buffer.seek(0)
    await write_file(args.output_file, output_buffer.read().strip())


if __name__ == "__main__":
    args = parse_args()

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    asyncio.run(main(args))
