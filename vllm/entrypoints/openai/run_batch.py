# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import tempfile
from argparse import Namespace
from collections.abc import Awaitable
from http import HTTPStatus
from io import StringIO
from typing import Callable, Optional

import aiohttp
import torch
from prometheus_client import start_http_server
from tqdm import tqdm

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
# yapf: disable
from vllm.entrypoints.openai.protocol import (BatchRequestInput,
                                              BatchRequestOutput,
                                              BatchResponseData,
                                              ChatCompletionResponse,
                                              EmbeddingResponse, ErrorResponse,
                                              RerankResponse, ScoreResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_score import ServingScores
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def make_arg_parser(parser: FlexibleArgumentParser):
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
    parser.add_argument(
        "--output-tmp-dir",
        type=str,
        default=None,
        help="The directory to store the output file before uploading it "
        "to the output URL.",
    )
    parser.add_argument("--response-role",
                        type=optional_type(str),
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

    parser.add_argument("--enable-metrics",
                        action="store_true",
                        help="Enable Prometheus metrics")
    parser.add_argument(
        "--url",
        type=str,
        default="0.0.0.0",
        help="URL to the Prometheus metrics server "
        "(only needed if enable-metrics is set).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the Prometheus metrics server "
        "(only needed if enable-metrics is set).",
    )
    parser.add_argument(
        "--enable-prompt-tokens-details",
        action='store_true',
        default=False,
        help="If set to True, enable prompt_tokens_details in usage.")

    return parser


def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible batch runner.")
    return make_arg_parser(parser).parse_args()


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


class BatchProgressTracker:

    def __init__(self):
        self._total = 0
        self._pbar: Optional[tqdm] = None

    def submitted(self):
        self._total += 1

    def completed(self):
        if self._pbar:
            self._pbar.update()

    def pbar(self) -> tqdm:
        enable_tqdm = not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0
        self._pbar = tqdm(total=self._total,
                          unit="req",
                          desc="Running batch",
                          mininterval=5,
                          disable=not enable_tqdm,
                          bar_format=_BAR_FORMAT)
        return self._pbar


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, encoding="utf-8") as f:
            return f.read()


async def write_local_file(output_path: str,
                           batch_outputs: list[BatchRequestOutput]) -> None:
    """
    Write the responses to a local file.
    output_path: The path to write the responses to.
    batch_outputs: The list of batch outputs to write.
    """
    # We should make this async, but as long as run_batch runs as a
    # standalone program, blocking the event loop won't affect performance.
    with open(output_path, "w", encoding="utf-8") as f:
        for o in batch_outputs:
            print(o.model_dump_json(), file=f)


async def upload_data(output_url: str, data_or_file: str,
                      from_file: bool) -> None:
    """
    Upload a local file to a URL.
    output_url: The URL to upload the file to.
    data_or_file: Either the data to upload or the path to the file to upload.
    from_file: If True, data_or_file is the path to the file to upload.
    """
    # Timeout is a common issue when uploading large files.
    # We retry max_retries times before giving up.
    max_retries = 5
    # Number of seconds to wait before retrying.
    delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            # We increase the timeout to 1000 seconds to allow
            # for large files (default is 300).
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
                    total=1000)) as session:
                if from_file:
                    with open(data_or_file, "rb") as file:
                        async with session.put(output_url,
                                               data=file) as response:
                            if response.status != 200:
                                raise Exception(f"Failed to upload file.\n"
                                                f"Status: {response.status}\n"
                                                f"Response: {response.text()}")
                else:
                    async with session.put(output_url,
                                           data=data_or_file) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to upload data.\n"
                                            f"Status: {response.status}\n"
                                            f"Response: {response.text()}")

        except Exception as e:
            if attempt < max_retries:
                logger.error(
                    "Failed to upload data (attempt %d). Error message: %s.\nRetrying in %d seconds...",  # noqa: E501
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise Exception(
                    f"Failed to upload data (attempt {attempt}). Error message: {str(e)}."  # noqa: E501
                ) from e


async def write_file(path_or_url: str, batch_outputs: list[BatchRequestOutput],
                     output_tmp_dir: str) -> None:
    """
    Write batch_outputs to a file or upload to a URL.
    path_or_url: The path or URL to write batch_outputs to.
    batch_outputs: The list of batch outputs to write.
    output_tmp_dir: The directory to store the output file before uploading it
    to the output URL.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        if output_tmp_dir is None:
            logger.info("Writing outputs to memory buffer")
            output_buffer = StringIO()
            for o in batch_outputs:
                print(o.model_dump_json(), file=output_buffer)
            output_buffer.seek(0)
            logger.info("Uploading outputs to %s", path_or_url)
            await upload_data(
                path_or_url,
                output_buffer.read().strip().encode("utf-8"),
                from_file=False,
            )
        else:
            # Write responses to a temporary file and then upload it to the URL.
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=output_tmp_dir,
                    prefix="tmp_batch_output_",
                    suffix=".jsonl",
            ) as f:
                logger.info("Writing outputs to temporary local file %s",
                            f.name)
                await write_local_file(f.name, batch_outputs)
                logger.info("Uploading outputs to %s", path_or_url)
                await upload_data(path_or_url, f.name, from_file=True)
    else:
        logger.info("Writing outputs to local file %s", path_or_url)
        await write_local_file(path_or_url, batch_outputs)


def make_error_request_output(request: BatchRequestInput,
                              error_msg: str) -> BatchRequestOutput:
    batch_output = BatchRequestOutput(
        id=f"vllm-{random_uuid()}",
        custom_id=request.custom_id,
        response=BatchResponseData(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id=f"vllm-batch-{random_uuid()}",
        ),
        error=error_msg,
    )
    return batch_output


async def make_async_error_request_output(
        request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    return make_error_request_output(request, error_msg)


async def run_request(serving_engine_func: Callable,
                      request: BatchRequestInput,
                      tracker: BatchProgressTracker) -> BatchRequestOutput:
    response = await serving_engine_func(request.body)

    if isinstance(
            response,
        (ChatCompletionResponse, EmbeddingResponse, ScoreResponse,
         RerankResponse),
    ):
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
                status_code=response.error.code,
                request_id=f"vllm-batch-{random_uuid()}"),
            error=response,
        )
    else:
        batch_output = make_error_request_output(
            request, error_msg="Request must not be sent in stream mode")

    tracker.completed()
    return batch_output


async def run_batch(
    engine_client: EngineClient,
    vllm_config: VllmConfig,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    model_config = vllm_config.model_config

    if envs.VLLM_USE_V1:
        supported_tasks = await engine_client \
            .get_supported_tasks()  # type: ignore
    else:
        supported_tasks = model_config.supported_tasks

    logger.info("Supported_tasks: %s", supported_tasks)

    # Create the openai serving objects.
    openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=None,
    )
    openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=None,
        chat_template_content_format="auto",
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if "generate" in supported_tasks else None
    openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
        chat_template=None,
        chat_template_content_format="auto",
    ) if "embed" in supported_tasks else None

    enable_serving_reranking = ("classify" in supported_tasks and getattr(
        model_config.hf_config, "num_labels", 0) == 1)

    openai_serving_scores = ServingScores(
        engine_client,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
    ) if ("embed" in supported_tasks or enable_serving_reranking) else None

    tracker = BatchProgressTracker()
    logger.info("Reading batch from %s...", args.input_file)

    # Submit all requests in the file to the engine "concurrently".
    response_futures: list[Awaitable[BatchRequestOutput]] = []
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        # Skip empty lines.
        request_json = request_json.strip()
        if not request_json:
            continue

        request = BatchRequestInput.model_validate_json(request_json)

        # Determine the type of request and run it.
        if request.url == "/v1/chat/completions":
            chat_handler_fn = openai_serving_chat.create_chat_completion if \
                openai_serving_chat is not None else None
            if chat_handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg=
                        "The model does not support Chat Completions API",
                    ))
                continue

            response_futures.append(
                run_request(chat_handler_fn, request, tracker))
            tracker.submitted()
        elif request.url == "/v1/embeddings":
            embed_handler_fn = openai_serving_embedding.create_embedding if \
                openai_serving_embedding is not None else None
            if embed_handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Embeddings API",
                    ))
                continue

            response_futures.append(
                run_request(embed_handler_fn, request, tracker))
            tracker.submitted()
        elif request.url.endswith("/score"):
            score_handler_fn = openai_serving_scores.create_score if \
                openai_serving_scores is not None else None
            if score_handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Scores API",
                    ))
                continue

            response_futures.append(
                run_request(score_handler_fn, request, tracker))
            tracker.submitted()
        elif request.url.endswith("/rerank"):
            rerank_handler_fn = openai_serving_scores.do_rerank if \
                openai_serving_scores is not None else None
            if rerank_handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Rerank API",
                    ))
                continue

            response_futures.append(
                run_request(rerank_handler_fn, request, tracker))
            tracker.submitted()
        else:
            response_futures.append(
                make_async_error_request_output(
                    request,
                    error_msg=f"URL {request.url} was used. "
                    "Supported endpoints: /v1/chat/completions, /v1/embeddings,"
                    " /score, /rerank ."
                    "See vllm/entrypoints/openai/api_server.py for supported "
                    "score/rerank versions.",
                ))

    with tracker.pbar():
        responses = await asyncio.gather(*response_futures)

    await write_file(args.output_file, responses, args.output_tmp_dir)


async def main(args: Namespace):
    from vllm.entrypoints.openai.api_server import build_async_engine_client
    from vllm.usage.usage_lib import UsageContext

    async with build_async_engine_client(
            args,
            usage_context=UsageContext.OPENAI_BATCH_RUNNER,
            disable_frontend_multiprocessing=False,
    ) as engine_client:
        vllm_config = await engine_client.get_vllm_config()

        await run_batch(engine_client, vllm_config, args)


if __name__ == "__main__":
    args = parse_args()

    logger.info("vLLM batch processing API version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    # Start the Prometheus metrics server. LLMEngine uses the Prometheus client
    # to publish metrics at the /metrics endpoint.
    if args.enable_metrics:
        logger.info("Prometheus metrics enabled")
        start_http_server(port=args.port, addr=args.url)
    else:
        logger.info("Prometheus metrics disabled")

    asyncio.run(main(args))
