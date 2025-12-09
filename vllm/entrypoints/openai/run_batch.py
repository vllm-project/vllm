# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import tempfile
from argparse import Namespace
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from http import HTTPStatus
from io import StringIO
from typing import Any, TypeAlias

import aiofiles
import aiohttp
import torch
from prometheus_client import start_http_server
from pydantic import TypeAdapter, field_validator
from pydantic_core.core_schema import ValidationInfo
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    OpenAIBaseModel,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest, EmbeddingResponse
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.pooling.score.protocol import (
    RerankRequest,
    RerankResponse,
    ScoreRequest,
    ScoreResponse,
)
from vllm.entrypoints.pooling.score.serving import ServingScores
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


BatchRequestInputBody: TypeAlias = (
    ChatCompletionRequest | EmbeddingRequest | ScoreRequest | RerankRequest
)


class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.

    NOTE: Currently only the `/v1/chat/completions` endpoint is supported.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request. Currently
    # /v1/chat/completions is supported.
    url: str

    # The parameters of the request.
    body: BatchRequestInputBody

    @field_validator("body", mode="plain")
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        # Use url to disambiguate models
        url: str = info.data["url"]
        if url == "/v1/chat/completions":
            return ChatCompletionRequest.model_validate(value)
        if url == "/v1/embeddings":
            return TypeAdapter(EmbeddingRequest).validate_python(value)
        if url.endswith("/score"):
            return ScoreRequest.model_validate(value)
        if url.endswith("/rerank"):
            return RerankRequest.model_validate(value)
        return TypeAdapter(BatchRequestInputBody).validate_python(value)


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: (
        ChatCompletionResponse
        | EmbeddingResponse
        | ScoreResponse
        | RerankResponse
        | None
    ) = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: BatchResponseData | None

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Any | None


def make_arg_parser(parser: FlexibleArgumentParser):
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help="The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help="The path or url to a single output file. Currently supports "
        "local file paths, or web (http or https) urls. If a URL is specified,"
        " the file should be available via HTTP PUT.",
    )
    parser.add_argument(
        "--output-tmp-dir",
        type=str,
        default=None,
        help="The directory to store the output file before uploading it "
        "to the output URL.",
    )
    parser.add_argument(
        "--response-role",
        type=optional_type(str),
        default="assistant",
        help="The role name to return if `request.add_generation_prompt=True`.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--max-log-len",
        type=int,
        default=None,
        help="Max number of prompt characters or prompt "
        "ID numbers being printed in log."
        "\n\nDefault: Unlimited",
    )

    parser.add_argument(
        "--enable-metrics", action="store_true", help="Enable Prometheus metrics"
    )
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
        action="store_true",
        default=False,
        help="If set to True, enable prompt_tokens_details in usage.",
    )
    parser.add_argument(
        "--enable-force-include-usage",
        action="store_true",
        default=False,
        help="If set to True, include usage on every request "
        "(even when stream_options is not specified)",
    )

    return parser


def parse_args():
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible batch runner.")
    return make_arg_parser(parser).parse_args()


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


class StreamingBatchProgressTracker:
    """Progress tracker that doesn't require knowing total upfront."""

    def __init__(self):
        self._submitted = 0
        self._completed = 0
        self._pbar: tqdm | None = None
        self._finalized = False

    def submitted(self):
        self._submitted += 1
        if self._pbar is not None and self._finalized:
            self._pbar.total = self._submitted
            self._pbar.refresh()

    def completed(self):
        self._completed += 1
        if self._pbar:
            self._pbar.update()

    def finalize_total(self):
        """Called when all requests have been submitted"""
        self._finalized = True
        if self._pbar:
            self._pbar.total = self._submitted
            self._pbar.refresh()

    def pbar(self) -> tqdm:
        enable_tqdm = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self._pbar = tqdm(
            total=None,
            unit="req",
            desc="Running batch",
            mininterval=5,
            disable=not enable_tqdm,
            bar_format=_BAR_FORMAT,
        )
        return self._pbar


async def stream_file_lines(path_or_url: str) -> AsyncIterator[str]:
    """Stream lines from a local file or URL without loading everything into memory."""
    if path_or_url.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as session:
            async with session.get(path_or_url) as resp:
                buffer = ""
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            yield line

                # Don't forget the last line if no trailing newline
                if buffer.strip():
                    yield buffer.strip()
    else:
        async with aiofiles.open(path_or_url, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    yield line


class OrderedStreamingWriter(AbstractAsyncContextManager):
    """
    Writes batch outputs to a file in order, streaming results as they complete.

    Buffers out-of-order results and flushes them as soon as preceding results arrive.
    This bounds memory usage by the maximum "out-of-orderness" rather than total batch size.
    """

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.next_to_write = 0
        self.buffer: dict[int, BatchRequestOutput] = {}
        self.file_handle = None
        self._lock = asyncio.Lock()
        self._is_url = False
        self._temp_path: str | None = None

    async def __aenter__(self):
        # If the output file is a HTTP upload we write to a named temporary file
        # a future implementation could use multipart uploads for S3-like storages.
        if self.output_path.startswith(("http://", "https://")):
            self._temp_file = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
            )
            self._temp_path = self._temp_file.name
            self._temp_file.close()
            self.file_handle = await aiofiles.open(
                self._temp_path, mode="w", encoding="utf-8"
            )
            self._is_url = True
        else:
            self.file_handle = await aiofiles.open(
                self.output_path, mode="w", encoding="utf-8"
            )
            self._is_url = False
            self._temp_path = None
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        if self.file_handle:
            await self.file_handle.close()

        # If output is a URL, upload the temp file
        if self._is_url and self._temp_path is not None:
            logger.info("Uploading outputs to %s", self.output_path)
            await upload_data(self.output_path, self._temp_path, from_file=True)
            import os

            os.unlink(self._temp_path)

    async def add_result(self, index: int, result: BatchRequestOutput) -> None:
        """Add a result and flush any ready results to disk."""
        async with self._lock:
            self.buffer[index] = result
            await self._flush()

    async def _flush(self) -> None:
        """Write all consecutive ready results starting from next_to_write."""
        if self.file_handle is None:
            raise Exception(
                "Flushed the streaming writer without entering into it's context"
            )
        while self.next_to_write in self.buffer:
            result = self.buffer.pop(self.next_to_write)
            await self.file_handle.write(result.model_dump_json() + "\n")
            self.next_to_write += 1

        # Periodically flush to disk to avoid buffering too much in the file handle
        if self.next_to_write % 100 == 0:
            await self.file_handle.flush()

    @property
    def buffered_count(self) -> int:
        """Number of results currently buffered (waiting for earlier results)."""
        return len(self.buffer)

    @property
    def written_count(self) -> int:
        return self.next_to_write


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, encoding="utf-8") as f:
            return f.read()


async def write_local_file(
    output_path: str, batch_outputs: list[BatchRequestOutput]
) -> None:
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


async def upload_data(output_url: str, data_or_file: str, from_file: bool) -> None:
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
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=1000)
            ) as session:
                if from_file:
                    with open(data_or_file, "rb") as file:
                        async with session.put(output_url, data=file) as response:
                            if response.status != 200:
                                raise Exception(
                                    f"Failed to upload file.\n"
                                    f"Status: {response.status}\n"
                                    f"Response: {response.text()}"
                                )
                else:
                    async with session.put(output_url, data=data_or_file) as response:
                        if response.status != 200:
                            raise Exception(
                                f"Failed to upload data.\n"
                                f"Status: {response.status}\n"
                                f"Response: {response.text()}"
                            )

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


async def write_file(
    path_or_url: str, batch_outputs: list[BatchRequestOutput], output_tmp_dir: str
) -> None:
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
                logger.info("Writing outputs to temporary local file %s", f.name)
                await write_local_file(f.name, batch_outputs)
                logger.info("Uploading outputs to %s", path_or_url)
                await upload_data(path_or_url, f.name, from_file=True)
    else:
        logger.info("Writing outputs to local file %s", path_or_url)
        await write_local_file(path_or_url, batch_outputs)


def make_error_request_output(
    request: BatchRequestInput, error_msg: str
) -> BatchRequestOutput:
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
    request: BatchRequestInput, error_msg: str
) -> BatchRequestOutput:
    return make_error_request_output(request, error_msg)


async def run_request(
    serving_engine_func: Callable,
    request: BatchRequestInput,
    index: int,
    tracker: StreamingBatchProgressTracker,
    writer: OrderedStreamingWriter,
) -> None:
    """Run a request and stream the result to the writer."""
    try:
        response = await serving_engine_func(request.body)

        if isinstance(
            response,
            (ChatCompletionResponse, EmbeddingResponse, ScoreResponse, RerankResponse),
        ):
            batch_output = BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=BatchResponseData(
                    body=response, request_id=f"vllm-batch-{random_uuid()}"
                ),
                error=None,
            )
        elif isinstance(response, ErrorResponse):
            batch_output = BatchRequestOutput(
                id=f"vllm-{random_uuid()}",
                custom_id=request.custom_id,
                response=BatchResponseData(
                    status_code=response.error.code,
                    request_id=f"vllm-batch-{random_uuid()}",
                ),
                error=response,
            )
        else:
            batch_output = make_error_request_output(
                request, error_msg="Request must not be sent in stream mode"
            )
    except Exception as e:
        batch_output = make_error_request_output(request, error_msg=str(e))

    await writer.add_result(index, batch_output)
    tracker.completed()


async def run_request_with_error(
    request: BatchRequestInput,
    index: int,
    error_msg: str,
    writer: OrderedStreamingWriter,
) -> None:
    """Handle error case and stream to writer."""
    batch_output = make_error_request_output(request, error_msg)
    await writer.add_result(index, batch_output)


def validate_run_batch_args(args):
    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )


async def run_batch(
    engine_client: EngineClient,
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
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    model_config = engine_client.model_config
    supported_tasks = await engine_client.get_supported_tasks()
    logger.info("Supported tasks: %s", supported_tasks)

    # Create the openai serving objects.
    openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=None,
    )

    openai_serving_chat = (
        OpenAIServingChat(
            engine_client,
            openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template=None,
            chat_template_content_format="auto",
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )

    openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            openai_serving_models,
            request_logger=request_logger,
            chat_template=None,
            chat_template_content_format="auto",
        )
        if "embed" in supported_tasks
        else None
    )

    enable_serving_reranking = (
        "classify" in supported_tasks
        and getattr(model_config.hf_config, "num_labels", 0) == 1
    )

    openai_serving_scores = (
        ServingScores(
            engine_client,
            openai_serving_models,
            request_logger=request_logger,
        )
        if ("embed" in supported_tasks or enable_serving_reranking)
        else None
    )

    tracker = StreamingBatchProgressTracker()
    logger.info("Reading batch from %s...", args.input_file)

    async with OrderedStreamingWriter(args.output_file) as writer:
        tasks: set[asyncio.Task] = set()

        with tracker.pbar():
            index = 0
            async for request_json in stream_file_lines(args.input_file):
                request = BatchRequestInput.model_validate_json(request_json)

                # Create task for this request
                task = await create_request_task(
                    request=request,
                    index=index,
                    tracker=tracker,
                    writer=writer,
                    openai_serving_chat=openai_serving_chat,
                    openai_serving_embedding=openai_serving_embedding,
                    openai_serving_scores=openai_serving_scores,
                )

                if task is not None:
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                index += 1

            # All requests submitted
            tracker.finalize_total()
            logger.info("All %d requests submitted, waiting for completion...", index)

            if tasks:
                await asyncio.gather(*tasks)

        logger.info(
            "Batch complete. Written: %d, Final buffer: %d",
            writer.written_count,
            writer.buffered_count,
        )


async def create_request_task(
    request: BatchRequestInput,
    index: int,
    tracker: StreamingBatchProgressTracker,
    writer: OrderedStreamingWriter,
    openai_serving_chat: OpenAIServingChat | None,
    openai_serving_embedding: OpenAIServingEmbedding | None,
    openai_serving_scores: ServingScores | None,
) -> asyncio.Task | None:
    """Create and return a task for processing a request."""

    if request.url == "/v1/chat/completions":
        if openai_serving_chat is None:
            await run_request_with_error(
                request,
                index,
                "The model does not support Chat Completions API",
                writer,
            )
            return None
        tracker.submitted()
        return asyncio.create_task(
            run_request(
                openai_serving_chat.create_chat_completion,
                request,
                index,
                tracker,
                writer,
            )
        )

    elif request.url == "/v1/embeddings":
        if openai_serving_embedding is None:
            await run_request_with_error(
                request,
                index,
                "The model does not support Embeddings API",
                writer,
            )
            return None
        tracker.submitted()
        return asyncio.create_task(
            run_request(
                openai_serving_embedding.create_embedding,
                request,
                index,
                tracker,
                writer,
            )
        )

    elif request.url.endswith("/score"):
        if openai_serving_scores is None:
            await run_request_with_error(
                request,
                index,
                "The model does not support Scores API",
                writer,
            )
            return None
        tracker.submitted()
        return asyncio.create_task(
            run_request(
                openai_serving_scores.create_score,
                request,
                index,
                tracker,
                writer,
            )
        )

    elif request.url.endswith("/rerank"):
        if openai_serving_scores is None:
            await run_request_with_error(
                request,
                index,
                "The model does not support Rerank API",
                writer,
            )
            return None
        tracker.submitted()
        return asyncio.create_task(
            run_request(
                openai_serving_scores.do_rerank,
                request,
                index,
                tracker,
                writer,
            )
        )

    else:
        await run_request_with_error(
            request,
            index,
            f"URL {request.url} was used. "
            "Supported endpoints: /v1/chat/completions, /v1/embeddings, /score, /rerank.",
            writer,
        )
        return None


async def main(args: Namespace):
    from vllm.entrypoints.openai.api_server import build_async_engine_client
    from vllm.usage.usage_lib import UsageContext

    validate_run_batch_args(args)

    async with build_async_engine_client(
        args,
        usage_context=UsageContext.OPENAI_BATCH_RUNNER,
        disable_frontend_multiprocessing=False,
    ) as engine_client:
        await run_batch(engine_client, args)


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
