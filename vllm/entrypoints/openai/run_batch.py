# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import json
import os
import shutil
import stat
import sys
import tempfile
from argparse import Namespace
from collections.abc import Awaitable, Callable, Iterable
from http import HTTPStatus
from io import BytesIO
from typing import IO, Any, TypeAlias, TypedDict
from urllib.parse import urlparse

import aiohttp
import pybase64 as base64
import pydantic
import torch
from fastapi import UploadFile
from prometheus_client import start_http_server
from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo
from starlette.datastructures import State
from starlette.responses import JSONResponse
from tqdm import tqdm
from urllib3.util import parse_url

import vllm.envs as envs
from vllm.config import config
from vllm.connections import global_http_connection
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.cli_args import BaseFrontendArgs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    OpenAIBaseModel,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from vllm.entrypoints.pooling.scoring.protocol import (
    RerankRequest,
    RerankResponse,
    ScoreRequest,
    ScoreResponse,
)
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class BatchTranscriptionRequest(TranscriptionRequest):
    """
    Batch transcription request that uses file_url instead of file.

    This class extends TranscriptionRequest but replaces the file field
    with file_url to support batch processing from audio files written in JSON format.
    """

    file_url: str = Field(
        ...,
        description=(
            "Either a URL of the audio or a data URL with base64 encoded audio data. "
        ),
    )

    # Override file to be optional and unused for batch processing
    file: UploadFile | None = Field(default=None, exclude=True)  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def validate_no_file(cls, data: Any):
        """Ensure file field is not provided in batch requests."""
        if isinstance(data, dict) and "file" in data:
            raise VLLMValidationError(
                "The 'file' field is not supported in batch requests. "
                "Use 'file_url' instead.",
                parameter="file",
            )
        return data


class BatchTranslationRequest(TranslationRequest):
    """
    Batch translation request that uses file_url instead of file.

    This class extends TranslationRequest but replaces the file field
    with file_url to support batch processing from audio files written in JSON format.
    """

    file_url: str = Field(
        ...,
        description=(
            "Either a URL of the audio or a data URL with base64 encoded audio data. "
        ),
    )

    # Override file to be optional and unused for batch processing
    file: UploadFile | None = Field(default=None, exclude=True)  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def validate_no_file(cls, data: Any):
        """Ensure file field is not provided in batch requests."""
        if isinstance(data, dict) and "file" in data:
            raise VLLMValidationError(
                "The 'file' field is not supported in batch requests. "
                "Use 'file_url' instead.",
                parameter="file",
            )
        return data


BatchRequestInputBody: TypeAlias = (
    ChatCompletionRequest
    | EmbeddingRequest
    | ScoreRequest
    | RerankRequest
    | BatchTranscriptionRequest
    | BatchTranslationRequest
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
            return TypeAdapter(ScoreRequest).validate_python(value)
        if url.endswith("/rerank"):
            return RerankRequest.model_validate(value)
        if url == "/v1/audio/transcriptions":
            return BatchTranscriptionRequest.model_validate(value)
        if url == "/v1/audio/translations":
            return BatchTranslationRequest.model_validate(value)
        return TypeAdapter(BatchRequestInputBody).validate_python(value)


AllResponse: TypeAlias = (
    ChatCompletionResponse
    | EmbeddingResponse
    | ScoreResponse
    | RerankResponse
    | TranscriptionResponse
    | TranscriptionResponseVerbose
    | TranslationResponse
    | TranslationResponseVerbose
)


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: AllResponse | None = None


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


@config
class BatchFrontendArgs(BaseFrontendArgs):
    """Arguments for the batch runner frontend."""

    input_file: str | None = None
    """The path or url to a single input file. Currently supports local file
    paths, or the http protocol (http or https). If a URL is specified,
    the file should be available via HTTP GET."""
    output_file: str | None = None
    """The path or url to a single output file. Currently supports
    local file paths, or web (http or https) urls. If a URL is specified,
    the file should be available via HTTP PUT."""
    output_tmp_dir: str | None = None
    """The directory used to stage files that cannot be streamed in place: the
    output file before it is uploaded to an output URL, and an input that has
    to be copied before it can be read twice (a URL or a pipe). Defaults to the
    system temporary directory, which may be small on containers."""
    max_inflight: int = 1024
    """Maximum number of requests submitted to the engine at once. Bounds
    frontend memory, which would otherwise grow with the size of the input
    file. The engine schedules its own batches, so this only caps how many
    requests are queued ahead of it."""
    enable_metrics: bool = False
    """Enable Prometheus metrics"""
    host: str | None = None
    """Host name for the Prometheus metrics server
    (only needed if enable-metrics is set)."""
    port: int = 8000
    """Port number for the Prometheus metrics server
    (only needed if enable-metrics is set)."""
    url: str = "0.0.0.0"
    """[DEPRECATED] Host name for the Prometheus metrics server
    (only needed if enable-metrics is set). Use --host instead."""

    @classmethod
    def _customize_cli_kwargs(
        cls,
        frontend_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        frontend_kwargs = super()._customize_cli_kwargs(frontend_kwargs)

        frontend_kwargs["input_file"]["flags"] = ["-i"]
        frontend_kwargs["input_file"]["required"] = True
        frontend_kwargs["output_file"]["flags"] = ["-o"]
        frontend_kwargs["output_file"]["required"] = True

        frontend_kwargs["enable_metrics"]["action"] = "store_true"

        frontend_kwargs["url"]["deprecated"] = True
        return frontend_kwargs


def make_arg_parser(parser: FlexibleArgumentParser):
    parser = BatchFrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser


def parse_args():
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible batch runner.")
    args = make_arg_parser(parser).parse_args()

    # Backward compatibility: If --url is set, use it for host
    url_explicit = any(arg == "--url" or arg.startswith("--url=") for arg in sys.argv)
    host_explicit = any(
        arg == "--host" or arg.startswith("--host=") for arg in sys.argv
    )
    if url_explicit and hasattr(args, "url") and not host_explicit:
        args.host = args.url
        logger.warning_once(
            "Using --url for metrics is deprecated. Please use --host instead."
        )

    return args


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501

_STAGING_CHUNK_SIZE = 1 << 20


class BatchProgressTracker:
    def __init__(self):
        self._pbar: tqdm | None = None

    def completed(self):
        if self._pbar:
            self._pbar.update()

    def pbar(self, total: int) -> tqdm:
        enable_tqdm = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self._pbar = tqdm(
            total=total,
            unit="req",
            desc="Running batch",
            mininterval=5,
            disable=not enable_tqdm,
            bar_format=_BAR_FORMAT,
        )
        return self._pbar


def is_descriptor_alias(path: str) -> bool:
    """Whether a path names an open descriptor rather than a file of its own.

    Reopening one of these shares the descriptor's offset on some platforms, so
    a second pass over it would start at EOF even though it stats as a regular
    file. Ordinary files under /dev, a batch on tmpfs for instance, are not
    affected.
    """
    if path in ("/dev/stdin", "/dev/stdout", "/dev/stderr"):
        return True
    directory, _, number = path.rpartition("/")
    if not number.isdigit():
        return False
    return directory == "/dev/fd" or (
        directory.startswith("/proc/") and directory.endswith("/fd")
    )


def needs_staging(path_or_url: str) -> bool:
    """Whether the input has to be copied before it can be read twice."""
    if path_or_url.startswith(("http://", "https://")):
        return True
    if is_descriptor_alias(path_or_url):
        return True
    return not stat.S_ISREG(os.stat(path_or_url).st_mode)


@contextlib.asynccontextmanager
async def local_input_path(path_or_url: str, tmp_dir: str | None):
    """
    Yield a local path that can be read twice.

    The batch is read once to validate it and once to run it. Anything that
    cannot be re-read, a URL body or a stream such as a pipe or process
    substitution, is staged to a temporary file first.
    """
    if not needs_staging(path_or_url):
        yield path_or_url
        return

    is_url = path_or_url.startswith(("http://", "https://"))

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=tmp_dir,
        prefix="tmp_batch_input_",
        suffix=".jsonl",
    ) as f:
        logger.info("Staging %s to %s", path_or_url, f.name)
        if is_url:
            async with (
                aiohttp.ClientSession() as session,
                session.get(path_or_url) as resp,
            ):
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(_STAGING_CHUNK_SIZE):
                    f.write(chunk)
        else:
            with open(path_or_url, "rb") as source:
                shutil.copyfileobj(source, f, _STAGING_CHUNK_SIZE)
        f.flush()
        yield f.name


def validate_batch(input_path: str) -> int:
    """
    Parse every request and return how many there are.

    Parsing up front keeps a malformed batch from producing partial output and
    gives the progress bar a total. The parsed requests are deliberately
    discarded rather than kept, so peak memory does not grow with the batch;
    the cost is that each request is parsed again when it runs.
    """
    num_requests = 0
    with open(input_path, encoding="utf-8") as f:
        for request_json in f:
            if request_json.strip():
                BatchRequestInput.model_validate_json(request_json)
                num_requests += 1
    return num_requests


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


@contextlib.asynccontextmanager
async def batch_output_writer(path_or_url: str, output_tmp_dir: str | None):
    """
    Yield a file that receives responses as they complete.

    Responses are appended and flushed while the batch runs, so an interrupted
    run leaves the work finished so far in a local output file. A URL
    destination is staged locally and uploaded only once the batch completes,
    so it does not carry that guarantee: a run that fails partway uploads
    nothing.
    """
    if not path_or_url.startswith(("http://", "https://")):
        logger.info("Writing outputs to local file %s", path_or_url)
        with open(path_or_url, "w", encoding="utf-8") as f:
            yield f
        return

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_tmp_dir,
        prefix="tmp_batch_output_",
        suffix=".jsonl",
    ) as f:
        logger.info("Writing outputs to temporary local file %s", f.name)
        yield f
        f.flush()
        logger.info("Uploading outputs to %s", path_or_url)
        await upload_data(path_or_url, f.name, from_file=True)


async def download_bytes_from_url(
    url: str,
    allowed_media_domains: list[str] | None = None,
) -> bytes:
    """
    Download data from a URL or decode from a data URL.

    Args:
        url: Either an HTTP/HTTPS URL or a data URL (data:...;base64,...)
        allowed_media_domains: If set, only HTTP/HTTPS URLs whose hostname
            is in this list are permitted. data: URLs are not subject to
            this restriction.

    Returns:
        Data as bytes
    """
    parsed = urlparse(url)

    # Handle data URLs (base64 encoded) - not subject to domain restrictions
    if parsed.scheme == "data":
        # Format: data:...;base64,<base64_data>
        if "," in url:
            header, data = url.split(",", 1)
            if "base64" in header:
                return base64.b64decode(data)
            else:
                raise ValueError(f"Unsupported data URL encoding: {header}")
        else:
            raise ValueError(f"Invalid data URL format: {url}")

    # Handle HTTP/HTTPS URLs
    elif parsed.scheme in ("http", "https"):
        if allowed_media_domains is not None:
            url_spec = parse_url(url)
            if url_spec.hostname not in allowed_media_domains:
                raise ValueError(
                    f"The URL must be from one of the allowed domains: "
                    f"{allowed_media_domains}. Input URL domain: "
                    f"{url_spec.hostname}"
                )
            # Use the normalized URL to prevent parsing discrepancies
            # between urllib3 and aiohttp (e.g. backslash-@ attacks).
            url = url_spec.url

        return await global_http_connection.async_get_bytes(
            url, allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS
        )

    else:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme}. "
            "Supported schemes: http, https, data"
        )


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
) -> BatchRequestOutput:
    try:
        response = await serving_engine_func(request.body)
    except Exception as e:
        response = create_error_response(e)

    if isinstance(response, JSONResponse):
        with contextlib.suppress(pydantic.ValidationError):
            response = TypeAdapter(AllResponse | ErrorResponse).validate_python(
                json.loads(response.body)
            )

    if isinstance(response, AllResponse):
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

    return batch_output


WrapperFn: TypeAlias = Callable[[Callable], Callable]


class EndpointConfig(TypedDict):
    """How a batch request URL is matched to the handler that serves it."""

    url: str
    # Score and rerank are served at several versioned paths, so they accept
    # any URL ending in theirs. Every other endpoint matches exactly.
    match_suffix: bool
    handler_getter: Callable[[], Callable | None]
    wrapper_fn: WrapperFn | None


def url_matches(config: EndpointConfig, url: str) -> bool:
    """Whether a request URL is served by this endpoint."""
    if config["match_suffix"]:
        return url.endswith(config["url"])
    return url == config["url"]


def handle_endpoint_request(
    request: BatchRequestInput,
    config: EndpointConfig,
) -> Awaitable[BatchRequestOutput] | None:
    """
    Generic handler for endpoint requests.

    Args:
        request: The batch request input
        config: The endpoint's URL matching rule and handler

    Returns:
        Awaitable[BatchRequestOutput] if the request was handled,
        None if URL didn't match
    """
    if not url_matches(config, request.url):
        return None

    handler_fn = config["handler_getter"]()
    if handler_fn is None:
        error_msg = f"Model does not support endpoint: {request.url}"
        return make_async_error_request_output(request, error_msg=error_msg)

    # Apply wrapper if provided (e.g., for transcriptions/translations)
    wrapper_fn = config["wrapper_fn"]
    if wrapper_fn is not None:
        handler_fn = wrapper_fn(handler_fn)

    return run_request(handler_fn, request)


def make_transcription_wrapper(
    is_translation: bool,
    allowed_media_domains: list[str] | None = None,
) -> WrapperFn:
    """
    Factory function to create a wrapper for transcription/translation handlers.
    The wrapper converts BatchTranscriptionRequest or BatchTranslationRequest
    to TranscriptionRequest or TranslationRequest and calls the appropriate handler.

    Args:
        is_translation: If True, process as translation; otherwise process
            as transcription
        allowed_media_domains: If set, only URLs from these domains are
            permitted for HTTP/HTTPS fetches.

    Returns:
        A function that takes a handler and returns a wrapped handler
    """

    def wrapper(handler_fn: Callable):
        async def transcription_wrapper(
            batch_request_body: (BatchTranscriptionRequest | BatchTranslationRequest),
        ) -> (
            TranscriptionResponse
            | TranscriptionResponseVerbose
            | TranslationResponse
            | TranslationResponseVerbose
            | ErrorResponse
        ):
            try:
                # Download data from URL
                audio_data = await download_bytes_from_url(
                    batch_request_body.file_url,
                    allowed_media_domains=allowed_media_domains,
                )

                # Create a mock file from the downloaded audio data
                mock_file = UploadFile(
                    file=BytesIO(audio_data),
                    filename="audio.bin",
                )

                # Convert batch request to regular request
                # by copying all fields except file_url and setting file to mock_file
                request_dict = batch_request_body.model_dump(exclude={"file_url"})
                request_dict["file"] = mock_file

                if is_translation:
                    # Create TranslationRequest from BatchTranslationRequest
                    translation_request = TranslationRequest.model_validate(
                        request_dict
                    )
                    return await handler_fn(audio_data, translation_request)
                else:
                    # Create TranscriptionRequest from BatchTranscriptionRequest
                    transcription_request = TranscriptionRequest.model_validate(
                        request_dict
                    )
                    return await handler_fn(audio_data, transcription_request)
            except Exception as e:
                operation = "translation" if is_translation else "transcription"
                return ErrorResponse(
                    error=ErrorInfo(
                        message=f"Failed to process {operation}: {str(e)}",
                        type="BadRequestError",
                        code=HTTPStatus.BAD_REQUEST.value,
                    )
                )

        return transcription_wrapper

    return wrapper


async def build_endpoint_registry(
    engine_client: EngineClient,
    args: Namespace,
) -> dict[str, EndpointConfig]:
    """
    Build the endpoint registry with all serving objects and handler configurations.

    Args:
        engine_client: The engine client
        args: Command line arguments

    Returns:
        Dictionary mapping endpoint keys to their configurations
    """
    supported_tasks = await engine_client.get_supported_tasks()
    logger.info("Supported tasks: %s", supported_tasks)

    # Create a state object to hold serving objects
    state = State()

    # Initialize all serving objects using init_app_state
    # This provides full functionality including chat template processing,
    # LoRA support, tool servers, etc.
    await init_app_state(engine_client, state, args, supported_tasks)

    # Get serving objects from state (defaulting to None if not set)
    openai_serving_chat = getattr(state, "openai_serving_chat", None)
    openai_serving_transcription = getattr(state, "openai_serving_transcription", None)
    openai_serving_translation = getattr(state, "openai_serving_translation", None)
    serving_embedding = getattr(state, "serving_embedding", None)
    serving_scores = getattr(state, "serving_scores", None)

    allowed_media_domains = getattr(args, "allowed_media_domains", None)

    # Registry of endpoint configurations
    endpoint_registry: dict[str, EndpointConfig] = {
        "completions": {
            "url": "/v1/chat/completions",
            "match_suffix": False,
            "handler_getter": lambda: (
                openai_serving_chat.create_chat_completion
                if openai_serving_chat is not None
                else None
            ),
            "wrapper_fn": None,
        },
        "embeddings": {
            "url": "/v1/embeddings",
            "match_suffix": False,
            "handler_getter": lambda: (
                serving_embedding if serving_embedding is not None else None
            ),
            "wrapper_fn": None,
        },
        "score": {
            "url": "/score",
            "match_suffix": True,
            "handler_getter": lambda: (
                serving_scores if serving_scores is not None else None
            ),
            "wrapper_fn": None,
        },
        "rerank": {
            "url": "/rerank",
            "match_suffix": True,
            "handler_getter": lambda: (
                serving_scores if serving_scores is not None else None
            ),
            "wrapper_fn": None,
        },
        "transcriptions": {
            "url": "/v1/audio/transcriptions",
            "match_suffix": False,
            "handler_getter": lambda: (
                openai_serving_transcription.create_transcription
                if openai_serving_transcription is not None
                else None
            ),
            "wrapper_fn": make_transcription_wrapper(
                is_translation=False,
                allowed_media_domains=allowed_media_domains,
            ),
        },
        "translations": {
            "url": "/v1/audio/translations",
            "match_suffix": False,
            "handler_getter": lambda: (
                openai_serving_translation.create_translation
                if openai_serving_translation is not None
                else None
            ),
            "wrapper_fn": make_transcription_wrapper(
                is_translation=True,
                allowed_media_domains=allowed_media_domains,
            ),
        },
    }

    return endpoint_registry


def is_same_local_file(input_file: str, output_file: str) -> bool:
    """Whether both paths name the same file, through symlinks or aliases."""
    if input_file.startswith(("http://", "https://")) or output_file.startswith(
        ("http://", "https://")
    ):
        return False
    if not (os.path.exists(input_file) and os.path.exists(output_file)):
        return False
    return os.path.samefile(input_file, output_file)


def validate_run_batch_args(args):
    if args.max_inflight < 1:
        raise ValueError(f"--max-inflight must be at least 1, got {args.max_inflight}.")

    if is_same_local_file(args.input_file, args.output_file):
        raise ValueError(
            f"--input-file and --output-file are the same file ({args.output_file}). "
            "Responses are written while the batch runs, so the input would be "
            "truncated before it has been read."
        )

    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )


async def run_one_request(
    request_json: str,
    endpoint_registry: dict[str, EndpointConfig],
) -> BatchRequestOutput:
    """Route a single line of the batch to its endpoint handler."""
    request = BatchRequestInput.model_validate_json(request_json)

    # Use the last segment of the URL as the endpoint key. The endpoint's own
    # rule decides whether the full URL is one it serves.
    endpoint_key = request.url.split("/")[-1]

    result = None
    if endpoint_key in endpoint_registry:
        result = handle_endpoint_request(request, endpoint_registry[endpoint_key])

    if result is None:
        result = make_async_error_request_output(
            request,
            error_msg=f"URL {request.url} is not a supported endpoint. "
            "Supported endpoints: "
            + ", ".join(config["url"] for config in endpoint_registry.values())
            + ".",
        )

    return await result


def write_finished(
    finished: Iterable[asyncio.Task[BatchRequestOutput]],
    output_file: IO[str],
    tracker: BatchProgressTracker,
) -> BaseException | None:
    """
    Write each response that completed, returning the first failure if any.

    Requests finish in groups, so a failure must not discard the responses that
    succeeded alongside it.
    """
    failure: BaseException | None = None
    for task in finished:
        error = task.exception()
        if error is not None:
            failure = failure or error
            continue
        print(task.result().model_dump_json(), file=output_file)
        tracker.completed()
    output_file.flush()
    return failure


async def dispatch_batch(
    input_path: str,
    output_file: IO[str],
    endpoint_registry: dict[str, EndpointConfig],
    tracker: BatchProgressTracker,
    max_inflight: int,
) -> None:
    """
    Run every request in the batch, writing responses as they finish.

    At most ``max_inflight`` requests are alive at once, so neither the parsed
    requests nor their responses accumulate for the whole batch.
    """
    pending: set[asyncio.Task[BatchRequestOutput]] = set()

    try:
        with open(input_path, encoding="utf-8") as f:
            requests = (line for line in f if line.strip())
            while True:
                # Top up; the generator resumes where the previous round left off.
                for request_json in requests:
                    pending.add(
                        asyncio.create_task(
                            run_one_request(request_json, endpoint_registry)
                        )
                    )
                    if len(pending) >= max_inflight:
                        break

                if not pending:
                    break

                finished, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                failure = write_finished(finished, output_file, tracker)
                if failure is not None:
                    raise failure
    finally:
        # An error leaves the rest of the batch in flight. Drop it, and wait for
        # the cancellations so the requests do not outlive the run.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def run_batch(
    engine_client: EngineClient,
    args: Namespace,
) -> None:
    endpoint_registry = await build_endpoint_registry(
        engine_client=engine_client,
        args=args,
    )

    tracker = BatchProgressTracker()
    logger.info("Reading batch from %s...", args.input_file)

    async with local_input_path(args.input_file, args.output_tmp_dir) as input_path:
        num_requests = validate_batch(input_path)

        async with batch_output_writer(
            args.output_file, args.output_tmp_dir
        ) as output_file:
            with tracker.pbar(total=num_requests):
                await dispatch_batch(
                    input_path,
                    output_file,
                    endpoint_registry,
                    tracker,
                    args.max_inflight,
                )


async def main(args: Namespace):
    from vllm.entrypoints.openai.api_server import build_async_engine_client
    from vllm.usage.usage_lib import UsageContext

    validate_run_batch_args(args)

    async with build_async_engine_client(
        args,
        usage_context=UsageContext.OPENAI_BATCH_RUNNER,
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
        start_http_server(port=args.port, addr=args.host)
    else:
        logger.info("Prometheus metrics disabled")

    asyncio.run(main(args))
