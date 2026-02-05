# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import tempfile
from argparse import Namespace
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from io import BytesIO, StringIO
from typing import Any, TypeAlias
from urllib.parse import urlparse

import aiohttp
import torch
from fastapi import UploadFile
from prometheus_client import start_http_server
from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorInfo,
    ErrorResponse,
    OpenAIBaseModel,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.translations.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)
from vllm.entrypoints.openai.translations.serving import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
)
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
    file: UploadFile | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_no_file(cls, data: Any):
        """Ensure file field is not provided in batch requests."""
        if isinstance(data, dict) and "file" in data:
            raise ValueError(
                "The 'file' field is not supported in batch requests. "
                "Use 'file_url' instead."
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
    file: UploadFile | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_no_file(cls, data: Any):
        """Ensure file field is not provided in batch requests."""
        if isinstance(data, dict) and "file" in data:
            raise ValueError(
                "The 'file' field is not supported in batch requests. "
                "Use 'file_url' instead."
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
        | TranscriptionResponse
        | TranscriptionResponseVerbose
        | TranslationResponse
        | TranslationResponseVerbose
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


class BatchProgressTracker:
    def __init__(self):
        self._total = 0
        self._pbar: tqdm | None = None

    def submitted(self):
        self._total += 1

    def completed(self):
        if self._pbar:
            self._pbar.update()

    def pbar(self) -> tqdm:
        enable_tqdm = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        self._pbar = tqdm(
            total=self._total,
            unit="req",
            desc="Running batch",
            mininterval=5,
            disable=not enable_tqdm,
            bar_format=_BAR_FORMAT,
        )
        return self._pbar


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


async def download_bytes_from_url(url: str) -> bytes:
    """
    Download data from a URL or decode from a data URL.

    Args:
        url: Either an HTTP/HTTPS URL or a data URL (data:...;base64,...)

    Returns:
        Data as bytes
    """
    parsed = urlparse(url)

    # Handle data URLs (base64 encoded)
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
        async with (
            aiohttp.ClientSession() as session,
            session.get(url) as resp,
        ):
            if resp.status != 200:
                raise Exception(
                    f"Failed to download data from URL: {url}. Status: {resp.status}"
                )
            return await resp.read()

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
    tracker: BatchProgressTracker,
) -> BatchRequestOutput:
    response = await serving_engine_func(request.body)

    if isinstance(
        response,
        (
            ChatCompletionResponse,
            EmbeddingResponse,
            ScoreResponse,
            RerankResponse,
            TranscriptionResponse,
            TranscriptionResponseVerbose,
            TranslationResponse,
            TranslationResponseVerbose,
        ),
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

    tracker.completed()
    return batch_output


def handle_endpoint_request(
    request: BatchRequestInput,
    tracker: BatchProgressTracker,
    url_matcher: Callable[[str], bool],
    handler_getter: Callable[[], Callable | None],
    wrapper_fn: Callable[[Callable], Callable] | None = None,
) -> Awaitable[BatchRequestOutput] | None:
    """
    Generic handler for endpoint requests that eliminates code duplication.

    Args:
        request: The batch request input
        tracker: Progress tracker for the batch
        url_matcher: Function that takes a URL and returns True if it matches
        handler_getter: Function that returns the handler function or None
        wrapper_fn: Optional function to wrap the handler (e.g., for transcriptions)

    Returns:
        Awaitable[BatchRequestOutput] if the request was handled,
        None if URL didn't match
    """
    if not url_matcher(request.url):
        return None

    handler_fn = handler_getter()
    if handler_fn is None:
        error_msg = f"Model does not support endpoint: {request.url}"
        return make_async_error_request_output(request, error_msg=error_msg)

    # Apply wrapper if provided (e.g., for transcriptions/translations)
    if wrapper_fn is not None:
        handler_fn = wrapper_fn(handler_fn)

    tracker.submitted()
    return run_request(handler_fn, request, tracker)


def make_transcription_wrapper(is_translation: bool):
    """
    Factory function to create a wrapper for transcription/translation handlers.
    The wrapper converts BatchTranscriptionRequest or BatchTranslationRequest
    to TranscriptionRequest or TranslationRequest and calls the appropriate handler.

    Args:
        is_translation: If True, process as translation; otherwise process
        as transcription

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
                audio_data = await download_bytes_from_url(batch_request_body.file_url)

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


def build_endpoint_registry(
    engine_client: EngineClient,
    args: Namespace,
    base_model_paths: list[BaseModelPath],
    request_logger: RequestLogger | None,
    supported_tasks: set[str],
) -> dict[str, dict[str, Any]]:
    """
    Build the endpoint registry with all serving objects and handler configurations.

    Args:
        engine_client: The engine client
        args: Command line arguments
        base_model_paths: List of base model paths
        request_logger: Optional request logger
        supported_tasks: Set of supported tasks

    Returns:
        Dictionary mapping endpoint keys to their configurations
    """
    model_config = engine_client.model_config

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
            default_chat_template_kwargs=getattr(
                args, "default_chat_template_kwargs", None
            ),
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
            score_template=None,
        )
        if ("embed" in supported_tasks or enable_serving_reranking)
        else None
    )

    openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )

    openai_serving_translation = (
        OpenAIServingTranslation(
            engine_client,
            openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )

    # Registry of endpoint configurations
    endpoint_registry: dict[str, dict[str, Any]] = {
        "completions": {
            "url_matcher": lambda url: url == "/v1/chat/completions",
            "handler_getter": lambda: (
                openai_serving_chat.create_chat_completion
                if openai_serving_chat is not None
                else None
            ),
            "wrapper_fn": None,
        },
        "embeddings": {
            "url_matcher": lambda url: url == "/v1/embeddings",
            "handler_getter": lambda: (
                openai_serving_embedding.create_embedding
                if openai_serving_embedding is not None
                else None
            ),
            "wrapper_fn": None,
        },
        "score": {
            "url_matcher": lambda url: url.endswith("/score"),
            "handler_getter": lambda: (
                openai_serving_scores.create_score
                if openai_serving_scores is not None
                else None
            ),
            "wrapper_fn": None,
        },
        "rerank": {
            "url_matcher": lambda url: url.endswith("/rerank"),
            "handler_getter": lambda: (
                openai_serving_scores.do_rerank
                if openai_serving_scores is not None
                else None
            ),
            "wrapper_fn": None,
        },
        "transcriptions": {
            "url_matcher": lambda url: url == "/v1/audio/transcriptions",
            "handler_getter": lambda: (
                openai_serving_transcription.create_transcription
                if openai_serving_transcription is not None
                else None
            ),
            "wrapper_fn": make_transcription_wrapper(is_translation=False),
        },
        "translations": {
            "url_matcher": lambda url: url == "/v1/audio/translations",
            "handler_getter": lambda: (
                openai_serving_translation.create_translation
                if openai_serving_translation is not None
                else None
            ),
            "wrapper_fn": make_transcription_wrapper(is_translation=True),
        },
    }

    return endpoint_registry


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

    supported_tasks = await engine_client.get_supported_tasks()
    logger.info("Supported tasks: %s", supported_tasks)

    endpoint_registry = build_endpoint_registry(
        engine_client=engine_client,
        args=args,
        base_model_paths=base_model_paths,
        request_logger=request_logger,
        supported_tasks=supported_tasks,
    )

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

        # Find matching endpoint in registry and handle the request
        # Extract the last segment from the URL to use as a key
        url_segments = [seg for seg in request.url.split("/") if seg]
        endpoint_key = url_segments[-1] if url_segments else None

        result = None
        if endpoint_key and endpoint_key in endpoint_registry:
            endpoint_config = endpoint_registry[endpoint_key]
            result = handle_endpoint_request(
                request,
                tracker,
                url_matcher=endpoint_config["url_matcher"],
                handler_getter=endpoint_config["handler_getter"],
                wrapper_fn=endpoint_config["wrapper_fn"],
            )

        if result is not None:
            response_futures.append(result)
        else:
            response_futures.append(
                make_async_error_request_output(
                    request,
                    error_msg=f"URL {request.url} was used. "
                    "Supported endpoints: /v1/chat/completions, /v1/embeddings,"
                    " /v1/audio/transcriptions, /v1/audio/translations, /score, "
                    " /rerank. See vllm/entrypoints/openai/api_server.py "
                    "for supported score/rerank versions.",
                )
            )

    with tracker.pbar():
        responses = await asyncio.gather(*response_futures)

    await write_file(args.output_file, responses, args.output_tmp_dir)


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
