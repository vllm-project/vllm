# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import dataclasses
import functools
import os
from argparse import Namespace
from logging import Logger
from string import Template
from typing import TYPE_CHECKING, Any

import regex as re
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask, BackgroundTasks

from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EmbedsPrompt, TokensPrompt
from vllm.logger import current_formatter_type, init_logger
from vllm.platforms import current_platform
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.argparse_utils import FlexibleArgumentParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import (
        StreamOptions,
    )
    from vllm.entrypoints.openai.models.protocol import LoRAModulePath
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = object
    CompletionRequest = object
    StreamOptions = object
    LoRAModulePath = object
    ResponsesRequest = object


logger = init_logger(__name__)

VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            # If load tracking is enabled *and* the counter exists, decrement
            # it. Combines the previous nested checks into a single condition
            # to satisfy the linter rule.
            if getattr(
                request.app.state, "enable_server_load_tracking", False
            ) and hasattr(request.app.state, "server_load_metrics"):
                request.app.state.server_load_metrics -= 1
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait(
            [handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def decrement_server_load(request: Request):
    request.app.state.server_load_metrics -= 1


def load_aware_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        raw_request = kwargs.get("raw_request", args[1] if len(args) > 1 else None)

        if raw_request is None:
            raise ValueError(
                "raw_request required when server load tracking is enabled"
            )

        if not getattr(raw_request.app.state, "enable_server_load_tracking", False):
            return await func(*args, **kwargs)

        # ensure the counter exists
        if not hasattr(raw_request.app.state, "server_load_metrics"):
            raw_request.app.state.server_load_metrics = 0

        raw_request.app.state.server_load_metrics += 1
        try:
            response = await func(*args, **kwargs)
        except Exception:
            raw_request.app.state.server_load_metrics -= 1
            raise

        if isinstance(response, (JSONResponse, StreamingResponse)):
            if response.background is None:
                response.background = BackgroundTask(decrement_server_load, raw_request)
            elif isinstance(response.background, BackgroundTasks):
                response.background.add_task(decrement_server_load, raw_request)
            elif isinstance(response.background, BackgroundTask):
                # Convert the single BackgroundTask to BackgroundTasks
                # and chain the decrement_server_load task to it
                tasks = BackgroundTasks()
                tasks.add_task(
                    response.background.func,
                    *response.background.args,
                    **response.background.kwargs,
                )
                tasks.add_task(decrement_server_load, raw_request)
                response.background = tasks
        else:
            raw_request.app.state.server_load_metrics -= 1

        return response

    return wrapper


def cli_env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _validate_truncation_size(
    max_model_len: int,
    truncate_prompt_tokens: int | None,
    tokenization_kwargs: dict[str, Any] | None = None,
) -> int | None:
    if truncate_prompt_tokens is not None:
        if truncate_prompt_tokens <= -1:
            truncate_prompt_tokens = max_model_len

        if truncate_prompt_tokens > max_model_len:
            raise ValueError(
                f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                f"is greater than max_model_len ({max_model_len})."
                f" Please, select a smaller truncation size."
            )

        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

    else:
        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = False

    return truncate_prompt_tokens


def get_max_tokens(
    max_model_len: int,
    request: "CompletionRequest | ChatCompletionRequest | ResponsesRequest",
    prompt: TokensPrompt | EmbedsPrompt,
    default_sampling_params: dict,
) -> int:
    # NOTE: Avoid isinstance() for better efficiency
    max_tokens: int | None = next(
        (
            val
            for attr in [
                # ChatCompletionRequest
                "max_completion_tokens",
                # ResponsesRequest
                "max_output_tokens",
                # CompletionRequest (also a fallback for ChatCompletionRequest)
                "max_tokens",
            ]
            if (val := getattr(request, attr, None)) is not None
        ),
        None,
    )

    input_length = length_from_prompt_token_ids_or_embeds(
        prompt.get("prompt_token_ids"),  # type: ignore[arg-type]
        prompt.get("prompt_embeds"),  # type: ignore[arg-type]
    )
    default_max_tokens = max_model_len - input_length
    max_output_tokens = current_platform.get_max_output_tokens(input_length)

    return min(
        val
        for val in (
            default_max_tokens,
            max_tokens,
            max_output_tokens,
            default_sampling_params.get("max_tokens"),
        )
        if val is not None
    )


def log_non_default_args(args: Namespace | EngineArgs):
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    non_default_args = {}

    # Handle Namespace
    if isinstance(args, Namespace):
        parser = make_arg_parser(FlexibleArgumentParser())
        for arg, default in vars(parser.parse_args([])).items():
            if default != getattr(args, arg):
                non_default_args[arg] = getattr(args, arg)

    # Handle EngineArgs instance
    elif isinstance(args, EngineArgs):
        default_args = EngineArgs(model=args.model)  # Create default instance
        for field in dataclasses.fields(args):
            current_val = getattr(args, field.name)
            default_val = getattr(default_args, field.name)
            if current_val != default_val:
                non_default_args[field.name] = current_val
        if default_args.model != EngineArgs.model:
            non_default_args["model"] = default_args.model
    else:
        raise TypeError(
            "Unsupported argument type. Must be Namespace or EngineArgs instance."
        )

    logger.info("non-default args: %s", non_default_args)


def should_include_usage(
    stream_options: "StreamOptions | None", enable_force_include_usage: bool
) -> tuple[bool, bool]:
    if stream_options:
        include_usage = stream_options.include_usage or enable_force_include_usage
        include_continuous_usage = include_usage and bool(
            stream_options.continuous_usage_stats
        )
    else:
        include_usage, include_continuous_usage = enable_force_include_usage, False
    return include_usage, include_continuous_usage


def process_lora_modules(
    args_lora_modules: list[LoRAModulePath], default_mm_loras: dict[str, str] | None
) -> list[LoRAModulePath]:
    from vllm.entrypoints.openai.models.serving import LoRAModulePath

    lora_modules = args_lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            )
            for modality, lora_path in default_mm_loras.items()
        ]
        if args_lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths
    return lora_modules


def sanitize_message(message: str) -> str:
    # Avoid leaking memory address from object reprs
    return re.sub(r" at 0x[0-9a-f]+>", ">", message)


def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    if envs.VLLM_DISABLE_LOG_LOGO or (formatter := current_formatter_type(lgr)) is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}  model   ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, version, model_name)
