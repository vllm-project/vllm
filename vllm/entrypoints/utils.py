# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import os
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask, BackgroundTasks

from vllm.logger import init_logger

logger = init_logger(__name__)

VLLM_SERVE_PARSER_EPILOG = (
    "Tip: Use `vllm serve --help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help=ModelConfig\n"
    "   - To view a single argument:    --help=max-num-seqs\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup")


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            if request.app.state.enable_server_load_tracking:
                # on timeout/cancellation the BackgroundTask in load_aware_call
                # cannot decrement the server load metrics.
                # Must be decremented by with_cancellation instead.
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

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
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
        raw_request = kwargs.get("raw_request",
                                 args[1] if len(args) > 1 else None)

        if raw_request is None:
            raise ValueError(
                "raw_request required when server load tracking is enabled")

        if not raw_request.app.state.enable_server_load_tracking:
            return await func(*args, **kwargs)

        raw_request.app.state.server_load_metrics += 1
        try:
            response = await func(*args, **kwargs)
        except Exception:
            raw_request.app.state.server_load_metrics -= 1
            raise

        if isinstance(response, (JSONResponse, StreamingResponse)):
            if response.background is None:
                response.background = BackgroundTask(decrement_server_load,
                                                     raw_request)
            elif isinstance(response.background, BackgroundTasks):
                response.background.add_task(decrement_server_load,
                                             raw_request)
            elif isinstance(response.background, BackgroundTask):
                # Convert the single BackgroundTask to BackgroundTasks
                # and chain the decrement_server_load task to it
                tasks = BackgroundTasks()
                tasks.add_task(response.background.func,
                               *response.background.args,
                               **response.background.kwargs)
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
    truncate_prompt_tokens: Optional[int],
    tokenization_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[int]:

    if truncate_prompt_tokens is not None:
        if truncate_prompt_tokens <= -1:
            truncate_prompt_tokens = max_model_len

        if truncate_prompt_tokens > max_model_len:
            raise ValueError(
                f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                f"is greater than max_model_len ({max_model_len})."
                f" Please, select a smaller truncation size.")

        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

    return truncate_prompt_tokens


def show_filtered_argument_or_group_from_help(parser):
    import sys
    for arg in sys.argv:
        if arg.startswith('--help='):
            search_keyword = arg.split('=', 1)[1]

            # List available groups
            if search_keyword == 'listgroup':
                print("\nAvailable argument groups:")
                for group in parser._action_groups:
                    if group.title and not group.title.startswith(
                            "positional arguments"):
                        print(f"  - {group.title}")
                        if group.description:
                            print("    " + group.description.strip())
                        print()
                sys.exit(0)

            # For group search
            formatter = parser._get_formatter()
            for group in parser._action_groups:
                if group.title and group.title.lower() == search_keyword.lower(
                ):
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    print(formatter.format_help())
                    sys.exit(0)

            # For single arg
            matched_actions = []

            for group in parser._action_groups:
                for action in group._group_actions:
                    # search option name
                    if any(search_keyword.lower() in opt.lower()
                           for opt in action.option_strings):
                        matched_actions.append(action)

            if matched_actions:
                print(f"\nParameters matching '{search_keyword}':\n")
                formatter = parser._get_formatter()
                formatter.add_arguments(matched_actions)
                print(formatter.format_help())
                sys.exit(0)

            print(f"\nNo group or parameter matching '{search_keyword}'")
            print("Tip: use `--help=listgroup` to view all groups.")
            sys.exit(1)
