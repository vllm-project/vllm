# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.config import ModelConfig
from vllm.tasks import SupportedTask

if TYPE_CHECKING:
    from vllm.entrypoints.sagemaker.api_router import (
        EndpointFn,
        GetHandlerFn,
        RequestType,
    )


def get_generate_invocation_types(
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    # NOTE: Items defined earlier take higher priority
    invocation_types: list[tuple[RequestType, tuple[GetHandlerFn, EndpointFn]]] = []

    if "generate" in supported_tasks:
        from vllm.entrypoints.openai.chat_completion.api_router import (
            chat,
            create_chat_completion,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
        from vllm.entrypoints.openai.completion.api_router import (
            completion,
            create_completion,
        )
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest

        invocation_types += [
            (ChatCompletionRequest, (chat, create_chat_completion)),
            (CompletionRequest, (completion, create_completion)),
        ]

    return invocation_types
