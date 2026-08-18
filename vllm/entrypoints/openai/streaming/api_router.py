# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State setup for video streaming serving.

Exposed via ``/v1/chat/completions``: the chat handler dispatches streaming
requests carrying a stream-scheme ``video_url`` to
:class:`VideoStreamingServing`, so there is no route to register.
"""

from typing import TYPE_CHECKING

from vllm.entrypoints.openai.streaming.serving import VideoStreamingServing
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object


def init_streaming_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: "RequestLogger | None",
    supported_tasks: "tuple[SupportedTask, ...]",
) -> None:
    state.video_streaming_serving = (
        VideoStreamingServing(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if "generate" in supported_tasks
        else None
    )
