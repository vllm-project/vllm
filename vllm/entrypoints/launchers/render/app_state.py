# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from starlette.datastructures import State

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.scale_out.factories import init_render_state
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.plugins.endpoint_plugins.interface import init_endpoint_plugins_state
from vllm.renderers import renderer_from_config
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer


async def init_render_app_state(
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
) -> None:
    """Initialise FastAPI app state for a CPU-only render server.

    Unlike :func:`init_app_state` this function does not require an
    :class:`~vllm.engine.protocol.EngineClient`; it bootstraps the
    preprocessing pipeline (renderer, input_processor)
    directly from the :class:`~vllm.config.VllmConfig`.
    """

    served_model_names = args.served_model_name or [args.model]
    model_registry = OpenAIModelRegistry(
        model_config=vllm_config.model_config,
        base_model_paths=[
            BaseModelPath(name=name, model_path=args.model)
            for name in served_model_names
        ],
    )

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    renderer = renderer_from_config(vllm_config)
    resolved_chat_template = load_chat_template(args.chat_template)

    state.online_renderer = OnlineRenderer(
        model_config=vllm_config.model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )
    state.online_renderer.warmup()

    state.online_derenderer = OnlineDerenderer(
        model_config=vllm_config.model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_models = model_registry
    state.serving_tokenization = ServingTokenization(
        model_registry,
        state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    init_render_state(state, request_logger)

    state.vllm_config = vllm_config
    # Disable stats logging — there is no engine to poll.
    state.log_stats = False
    state.engine_client = None
    state.args = args
    state.enable_server_load_tracking = False
    state.server_load_metrics = 0

    # No `EngineClient` exists for the render server, so plugins get `None` and
    # must handle it themselves (see `EndpointPlugin.init_state`).
    await init_endpoint_plugins_state(None, state, args)
