# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from starlette.datastructures import State

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launchers.cli_args import resolve_default_chat_template_kwargs
from vllm.entrypoints.mcp.tool_server import init_tool_server
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.scale_out.factories import init_render_state
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.plugins.endpoint_plugins.interface import init_endpoint_plugins_state
from vllm.renderers import renderer_from_config
from vllm.renderers.multi_model import (
    MultiModelOnlineDerenderer,
    MultiModelOnlineRenderer,
)
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer


def _build_online_renderer(
    model_config: ModelConfig,
    args: Namespace,
    request_logger: RequestLogger | None,
    chat_template: str | None,
    default_chat_template_kwargs: dict,
) -> OnlineRenderer:
    vllm_config = VllmConfig(model_config=model_config)
    renderer = renderer_from_config(vllm_config)
    return OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        trust_request_mm_kwargs=args.trust_request_mm_kwargs,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=args.reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )


def _build_online_derenderer(
    model_config: ModelConfig,
    args: Namespace,
    request_logger: RequestLogger | None,
    chat_template: str | None,
    default_chat_template_kwargs: dict,
) -> OnlineDerenderer:
    vllm_config = VllmConfig(model_config=model_config)
    renderer = renderer_from_config(vllm_config)
    return OnlineDerenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=args.reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )


async def init_render_app_state(
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
    extra_model_configs: dict[str, ModelConfig] | None = None,
) -> None:
    """Initialise FastAPI app state for a CPU-only render server.

    Unlike :func:`init_app_state` this function does not require an
    :class:`~vllm.engine.protocol.EngineClient`; it bootstraps the
    preprocessing pipeline (renderer, input_processor) directly from the
    :class:`~vllm.config.VllmConfig`.

    When `extra_model_configs` is non-empty, one renderer/derenderer is built
    per model and wrapped in a `MultiModelOnlineRenderer` dispatcher. The
    primary model — the one passed via `--model` — remains the fallback that
    every legacy code path resolves to when the request omits `model`.
    """
    primary_served_names = args.served_model_name or [args.model]
    primary_name = primary_served_names[0]
    extra_model_configs = extra_model_configs or {}
    conflicts = set(extra_model_configs) & set(primary_served_names)
    assert not conflicts, (
        f"--extra-served-model names {sorted(conflicts)} collide with primary "
        f"served names {sorted(primary_served_names)}"
    )

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in primary_served_names
    ]
    model_registry = OpenAIModelRegistry(
        model_config=vllm_config.model_config,
        base_model_paths=base_model_paths,
        extra_model_configs=extra_model_configs or None,
    )

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    resolved_chat_template = load_chat_template(args.chat_template)
    default_chat_template_kwargs = resolve_default_chat_template_kwargs(args)
    state.tool_server = await init_tool_server(args)

    # Build one renderer/derenderer per served model. Every primary alias
    # points at the same primary renderer instance so alias lookup is O(1)
    # and reuses the same tokenizer/processor.
    renderers: dict[str, OnlineRenderer] = {}
    derenderers: dict[str, OnlineDerenderer] = {}
    primary_renderer = _build_online_renderer(
        vllm_config.model_config,
        args,
        request_logger,
        resolved_chat_template,
        default_chat_template_kwargs,
    )
    primary_derenderer = _build_online_derenderer(
        vllm_config.model_config,
        args,
        request_logger,
        resolved_chat_template,
        default_chat_template_kwargs,
    )
    for name in primary_served_names:
        renderers[name] = primary_renderer
        derenderers[name] = primary_derenderer
    for name, cfg in extra_model_configs.items():
        renderers[name] = _build_online_renderer(
            cfg,
            args,
            request_logger,
            resolved_chat_template,
            default_chat_template_kwargs,
        )
        derenderers[name] = _build_online_derenderer(
            cfg,
            args,
            request_logger,
            resolved_chat_template,
            default_chat_template_kwargs,
        )

    dispatcher = MultiModelOnlineRenderer(primary_name, renderers)
    derender_dispatcher = MultiModelOnlineDerenderer(primary_name, derenderers)
    dispatcher.warmup()

    # Legacy fields keep pointing at the primary renderer so single-model
    # code paths that predate multi-model support keep working unchanged.
    state.online_renderer = primary_renderer
    state.online_derenderer = primary_derenderer
    # New fields expose the full multi-model view.
    state.online_renderers = dispatcher
    state.online_derenderers = derender_dispatcher

    state.openai_serving_models = model_registry
    state.serving_tokenization = ServingTokenization(
        model_registry,
        dispatcher,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=default_chat_template_kwargs,
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
