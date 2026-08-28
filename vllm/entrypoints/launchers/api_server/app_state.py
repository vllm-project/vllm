# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from argparse import Namespace
from typing import cast

from starlette.datastructures import State

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import process_lora_modules
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.plugins.endpoint_plugins.interface import init_endpoint_plugins_state
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tasks import FALLBACK_SUPPORTED_TASKS, POOLING_TASKS, SupportedTask


async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
) -> None:
    vllm_config = engine_client.vllm_config

    if args.tool_call_parser is not None:
        from vllm.parser.metrics import init_parser_metrics

        init_parser_metrics(
            model_name=cast(str, vllm_config.model_config.served_model_name)
        )

    if supported_tasks is None:
        warnings.warn(
            "The 'supported_tasks' parameter was not provided to "
            "init_app_state and will be required in a future version. "
            "Please pass 'supported_tasks' explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        supported_tasks = FALLBACK_SUPPORTED_TASKS

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

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    state.args = args
    resolved_chat_template = load_chat_template(args.chat_template)

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()

    state.online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )
    state.online_renderer.warmup()

    state.online_derenderer = OnlineDerenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.serving_tokenization = ServingTokenization(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    if "generate" in supported_tasks:
        from vllm.entrypoints.generate.api_router import init_generate_state

        await init_generate_state(
            engine_client, state, args, request_logger, supported_tasks
        )

        from vllm.entrypoints.scale_out.factories import init_scale_out_state

        init_scale_out_state(state, args, engine_client, request_logger)

    if "transcription" in supported_tasks or "realtime" in supported_tasks:
        from vllm.entrypoints.speech_to_text.factories import init_speech_to_text_state

        init_speech_to_text_state(
            engine_client, state, args, request_logger, supported_tasks
        )

    if any(task in POOLING_TASKS for task in supported_tasks):
        from vllm.entrypoints.pooling.factories import init_pooling_state

        init_pooling_state(engine_client, state, args, request_logger, supported_tasks)

    await init_endpoint_plugins_state(engine_client, state, args)

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0
