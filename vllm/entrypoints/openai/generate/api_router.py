# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object


def register_generate_api_routers(app: FastAPI):
    from vllm.entrypoints.openai.chat_completion.api_router import (
        attach_router as register_chat_api_router,
    )

    register_chat_api_router(app)

    from vllm.entrypoints.openai.responses.api_router import (
        attach_router as register_responses_api_router,
    )

    register_responses_api_router(app)

    from vllm.entrypoints.openai.completion.api_router import (
        attach_router as register_completion_api_router,
    )

    register_completion_api_router(app)

    from vllm.entrypoints.anthropic.api_router import (
        attach_router as register_anthropic_api_router,
    )

    register_anthropic_api_router(app)


async def init_generate_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    from vllm.entrypoints.chat_utils import (
        UsagePolicy,
        load_chat_template,
    )
    from vllm.entrypoints.mcp.tool_server import (
        DemoToolServer,
        MCPToolServer,
        ToolServer,
    )
    from vllm.entrypoints.openai.chat_completion.batch_serving import (
        OpenAIServingChatBatch,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.fingerprint import set_default_fingerprint_mode
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
    from vllm.entrypoints.serve.disagg.serving import ServingTokens

    # Applied before any serving class is constructed so that each one picks
    # up the chosen mode on its first cache miss.
    set_default_fingerprint_mode(
        getattr(args, "fingerprint_mode", "full"),
        getattr(args, "fingerprint_value", None),
    )

    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None
    resolved_chat_template = load_chat_template(args.chat_template)

    # Build UsagePolicy from CLI args
    usage_policy = UsagePolicy(
        include_usage=args.include_usage_policy,
        continuous_usage=args.continuous_usage_policy,
    )

    # ---
    # Deprecated --enable-force-include-usage → UsagePolicy conversion.
    #
    # If the old flag is set and the user has not already configured
    # --include-usage-policy / --continuous-usage-policy, translate the
    # per-endpoint historical behaviour into the new policy mechanism.
    # Once the flag is fully removed this helper and its call sites can
    # be deleted.
    # ---

    def _maybe_force_usage(
        policy: UsagePolicy,
        enable_force: bool,
        *,
        set_continuous: bool,
    ) -> UsagePolicy:
        if enable_force and policy.include_usage is None:
            return UsagePolicy(
                include_usage="always",
                continuous_usage="always" if set_continuous else None,
            )
        return policy

    # Chat/Completion/Anthropic: flag → (True, True)
    usage_policy = _maybe_force_usage(
        usage_policy, args.enable_force_include_usage, set_continuous=True
    )
    # Responses: flag was stored but never used → no conversion needed
    # Disagg: flag has no effect → no conversion needed

    # Render endpoints are always backed by OpenAIServingRender so that
    # /v1/chat/completions/render and /v1/completions/render work on both
    # generate-mode and render-only servers. Created in init_app_state.

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            usage_policy=usage_policy,
            enable_log_outputs=args.enable_log_outputs,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        openai_serving_render=state.openai_serving_render,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        usage_policy=usage_policy,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
    )
    state.openai_serving_chat = (
        OpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    )
    state.openai_serving_chat_batch = (
        OpenAIServingChatBatch(**_chat_kwargs)
        if "generate" in supported_tasks
        else None
    )
    if state.openai_serving_chat is not None:
        state.openai_serving_chat.warmup()
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            usage_policy=usage_policy,
        )
        if "generate" in supported_tasks
        else None
    )
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            usage_policy=usage_policy,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
            usage_policy=usage_policy,
        )
        if "generate" in supported_tasks
        else None
    )
