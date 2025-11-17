# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Command-line entry point for vLLM gRPC server."""

import argparse
import asyncio
from argparse import Namespace

import uvloop

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.grpc.server import run_grpc_server
from vllm.entrypoints.utils import (
    cli_env_setup,
    process_chat_template,
    process_lora_modules,
)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


async def create_serving_chat(
    engine_client: EngineClient,
    args: Namespace,
) -> OpenAIServingChat:
    """
    创建 OpenAIServingChat 实例。

    复用 init_app_state 中的逻辑。
    """
    vllm_config = engine_client.vllm_config

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

    resolved_chat_template = await process_chat_template(
        args.chat_template, engine_client, vllm_config.model_config
    )

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await openai_serving_models.init_static_loras()

    if "generate" not in supported_tasks:
        raise ValueError("Model does not support generation task")

    serving_chat = OpenAIServingChat(
        engine_client,
        openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        log_error_stack=args.log_error_stack,
    )

    return serving_chat


async def run_grpc_server_main(args: Namespace) -> None:
    """
    运行 gRPC 服务器主函数。
    """
    async with build_async_engine_client(
        args,
        usage_context=UsageContext.OPENAI_API_SERVER,
    ) as engine_client:
        serving_chat = await create_serving_chat(engine_client, args)

        await run_grpc_server(
            engine_client,
            serving_chat,
            args,
            host=args.grpc_host,
            port=args.grpc_port,
        )


def parse_args() -> Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="vLLM gRPC Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 复用 OpenAI API server 的参数
    parser = make_arg_parser(parser)

    # gRPC 特定参数
    parser.add_argument(
        "--grpc-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind gRPC server",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=8033,
        help="Port for gRPC server",
    )

    return parser.parse_args()


def main() -> None:
    """
    主入口函数。
    """
    cli_env_setup()
    args = parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_grpc_server_main(args))


if __name__ == "__main__":
    main()
