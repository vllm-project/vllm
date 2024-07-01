"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl

from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split('=')
            lora_list.append(LoRAModulePath(name, path))
        setattr(namespace, self.dest, lora_list)


def make_arg_parser():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host",
                        type=nullable_str,
                        default=None,
                        help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="log level for uvicorn")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--api-key",
                        type=nullable_str,
                        default=None,
                        help="If provided, the server will require this key "
                        "to be presented in the header.")
    parser.add_argument(
        "--lora-modules",
        type=nullable_str,
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help="LoRA module configurations in the format name=path. "
        "Multiple modules can be specified.")
    parser.add_argument("--chat-template",
                        type=nullable_str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=nullable_str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=nullable_str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument("--ssl-ca-certs",
                        type=nullable_str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=nullable_str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=nullable_str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server "
        "using @app.middleware('http'). "
        "If a class is provided, vLLM will add it to the server "
        "using app.add_middleware(). ")

    parser.add_argument("--enable-api-tools",
                        action="store_true",
                        help="Enable OpenAI-like tools API "
                             "(only function calls are currently supported)")

    parser.add_argument("--enable-auto-tool-choice",
                        action="store_true",
                        help='Enable auto tool choice for models that support it. '
                             'Requires specifying --tool-use-prompt-template.'
                        )

    parser.add_argument(
        '--tool-use-prompt-template',
        type=str,
        default=None,
        help="The path to the jinja template that should be used to format "
             "any provided OpenAI API-style function definitions into a system prompt "
             "that instructs the model how to use tools, and which tools are "
             "available. If not provided, tools will be ignored. An example is "
             "provided at 'examples/tool_template_hermes_2_pro.jinja'."
    )

    parser.add_argument(
        '--tool-use-prompt-role',
        type=str,
        default='system',
        help='The chat role to use for the system prompt that instructs the model what tools are '
        'available and how to use them. The default is "system". If the "system" role is used for the tool '
        'use system prompt (default) _and_ the client specifies a system prompt, then the client-'
        'specified system prompt will be appended to the tool use system prompt. If a non-"system" role is '
        'specified, it will be placed as the first non-system message in the conversation.'
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser
