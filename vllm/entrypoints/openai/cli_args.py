# SPDX-License-Identifier: Apache-2.0
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
from collections.abc import Sequence
from typing import Optional, Union, get_args

from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         validate_chat_template)
from vllm.entrypoints.openai.serving_models import (LoRAModulePath,
                                                    PromptAdapterPath)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class LoRAParserAction(argparse.Action):

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Optional[Union[str, Sequence[str]]],
        option_string: Optional[str] = None,
    ):
        if values is None:
            values = []
        if isinstance(values, str):
            raise TypeError("Expected values to be a list")

        lora_list: list[LoRAModulePath] = []
        for item in values:
            if item in [None, '']:  # Skip if item is None or empty string
                continue
            if '=' in item and ',' not in item:  # Old format: name=path
                name, path = item.split('=')
                lora_list.append(LoRAModulePath(name, path))
            else:  # Assume JSON format
                try:
                    lora_dict = json.loads(item)
                    lora = LoRAModulePath(**lora_dict)
                    lora_list.append(lora)
                except json.JSONDecodeError:
                    parser.error(
                        f"Invalid JSON format for --lora-modules: {item}")
                except TypeError as e:
                    parser.error(
                        f"Invalid fields for --lora-modules: {item} - {str(e)}"
                    )
        setattr(namespace, self.dest, lora_list)


class PromptAdapterParserAction(argparse.Action):

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Optional[Union[str, Sequence[str]]],
        option_string: Optional[str] = None,
    ):
        if values is None:
            values = []
        if isinstance(values, str):
            raise TypeError("Expected values to be a list")

        adapter_list: list[PromptAdapterPath] = []
        for item in values:
            name, path = item.split('=')
            adapter_list.append(PromptAdapterPath(name, path))
        setattr(namespace, self.dest, adapter_list)


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--host",
                        type=optional_type(str),
                        default=None,
                        help="Host name.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="Log level for uvicorn.")
    parser.add_argument("--disable-uvicorn-access-log",
                        action="store_true",
                        help="Disable uvicorn access log.")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="Allow credentials.")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="Allowed origins.")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="Allowed methods.")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="Allowed headers.")
    parser.add_argument("--api-key",
                        type=optional_type(str),
                        default=None,
                        help="If provided, the server will require this key "
                        "to be presented in the header.")
    parser.add_argument(
        "--lora-modules",
        type=optional_type(str),
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help="LoRA module configurations in either 'name=path' format"
        "or JSON format. "
        "Example (old format): ``'name=path'`` "
        "Example (new format): "
        "``{\"name\": \"name\", \"path\": \"lora_path\", "
        "\"base_model_name\": \"id\"}``")
    parser.add_argument(
        "--prompt-adapters",
        type=optional_type(str),
        default=None,
        nargs='+',
        action=PromptAdapterParserAction,
        help="Prompt adapter configurations in the format name=path. "
        "Multiple adapters can be specified.")
    parser.add_argument("--chat-template",
                        type=optional_type(str),
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model.")
    parser.add_argument(
        '--chat-template-content-format',
        type=str,
        default="auto",
        choices=get_args(ChatTemplateContentFormatOption),
        help='The format to render message content within a chat template.'
        '\n\n'
        '* "string" will render the content as a string. '
        'Example: ``"Hello World"``\n'
        '* "openai" will render the content as a list of dictionaries, '
        'similar to OpenAI schema. '
        'Example: ``[{"type": "text", "text": "Hello world!"}]``')
    parser.add_argument("--response-role",
                        type=optional_type(str),
                        default="assistant",
                        help="The role name to return if "
                        "``request.add_generation_prompt=true``.")
    parser.add_argument("--ssl-keyfile",
                        type=optional_type(str),
                        default=None,
                        help="The file path to the SSL key file.")
    parser.add_argument("--ssl-certfile",
                        type=optional_type(str),
                        default=None,
                        help="The file path to the SSL cert file.")
    parser.add_argument("--ssl-ca-certs",
                        type=optional_type(str),
                        default=None,
                        help="The CA certificates file.")
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)."
    )
    parser.add_argument(
        "--root-path",
        type=optional_type(str),
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy."
    )
    parser.add_argument(
        "--middleware",
        type=optional_type(str),
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server "
        "using ``@app.middleware('http')``. "
        "If a class is provided, vLLM will add it to the server "
        "using ``app.add_middleware()``. ")
    parser.add_argument(
        "--return-tokens-as-token-ids",
        action="store_true",
        help="When ``--max-logprobs`` is specified, represents single tokens "
        " as strings of the form 'token_id:{token_id}' so that tokens "
        "that are not JSON-encodable can be identified.")
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="If specified, will run the OpenAI frontend server in the same "
        "process as the model serving engine.")
    parser.add_argument(
        "--enable-request-id-headers",
        action="store_true",
        help="If specified, API server will add X-Request-Id header to "
        "responses. Caution: this hurts performance at high QPS.")
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=False,
        help="Enable auto tool choice for supported models. Use "
        "``--tool-call-parser`` to specify which parser to use.")

    valid_tool_parsers = ToolParserManager.tool_parsers.keys()
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        metavar="{" + ",".join(valid_tool_parsers) + "} or name registered in "
        "--tool-parser-plugin",
        default=None,
        help=
        "Select the tool call parser depending on the model that you're using."
        " This is used to parse the model-generated tool call into OpenAI API "
        "format. Required for ``--enable-auto-tool-choice``.")

    parser.add_argument(
        "--tool-parser-plugin",
        type=str,
        default="",
        help=
        "Special the tool parser plugin write to parse the model-generated tool"
        " into OpenAI API format, the name register in this plugin can be used "
        "in ``--tool-call-parser``.")

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        ' The default of None means unlimited.')

    parser.add_argument(
        "--disable-fastapi-docs",
        action='store_true',
        default=False,
        help="Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint."
    )
    parser.add_argument(
        "--enable-prompt-tokens-details",
        action='store_true',
        default=False,
        help="If set to True, enable prompt_tokens_details in usage.")
    parser.add_argument(
        "--enable-server-load-tracking",
        action='store_true',
        default=False,
        help=
        "If set to True, enable tracking server_load_metrics in the app state."
    )

    return parser


def validate_parsed_serve_args(args: argparse.Namespace):
    """Quick checks for model serve args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "serve":
        return

    # Ensure that the chat template is valid; raises if it likely isn't
    validate_chat_template(args.chat_template)

    # Enable auto tool needs a tool call parser to be valid
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        raise TypeError("Error: --enable-auto-tool-choice requires "
                        "--tool-call-parser")


def log_non_default_args(args: argparse.Namespace):
    non_default_args = {}
    parser = make_arg_parser(FlexibleArgumentParser())
    for arg, default in vars(parser.parse_args([])).items():
        if default != getattr(args, arg):
            non_default_args[arg] = getattr(args, arg)
    logger.info("non-default args: %s", non_default_args)


def create_parser_for_docs() -> FlexibleArgumentParser:
    parser_for_docs = FlexibleArgumentParser(
        prog="-m vllm.entrypoints.openai.api_server")
    return make_arg_parser(parser_for_docs)
