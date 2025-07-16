# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
from collections.abc import Sequence
from dataclasses import field
from typing import Literal, Optional, Union

from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import config
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


@config
@dataclass
class FrontendArgs:
    """Arguments for the OpenAI-compatible frontend server."""
    host: Optional[str] = None
    """Host name."""
    port: int = 8000
    """Port number."""
    uvicorn_log_level: Literal["debug", "info", "warning", "error", "critical",
                               "trace"] = "info"
    """Log level for uvicorn."""
    disable_uvicorn_access_log: bool = False
    """Disable uvicorn access log."""
    allow_credentials: bool = False
    """Allow credentials."""
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    """Allowed origins."""
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    """Allowed methods."""
    allowed_headers: list[str] = field(default_factory=lambda: ["*"])
    """Allowed headers."""
    api_key: Optional[str] = None
    """If provided, the server will require this key to be presented in the
    header."""
    lora_modules: Optional[list[LoRAModulePath]] = None
    """LoRA modules configurations in either 'name=path' format or JSON format
    or JSON list format. Example (old format): `'name=path'` Example (new 
    format): `{\"name\": \"name\", \"path\": \"lora_path\", 
    \"base_model_name\": \"id\"}`"""
    prompt_adapters: Optional[list[PromptAdapterPath]] = None
    """Prompt adapter configurations in the format name=path. Multiple adapters 
    can be specified."""
    chat_template: Optional[str] = None
    """The file path to the chat template, or the template in single-line form 
    for the specified model."""
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    """The format to render message content within a chat template.

* "string" will render the content as a string. Example: `"Hello World"`
* "openai" will render the content as a list of dictionaries, similar to OpenAI 
schema. Example: `[{"type": "text", "text": "Hello world!"}]`"""
    response_role: str = "assistant"
    """The role name to return if `request.add_generation_prompt=true`."""
    ssl_keyfile: Optional[str] = None
    """The file path to the SSL key file."""
    ssl_certfile: Optional[str] = None
    """The file path to the SSL cert file."""
    ssl_ca_certs: Optional[str] = None
    """The CA certificates file."""
    enable_ssl_refresh: bool = False
    """Refresh SSL Context when SSL certificate files change"""
    ssl_cert_reqs: int = int(ssl.CERT_NONE)
    """Whether client certificate is required (see stdlib ssl module's)."""
    root_path: Optional[str] = None
    """FastAPI root_path when app is behind a path based routing proxy."""
    middleware: list[str] = field(default_factory=lambda: [])
    """Additional ASGI middleware to apply to the app. We accept multiple 
    --middleware arguments. The value should be an import path. If a function 
    is provided, vLLM will add it to the server using 
    `@app.middleware('http')`. If a class is provided, vLLM will 
    add it to the server using `app.add_middleware()`."""
    return_tokens_as_token_ids: bool = False
    """When `--max-logprobs` is specified, represents single tokens as 
    strings of the form 'token_id:{token_id}' so that tokens that are not 
    JSON-encodable can be identified."""
    disable_frontend_multiprocessing: bool = False
    """If specified, will run the OpenAI frontend server in the same process as 
    the model serving engine."""
    enable_request_id_headers: bool = False
    """If specified, API server will add X-Request-Id header to responses. 
    Caution: this hurts performance at high QPS."""
    enable_auto_tool_choice: bool = False
    """Enable auto tool choice for supported models. Use `--tool-call-parser` 
    to specify which parser to use."""
    tool_call_parser: Optional[str] = None
    """Select the tool call parser depending on the model that you're using. 
    This is used to parse the model-generated tool call into OpenAI API format. 
    Required for `--enable-auto-tool-choice`. You can choose any option from 
    the built-in parsers or register a plugin via `--tool-parser-plugin`."""
    tool_parser_plugin: str = ""
    """Special the tool parser plugin write to parse the model-generated tool 
    into OpenAI API format, the name register in this plugin can be used in 
    `--tool-call-parser`."""
    log_config_file: Optional[str] = envs.VLLM_LOGGING_CONFIG_PATH
    """Path to logging config JSON file for both vllm and uvicorn"""
    max_log_len: Optional[int] = None
    """Max number of prompt characters or prompt ID numbers being printed in 
    log. The default of None means unlimited."""
    disable_fastapi_docs: bool = False
    """Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint."""
    enable_prompt_tokens_details: bool = False
    """If set to True, enable prompt_tokens_details in usage."""
    enable_server_load_tracking: bool = False
    """If set to True, enable tracking server_load_metrics in the app state."""
    enable_force_include_usage: bool = False
    """If set to True, including usage on every request."""

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        from vllm.engine.arg_utils import get_kwargs

        frontend_kwargs = get_kwargs(FrontendArgs)

        # Special case: allowed_origins, allowed_methods, allowed_headers all
        # need json.loads type
        # Should also remove nargs
        frontend_kwargs["allowed_origins"]["type"] = json.loads
        frontend_kwargs["allowed_methods"]["type"] = json.loads
        frontend_kwargs["allowed_headers"]["type"] = json.loads
        del frontend_kwargs["allowed_origins"]["nargs"]
        del frontend_kwargs["allowed_methods"]["nargs"]
        del frontend_kwargs["allowed_headers"]["nargs"]

        # Special case: LoRA modules need custom parser action and
        # optional_type(str)
        frontend_kwargs["lora_modules"]["type"] = optional_type(str)
        frontend_kwargs["lora_modules"]["action"] = LoRAParserAction

        # Special case: Prompt adapters need custom parser action and
        # optional_type(str)
        frontend_kwargs["prompt_adapters"]["type"] = optional_type(str)
        frontend_kwargs["prompt_adapters"][
            "action"] = PromptAdapterParserAction

        # Special case: Middleware needs append action
        frontend_kwargs["middleware"]["action"] = "append"

        # Special case: Tool call parser shows built-in options.
        valid_tool_parsers = list(ToolParserManager.tool_parsers.keys())
        frontend_kwargs["tool_call_parser"]["choices"] = valid_tool_parsers

        frontend_group = parser.add_argument_group(
            title="Frontend",
            description=FrontendArgs.__doc__,
        )

        for key, value in frontend_kwargs.items():
            frontend_group.add_argument(f"--{key.replace('_', '-')}", **value)

        return parser


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    """Create the CLI argument parser used by the OpenAI API server.

    We rely on the helper methods of `FrontendArgs` and `AsyncEngineArgs` to
    register all arguments instead of manually enumerating them here. This
    avoids code duplication and keeps the argument definitions in one place.
    """
    parser.add_argument("model_tag",
                        type=str,
                        nargs="?",
                        help="The model tag to serve "
                        "(optional if specified in config)")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.")
    parser.add_argument(
        "--data-parallel-start-rank",
        "-dpr",
        type=int,
        default=0,
        help="Starting data parallel rank for secondary nodes. "
        "Requires --headless.")
    parser.add_argument("--api-server-count",
                        "-asc",
                        type=int,
                        default=1,
                        help="How many API server processes to run.")
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "Must be a YAML with the following options: "
        "https://docs.vllm.ai/en/latest/configuration/serve_args.html")
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

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
    if args.enable_prompt_embeds and args.enable_prompt_adapter:
        raise ValueError(
            "Cannot use prompt embeds and prompt adapter at the same time.")


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
