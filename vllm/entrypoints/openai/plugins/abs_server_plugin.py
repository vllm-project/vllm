# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace

from fastapi import FastAPI
from fastapi.datastructures import State

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.reasoning_parsers import (ReasoningParser,
                                                       ReasoningParserManager)
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.utils import FlexibleArgumentParser


class ServerPlugin:
    """
    Base class for server plugins. All server plugins should inherit from this
    class and implement the abstract methods `make_arg_parser` and
    `apply_plugin`.
    """

    _additional_tool_parsers: dict[str, ToolParser] = {}
    _additional_reasoning_parsers: dict[str, ReasoningParser] = {}

    @classmethod
    def make_arg_parser(
            cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """
        Add plugin-specific arguments to the argument parser.
        """
        return parser

    @classmethod
    def register_parser(cls):
        """
        Register parsers such as ToolParsers and ReasoningParsers into
        corresponding registries.
        """
        for name, parser in cls._additional_tool_parsers.items():
            ToolParserManager.register_module(name=name, module=parser)
        for name, parser in cls._additional_reasoning_parsers.items():
            ReasoningParserManager.register_module(name=name, module=parser)

    @classmethod
    def apply_plugin(
        cls,
        app: FastAPI,
        engine_client: EngineClient,
        model_config: ModelConfig,
        state: State,
        args: Namespace,
    ):
        """
        Apply the plugin to the server application.
        """
        pass
