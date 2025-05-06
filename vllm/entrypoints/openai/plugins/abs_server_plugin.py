# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

    from fastapi import FastAPI
    from fastapi.datastructures import State

    from vllm.config import VllmConfig
    from vllm.engine.protocol import EngineClient
    from vllm.utils import FlexibleArgumentParser


class ServerPlugin:
    """
    Base class for server plugins. All server plugins should inherit from this
    class and implement the abstract methods `make_arg_parser` and
    `apply_plugin`.
    """

    @classmethod
    def make_arg_parser(
            cls, parser: "FlexibleArgumentParser") -> "FlexibleArgumentParser":
        """
        Add plugin-specific arguments to the argument parser.
        """
        return parser

    @classmethod
    def apply_plugin(
        cls,
        app: "FastAPI",
        engine_client: "EngineClient",
        vllm_config: "VllmConfig",
        state: "State",
        args: "Namespace",
    ):
        """
        Apply the plugin to the server application.
        """
        pass
