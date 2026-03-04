#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM AFD FFN Server Entry Point

This script provides a standalone entry point for running FFN servers in an AFD
(Attention-FFN Disaggregation) setup. FFN servers handle remote FFN computation
for attention workers.

Usage:
    python -m vllm.entrypoints.afd_ffn_server /path/to/model \
        --tensor-parallel-size 8 \
        --afd-config '{"afd_connector": "dummy", "afd_role": "ffn"}' \
"""

import threading
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


class AFDFFNServer:
    """AFD FFN Server main class."""

    def __init__(self, args: Any):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.vllm_config = engine_args.create_engine_config()
        logger.info("Start AFD FFN Server with vllm_config: %s", self.vllm_config)

    def start(self) -> None:
        """Start the AFD FFN server."""
        try:
            # Import here to avoid circular imports
            from vllm.v1.executor.abstract import Executor

            # Create configurations
            executor_class = Executor.get_class(self.vllm_config)
            self.model_executor = executor_class(vllm_config=self.vllm_config)
            # Start the FFN server loop
            self._run_server_loop()

        except Exception as e:
            logger.error("Failed to start AFD FFN server: %s", e)
            raise

    def _run_server_loop(self) -> None:
        """Start FFN workers and wait for completion"""
        logger.info("AFD FFN Server started, workers running...")
        try:
            # Tell workers to start FFN server loops (one-time call)
            self.model_executor.collective_rpc("start_ffn_server_loop")

            # Main thread waits without busy polling
            shutdown_event = threading.Event()
            shutdown_event.wait()  # Block until interrupted

        except KeyboardInterrupt:
            logger.info("Server shutting down...")
            self.model_executor.collective_rpc("stop_ffn_server_loop")
        except Exception as e:
            logger.error("Server error: %s", e)
            raise


def main(args: Any) -> None:
    """Main entry point for AFD FFN server."""
    try:
        server = AFDFFNServer(args)
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Server error: %s", e)
        raise


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add model as positional argument (like vllm serve)
    parser.add_argument("model", type=str, help="Model name or path")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Set the model from positional argument
    args.model = args.model

    main(args)
