"""
vLLM's OpenAI-compatible API server with simulation hooks installed.

Usage:
    python -m vllm_simulator.simulation.vllm.launch_server --model <model_path> [vLLM args...]

Environment variables:
    VLLM_SIMULATOR_CONFIG_PATH: Path to simulation config JSON
    VLLM_SIMULATOR_OUTPUT_MODE: "BLOCKING" or "NON_BLOCKING" (default)
"""

import os

from vllm_simulator.simulation.manager import Envs
from vllm_simulator.simulation.vllm.startup import init_hook
from vllm_simulator.utils import get_logger

init_hook()

logger = get_logger()


if __name__ == "__main__":
    import uvloop
    from vllm.entrypoints.launchers.api_server.entry import run_server
    from vllm.entrypoints.launchers.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="vLLM Simulation Server (OpenAI-compatible)"
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--sim-config-path",
        type=str,
        default=None,
        help="Path to simulation JSON config (same as VLLM_SIMULATOR_CONFIG_PATH).",
    )

    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # Set config path from CLI arg if env var not set
    config_path = os.getenv("VLLM_SIMULATOR_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from env: {config_path}")
    elif args.sim_config_path:
        os.environ["VLLM_SIMULATOR_CONFIG_PATH"] = args.sim_config_path
        logger.info(f"Using config from arg: {args.sim_config_path}")

    logger.info(
        "Starting vLLM simulation server in %s mode.", Envs.simulation_mode()
    )

    uvloop.run(run_server(args))
