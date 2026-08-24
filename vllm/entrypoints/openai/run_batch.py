# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import warnings

from prometheus_client import start_http_server

from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

warnings.warn(
    "The `python -m vllm.entrypoints.openai.run_batch` command is deprecated "
    "and may be removed in a future release. Please use `vllm run_batch` instead.",
    DeprecationWarning,
    stacklevel=1,
)


if __name__ == "__main__":
    from vllm.entrypoints.launchers.run_batch import main, parse_args

    logger = init_logger(__name__)

    args = parse_args()

    logger.info("vLLM batch processing API version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    # Start the Prometheus metrics server. LLMEngine uses the Prometheus client
    # to publish metrics at the /metrics endpoint.
    if args.enable_metrics:
        logger.info("Prometheus metrics enabled")
        start_http_server(port=args.port, addr=args.host)
    else:
        logger.info("Prometheus metrics disabled")

    asyncio.run(main(args))
