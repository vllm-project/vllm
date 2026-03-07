# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only mode: populate the torch.compile / Inductor cache
without loading real model weights or allocating KV caches.

Can be invoked in two ways:

1. From CLI args (``vllm compile <model>``):
   ``run_compile_only(args)`` builds an ``EngineArgs`` from the parsed
   namespace, sets ``compile_only=True``, then creates an ``EngineCore``
   which triggers compilation.

2. From a serialized ``VllmConfig`` (overlap subprocess):
   ``python -m vllm.compile_only --config-path <path>`` loads a
   pickled ``VllmConfig``, patches it for compile-only mode, and
   runs compilation directly.

The compile-only flag causes the model loader to be wrapped with
FakeTensorMode (see ``fake_loader.wrap_loader_with_fake``), so the
user's original ``load_format`` is preserved and the real loader's
full pipeline runs — just with fake tensors instead of real weights.
"""

import argparse
import pickle

from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


def run_compile_only(args: argparse.Namespace) -> None:
    """Run compile-only mode from CLI arguments."""
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.enforce_eager = False

    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.LLM_CLASS)
    vllm_config.compilation_config.compile_only = True

    _run_compile_with_config(vllm_config)


def run_compile_only_from_config(config_path: str) -> None:
    """Run compile-only mode from a serialized VllmConfig."""
    with open(config_path, "rb") as f:
        vllm_config = pickle.load(f)

    # Patch for compile-only mode
    vllm_config.compilation_config.compile_only = True
    vllm_config.compilation_config.overlap_compile = False

    _run_compile_with_config(vllm_config)


def _run_compile_with_config(vllm_config) -> None:
    """Shared compile-only logic."""
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor import Executor

    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
    )

    logger.info("Compile-only mode complete. Cache populated.")
    engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description="vLLM compile-only mode (internal)")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to pickled VllmConfig",
    )
    args = parser.parse_args()
    run_compile_only_from_config(args.config_path)


if __name__ == "__main__":
    main()
