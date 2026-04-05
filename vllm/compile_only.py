# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only mode: populate vLLM's torch.compile cache
without loading real model weights or allocating KV caches.

The compile-only flag causes the model loader to be wrapped with
FakeTensorMode (see ``fake_loader.wrap_loader_with_fake``), so the
user's original ``load_format`` is preserved and the real loader's
full pipeline runs — just with fake tensors instead of real weights.
"""

import argparse

from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


def compile_model(
    model: str,
    *,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    quantization: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> None:
    """Pre-populate vLLM's torch.compile cache for a model.

    Runs compilation using fake weights (zero GPU memory)
    so that vLLM's torch.compile cache is populated.  Subsequent
    ``vllm serve`` or ``LLM(...)`` calls for the same model
    configuration will hit the warm cache and skip compilation.

    Args:
        model: HuggingFace model name or path.
        tensor_parallel_size: Number of tensor parallel GPUs.
        pipeline_parallel_size: Number of pipeline parallel stages.
        quantization: Quantization method (e.g. "fp8").
        dtype: Model dtype.
        trust_remote_code: Trust remote code from HuggingFace.
        **kwargs: Additional arguments passed to ``EngineArgs``.
    """
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        quantization=quantization,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        enforce_eager=False,
        **kwargs,
    )

    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.LLM_CLASS)
    vllm_config.compilation_config.compile_only = True

    _run_compile_with_config(vllm_config)


def run_compile_only(args: argparse.Namespace) -> None:
    """Run compile-only mode from CLI arguments."""
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.enforce_eager = False

    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.LLM_CLASS)
    vllm_config.compilation_config.compile_only = True

    _run_compile_with_config(vllm_config)


def _run_compile_with_config(vllm_config) -> None:
    """Shared compile-only logic."""
    from vllm.plugins import load_general_plugins
    from vllm.v1.executor import Executor

    load_general_plugins()

    executor_class = Executor.get_class(vllm_config)
    executor = executor_class(vllm_config)
    executor.collective_rpc("compile_or_warm_up_model")

    logger.info("Compile-only mode complete. Cache populated.")
    executor.shutdown()
