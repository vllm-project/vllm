# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from fastapi import FastAPI

from vllm import envs
from vllm.logger import init_logger

logger = init_logger("vllm.entrypoints.serve")


def register_vllm_serve_api_routers(app: FastAPI, args: Namespace):
    if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        logger.warning(
            "LoRA dynamic loading & unloading is enabled in the API server. "
            "This should ONLY be used for local development!"
        )
        from vllm.entrypoints.serve.lora.api_router import router as lora_router

        app.include_router(lora_router)
    from vllm.entrypoints.serve.elastic_ep.api_router import (
        router as elastic_ep_router,
    )

    app.include_router(elastic_ep_router)

    if envs.VLLM_TORCH_PROFILER_DIR:
        logger.warning_once(
            "Torch Profiler is enabled in the API server. This should ONLY be "
            "used for local development!"
        )
    elif envs.VLLM_TORCH_CUDA_PROFILE:
        logger.warning_once(
            "CUDA Profiler is enabled in the API server. This should ONLY be "
            "used for local development!"
        )
    if envs.VLLM_TORCH_PROFILER_DIR or envs.VLLM_TORCH_CUDA_PROFILE:
        from vllm.entrypoints.serve.profile.api_router import (
            router as profile_router,
        )

        app.include_router(profile_router)
    if envs.VLLM_SERVER_DEV_MODE:
        logger.warning(
            "SECURITY WARNING: Development endpoints are enabled! "
            "This should NOT be used in production!"
        )
        from vllm.entrypoints.serve.sleep.api_router import router as sleep_router

        app.include_router(sleep_router)

    from vllm.entrypoints.serve.tokenize.api_router import router as tokenize_router
    from vllm.entrypoints.serve.tokenize.api_router import tokenizer_info_router

    app.include_router(tokenize_router)
    if getattr(args, "enable_tokenizer_info_endpoint", False):
        """Conditionally register the tokenizer info endpoint if enabled."""
        app.include_router(tokenizer_info_router)

    from vllm.entrypoints.serve.disagg.api_router import router as disagg_router

    app.include_router(disagg_router)
    # Optional endpoints
    if getattr(args, "tokens_only", False):
        """Conditionally register the disagg abort endpoint if enabled."""
        from vllm.entrypoints.serve.disagg.api_router import abort_router

        app.include_router(abort_router)
    from vllm.entrypoints.serve.instrumentator.health import (
        router as health_router,
    )

    app.include_router(health_router)

    from vllm.entrypoints.serve.instrumentator.metrics import mount_metrics

    mount_metrics(app)
