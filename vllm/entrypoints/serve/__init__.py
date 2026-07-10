# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

from vllm.logger import init_logger

logger = init_logger(__name__)


def register_vllm_serve_api_routers(app: FastAPI):
    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)


def register_vllm_dev_api_routers(app: FastAPI):
    logger.warning(
        "SECURITY WARNING: Development endpoints are enabled! "
        "This should NOT be used in production!"
    )

    from .dev.cache.api_router import attach_router as attach_cache_router

    attach_cache_router(app)

    from .dev.rlhf.api_router import attach_router as attach_rlhf_router

    attach_rlhf_router(app)

    from .dev.rpc.api_router import attach_router as attach_rpc_router

    attach_rpc_router(app)

    from .dev.server_info.api_router import (
        attach_router as attach_server_info_router,
    )

    attach_server_info_router(app)

    from .dev.sleep.api_router import attach_router as attach_sleep_router

    attach_sleep_router(app)
