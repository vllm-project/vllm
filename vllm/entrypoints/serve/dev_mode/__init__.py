# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

from vllm.logger import init_logger

logger = init_logger(__name__)


def register_vllm_dev_mode_api_routers(app: FastAPI):
    logger.warning(
        "SECURITY WARNING: Development endpoints are enabled! "
        "This should NOT be used in production!"
    )

    from .cache.api_router import attach_router as attach_cache_router

    attach_cache_router(app)

    from .rlhf.api_router import attach_router as attach_rlhf_router

    attach_rlhf_router(app)

    from .rpc.api_router import attach_router as attach_rpc_router

    attach_rpc_router(app)

    from .server_info.api_router import attach_router as attach_server_info_router

    attach_server_info_router(app)

    from .sleep.api_router import attach_router as attach_sleep_router

    attach_sleep_router(app)
