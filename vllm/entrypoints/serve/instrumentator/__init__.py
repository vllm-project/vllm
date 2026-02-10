# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

from vllm import envs


def register_instrumentator_api_routers(app: FastAPI):
    from .basic import router as basic_router

    app.include_router(basic_router)

    from .health import router as health_router

    app.include_router(health_router)

    from .metrics import attach_router as metrics_attach_router

    metrics_attach_router(app)

    from .offline_docs import attach_router as offline_docs_attach_router

    offline_docs_attach_router(app)

    if envs.VLLM_SERVER_DEV_MODE:
        from .server_info import router as server_info_router

        app.include_router(server_info_router)
