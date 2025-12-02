# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI


def register_pooling_api_routers(app: FastAPI):
    from vllm.entrypoints.pooling.classify.api_router import router as classify_router
    from vllm.entrypoints.pooling.embed.api_router import router as embed_router
    from vllm.entrypoints.pooling.pooling.api_router import router as pooling_router
    from vllm.entrypoints.pooling.score.api_router import router as score_router

    app.include_router(classify_router)
    app.include_router(embed_router)
    app.include_router(score_router)
    app.include_router(pooling_router)
