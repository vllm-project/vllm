# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import FastAPI


def register_rl_api_routers(app: FastAPI):
    from .online.api_router import router as rl_router

    app.include_router(rl_router)
