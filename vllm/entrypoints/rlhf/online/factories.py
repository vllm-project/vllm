# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import FastAPI


def register_rlhf_api_routers(app: FastAPI):
    from .api_router import router as rlhf_router

    app.include_router(rlhf_router)
