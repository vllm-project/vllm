# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import FastAPI


def register_rl_api_routers(app: FastAPI):
    from .online.api_router import router as rl_router
    from .online.metrics import weight_operation_metrics

    weight_operation_metrics()
    app.include_router(rl_router)
