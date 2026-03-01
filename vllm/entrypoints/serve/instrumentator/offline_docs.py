# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline FastAPI documentation support for air-gapped environments."""

import pathlib

from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

from vllm.logger import init_logger

logger = init_logger(__name__)


def attach_router(app: FastAPI) -> None:
    """Attach offline docs router if enabled via args."""
    args = getattr(app.state, "args", None)
    if args is None or not getattr(args, "enable_offline_docs", False):
        return

    static_dir = pathlib.Path(__file__).parent / "static"

    if not static_dir.exists():
        logger.warning(
            "Static directory not found at %s. Offline docs will not be available.",
            static_dir,
        )
        return

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    logger.info("Offline documentation enabled with vendored static assets")
