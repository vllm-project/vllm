# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
from fastapi import APIRouter, FastAPI

# First Party
from lmcache.logging import init_logger
from lmcache.v1.utils.router_discovery import discover_api_routers

logger = init_logger(__name__)


class HTTPAPIRegistry:
    """
    Automatically discovers and registers HTTP API routes
    from the ``http_apis`` sub-package.

    Any module whose name ends with ``_api`` and exposes a
    module-level ``router`` (:class:`~fastapi.APIRouter`) will
    be picked up automatically.
    """

    def __init__(self, app: FastAPI):
        self.app = app
        self.router = APIRouter()

    def register_all_apis(self) -> None:
        """
        Discover and register all ``*_api`` modules under
        the ``http_apis`` directory.
        """
        apis_path = Path(__file__).parent / "http_apis"
        if not apis_path.exists():
            logger.warning("http_apis directory not found")
            return

        apis_package = f"{__package__}.http_apis"

        for r in discover_api_routers(apis_path, apis_package):
            self.router.include_router(r)

        self.app.include_router(self.router)
