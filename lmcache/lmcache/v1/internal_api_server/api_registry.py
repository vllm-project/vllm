# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import List, Literal, Optional

# Third Party
from fastapi import APIRouter

# First Party
from lmcache.v1.utils.router_discovery import discover_api_routers

APICategory = Literal["common", "vllm", "controller"]


class APIRegistry:
    """
    Automatically discovers and registers API routes by category

    Categories:
    - common: APIs that work for all components (metrics, logs, etc.)
    - vllm: APIs specific to vLLM scheduler/worker
    - controller: APIs specific to LMCache controller
    """

    def __init__(self, app):
        self.app = app
        self.router = APIRouter()

    def register_all_apis(self, categories: Optional[List[APICategory]] = None):
        """
        Discover and register API modules from specified categories

        Args:
            categories: List of categories to register.
                       If None, registers all categories.
        """
        if categories is None:
            categories = ["common", "vllm", "controller"]

        package_path = Path(__file__).parent
        package_name = __package__

        for category in categories:
            category_path = package_path / category
            if not category_path.exists():
                continue

            category_package = f"{package_name}.{category}"
            for r in discover_api_routers(category_path, category_package):
                self.router.include_router(r)

        self.app.include_router(self.router)
