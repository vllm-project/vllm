# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import logging
import sys
from unittest.mock import MagicMock

logger = logging.getLogger("mkdocs")

sys.modules["vllm._C"] = MagicMock()


class PydanticMagicMock(MagicMock):
    """`MagicMock` that's able to generate pydantic-core schemas."""

    def __get_pydantic_core_schema__(self, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.any_schema()


def auto_mock(module, attr, max_tries=50):
    """Function that automatically mocks missing modules during imports."""
    for _ in range(max_tries):
        try:
            return getattr(importlib.import_module(module), attr,
                           importlib.import_module(f"{module}.{attr}"))
        except importlib.metadata.PackageNotFoundError as e:
            raise e
        except ModuleNotFoundError as e:
            logger.info("Mocking %s for argparse doc generation", e.name)
            sys.modules[e.name] = PydanticMagicMock()
    else:
        raise ImportError(
            f"Failed to import {module}.{attr} after {max_tries} attempts")
