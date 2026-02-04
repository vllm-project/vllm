# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import MagicMock

import regex as re
from pydantic_core import core_schema

os.environ["VLLM_LOGGING_LEVEL"] = "error"
logger = logging.getLogger("fetch_api_files")

ROOT_DIR = Path(__file__).parent.parent

sys.path.insert(0, str(ROOT_DIR))


def mock_if_no_torch(mock_module: str, mock: MagicMock):
    if not importlib.util.find_spec("torch"):
        sys.modules[mock_module] = mock


# Mock custom op code
class MockCustomOp:
    @staticmethod
    def register(name):
        def decorator(cls):
            return cls

        return decorator


mock_if_no_torch("vllm._C", MagicMock())
mock_if_no_torch("vllm.model_executor.custom_op", MagicMock(CustomOp=MockCustomOp))
mock_if_no_torch(
    "vllm.utils.torch_utils", MagicMock(direct_register_custom_op=lambda *a, **k: None)
)
mock_if_no_torch("vllm.platforms", MagicMock(current_platform=MagicMock()))

# Mock any version checks by reading from compiled CI requirements
with open(ROOT_DIR / "requirements/test.txt") as f:
    VERSIONS = dict(line.strip().split("==") for line in f if "==" in line)
importlib.metadata.version = lambda name: VERSIONS.get(name) or "0.0.0"


# Make torch.nn.Parameter safe to inherit from
mock_if_no_torch("torch.nn", MagicMock(Parameter=object))
mock_if_no_torch("tokenizers", MagicMock(__version__="0.0.0"))


class PydanticMagicMock(MagicMock):
    """`MagicMock` that's able to generate pydantic-core schemas."""

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.__spec__ = ModuleSpec(name, None)

    def __get_pydantic_core_schema__(self, source_type, handler):
        return core_schema.any_schema()


def auto_mock(module_name: str, attr: str, max_mocks: int = 100):
    """Function that automatically mocks missing modules during imports."""
    logger.info("Importing %s from %s", attr, module_name)

    for i in range(max_mocks):
        try:
            module = importlib.import_module(module_name)

            # First treat attr as an attr, then as a submodule
            if hasattr(module, attr):
                return getattr(module, attr)

            return importlib.import_module(f"{module_name}.{attr}")
        except ModuleNotFoundError as e:
            assert e.name is not None
            logger.info("Mocking %s for argparse doc generation: %d: %s", e.name, i, e)
            sys.modules[e.name] = PydanticMagicMock(name=e.name)
        except Exception:
            logger.exception("Failed to import %s.%s", module_name, attr)

    raise ImportError(
        f"Failed to import {module_name}.{attr} after mocking {max_mocks} imports"
    )


with open(Path(ROOT_DIR, "docs/api/README.md")) as f:
    files = set()
    for line in f:
        m = re.match(r"- \[(vllm\..+)\]\[\]", line)
        if m is not None:
            full_name = m.group(1)
            for i in range(2):
                module_name, attr = full_name.rsplit(".", 1)
                # if full_name != "vllm.LLM":
                #     continue
                x = auto_mock(module_name, attr, max_mocks=1000)
                try:
                    p = Path(inspect.getfile(x))
                    files.add(p.relative_to(ROOT_DIR).as_posix())
                    break
                except TypeError:
                    # try to fetch from parent module
                    full_name = module_name
print("\n".join(sorted(files)))
