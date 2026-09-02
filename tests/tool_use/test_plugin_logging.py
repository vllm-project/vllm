# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A parser loaded from a plugin path must log like the rest of vLLM.

Only the ``vllm`` logger is configured (``vllm/logger.py``), so a plugin
imported under a module name taken from its file name gets a logger with no
handler anywhere up its chain: ``logger.info`` and ``logger.debug`` are dropped
and warnings escape outside vLLM's formatting and level settings. Importing the
plugin under the ``vllm.*`` namespace is what keeps it configured, and plugin
authors need to do nothing.
"""

import logging
import sys

import pytest

from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager


def _write_plugin(tmp_path, name):
    """A plugin that records the module name it was imported under."""
    plugin = tmp_path / f"{name}.py"
    plugin.write_text("IMPORTED_AS = __name__\n")
    return plugin


def _configured_ancestor(name):
    """The logger that would actually handle a record from ``name``."""
    logger = logging.getLogger(name)
    while logger:
        if logger.handlers:
            return logger
        if not logger.propagate:
            return None
        logger = logger.parent
    return None


@pytest.mark.parametrize(
    "manager,importer,prefix",
    [
        (ToolParserManager, "import_tool_parser", "vllm.tool_parsers.plugins."),
        (
            ReasoningParserManager,
            "import_reasoning_parser",
            "vllm.reasoning.plugins.",
        ),
    ],
)
def test_plugin_is_imported_under_the_vllm_logging_namespace(
    tmp_path, manager, importer, prefix
):
    name = "dummy_parser_plugin"
    getattr(manager, importer)(str(_write_plugin(tmp_path, name)))

    module = sys.modules[prefix + name]
    assert module.IMPORTED_AS == prefix + name

    handler_owner = _configured_ancestor(module.IMPORTED_AS)
    assert handler_owner is not None, (
        f"records from {module.IMPORTED_AS} reach no handler"
    )
    assert handler_owner.name == "vllm"
