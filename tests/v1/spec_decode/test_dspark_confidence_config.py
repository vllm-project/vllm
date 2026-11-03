# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for where DSpark checkpoints declare their confidence head."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.qwen3_dspark import resolve_confidence_head_config


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            SimpleNamespace(
                enable_confidence_head=True, confidence_head_with_markov=True
            ),
            (True, True),
            id="speculators-top-level",
        ),
        pytest.param(
            SimpleNamespace(
                dflash_config={
                    "use_confidence_head": True,
                    "confidence_head_with_markov": False,
                }
            ),
            (True, False),
            id="nvidia-nested-use",
        ),
        pytest.param(
            SimpleNamespace(
                dflash_config={
                    "enable_confidence_head": True,
                    "confidence_head_with_markov": True,
                }
            ),
            (True, True),
            id="nested-enable",
        ),
        pytest.param(
            SimpleNamespace(
                enable_confidence_head=False,
                dflash_config={"use_confidence_head": True},
            ),
            (False, False),
            id="top-level-wins",
        ),
        pytest.param(SimpleNamespace(), (False, False), id="absent"),
    ],
)
def test_resolve_confidence_head_config(config, expected):
    assert resolve_confidence_head_config(config) == expected
