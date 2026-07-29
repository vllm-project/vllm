# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for https://github.com/vllm-project/vllm/issues/49614.

DeepSeek ships the same ``Qwen3DSparkModel`` architecture for both its DSpark
(``markov_rank > 0``) and DFlash (``markov_rank == 0``) Qwen3 checkpoints.
``method="dflash"`` on such a checkpoint must be routed to the DSpark path
(``Qwen3DSparkForCausalLM`` serves both) rather than EAGLE-renamed to the
unregistered ``DFlashQwen3DSparkModel``, which fails to load.
"""

import pytest

from vllm.config.speculative import SpeculativeConfig

route = SpeculativeConfig._route_self_contained_qwen3_dspark


@pytest.mark.cpu_test
def test_dflash_on_qwen3dspark_arch_routes_to_dspark():
    assert (
        route("dflash", ["Qwen3DSparkModel"], "deepseek-ai/dflash_qwen3_4b_block7")
        == "dspark"
    )


@pytest.mark.cpu_test
def test_explicit_dspark_is_unchanged():
    assert (
        route("dspark", ["Qwen3DSparkModel"], "deepseek-ai/dspark_qwen3_4b_block7")
        == "dspark"
    )


@pytest.mark.cpu_test
def test_standalone_dflash_arch_is_unchanged():
    # A standalone DFlash checkpoint (DFlashDraftModel arch) keeps method="dflash"
    # and its normal EAGLE-style arch handling.
    assert route("dflash", ["DFlashDraftModel"], "z-lab/Qwen3.5-9B-DFlash") == "dflash"


@pytest.mark.cpu_test
def test_other_methods_are_unchanged():
    assert route("eagle3", ["Qwen3DSparkModel"], "some/model") == "eagle3"
