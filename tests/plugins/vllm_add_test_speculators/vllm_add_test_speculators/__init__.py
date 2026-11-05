# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only speculative methods, registered as a vLLM general plugin so that
every process (engine core and workers, forked or spawned) has them."""

# Method name -> config class in `configs`, imported lazily by qualified name.
TEST_SPECULATORS = {
    "test_ngram_budget_one": "NgramBudgetOneConfig",
    "test_eagle3_budget_one": "Eagle3BudgetOneConfig",
    "test_mtp_alternating_budget": "MTPAlternatingBudgetConfig",
    "test_ngram_budget_one_on_rank_zero": "NgramBudgetOneOnRankZeroConfig",
    "test_ngram_tree": "NgramTreeConfig",
}


def register():
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )

    for name, config_cls in TEST_SPECULATORS.items():
        register_speculative_method(name, f"{__name__}.configs.{config_cls}")
