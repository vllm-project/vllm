# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test speculators and the CUSTOM tree attention backend, as a general plugin."""

# Method name -> config class in `configs`, imported lazily by qualified name.
TEST_SPECULATORS = {
    "test_ngram_budget_one": "NgramBudgetOneConfig",
    "test_eagle3_budget_one": "Eagle3BudgetOneConfig",
    "test_mtp_alternating_budget": "MTPAlternatingBudgetConfig",
    "test_ngram_budget_one_on_rank_zero": "NgramBudgetOneOnRankZeroConfig",
    "test_ngram_tree": "NgramTreeConfig",
}
# Printed once per worker when the tree speculator accepts a path that leaves
# the chain (skips a sibling).
OFF_CHAIN = "test_ngram_tree: accepted a path off the chain"


def register():
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )

    for name, config_cls in TEST_SPECULATORS.items():
        register_speculative_method(name, f"{__name__}.configs.{config_cls}")
    register_backend(
        AttentionBackendEnum.CUSTOM, f"{__name__}.tree_attn.TreeAttentionBackend"
    )
