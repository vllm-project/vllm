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
    "test_ngram_tree_cycle": "NgramTreeCycleConfig",
}
# Printed once per worker when the tree speculator accepts a path that leaves
# the chain (skips a sibling).
OFF_CHAIN = "test_ngram_tree: accepted a path off the chain"
# Printed once per worker and topology the cycling tree speculator plans.
PLANNED = "test_ngram_tree_cycle: planned"
# Printed by the CUSTOM backend with the cycling speculator's topology, once
# per step where every request verifies all its rows (FULL CUDA graphs replay
# those): when it builds the step's metadata, and when it runs the step's
# attention in Python. Built but not run: the step replayed a CUDA graph.
UNIFORM_VERIFY = "tree attention: built a uniform verify batch:"
EAGER_UNIFORM_VERIFY = "tree attention: ran a uniform verify batch eagerly:"


class VerifySteps:
    """The topology the cycling tree speculator planned for this step; None
    before its first plan (the model runner's warmup and capture runs)."""

    topology: str | None = None


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
