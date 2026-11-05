# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configs of the test-only speculators. Configs are imported in every process,
including the scheduler's, so the tree speculator (which imports the model
runner) lives in `speculators` and is named by its qualified name."""

from dataclasses import replace

from vllm.v1.spec_decode.resources import BudgetView, SpecResources
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramGPUSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

_SPECULATORS = f"{__package__}.speculators"


class _BudgetOne(SpeculatorConfig):
    """Every step's drafts may take 1 row: no drafts, plain decode."""

    def candidate_budget(self, view: BudgetView) -> int:
        return 1


class _AlternatingBudget(SpeculatorConfig):
    """Alternates 1 and K + 1 rows: the step after a budget-1 step drafts again."""

    def candidate_budget(self, view: BudgetView) -> int:
        self._steps = getattr(self, "_steps", 0) + 1
        return 1 if self._steps % 2 else self.num_speculative_tokens + 1


class _BudgetOneOnRankZero(SpeculatorConfig):
    """Data parallel rank 0 never drafts while the other ranks do."""

    def candidate_budget(self, view: BudgetView) -> int:
        if self.spec_config.target_parallel_config.data_parallel_index == 0:
            return 1
        return self.num_speculative_tokens + 1


class NgramBudgetOneConfig(_BudgetOne, NgramGPUSpeculatorConfig):
    pass


class Eagle3BudgetOneConfig(_BudgetOne, Eagle3SpeculatorConfig):
    pass


class MTPAlternatingBudgetConfig(_AlternatingBudget, MTPSpeculatorConfig):
    pass


class NgramBudgetOneOnRankZeroConfig(_BudgetOneOnRankZero, NgramGPUSpeculatorConfig):
    pass


class NgramTreeConfig(NgramGPUSpeculatorConfig):
    """A 2-way branching speculator built on the n-gram drafts."""

    speculator_class = f"{_SPECULATORS}.NgramTreeSpeculator"

    def resources(self, max_num_seqs: int) -> SpecResources:
        return replace(super().resources(max_num_seqs), max_branching=2)
