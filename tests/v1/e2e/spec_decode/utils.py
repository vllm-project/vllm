# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts, evaluate_gsm8k_offline
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.outputs import RequestOutput
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.metrics.reader import Counter, Gauge, Metric
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramGPUSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.round import BudgetView, SpecResources
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    register_speculative_method,
)


def _skip_if_insufficient_gpus_for_tp(tp_size: int):
    """Skip if fewer than ``tp_size`` accelerators are available."""
    available_gpus = torch.accelerator.device_count()
    if available_gpus < tp_size:
        pytest.skip(
            f"Test requires {tp_size} GPUs, but only {available_gpus} available"
        )


Messages = list[dict[str, Any]]


def get_test_prompts(mm_enabled: bool, num_prompts: int = 100) -> list[Messages]:
    prompt_types = ["repeat", "gsm8k"]
    if mm_enabled:
        prompt_types.append("mm")
    prompts: list[Messages] = []

    num_repeat_prompts = num_prompts // len(prompt_types)
    if mm_enabled:
        num_gsm8k_prompts = num_prompts // len(prompt_types)
        num_mm_prompts = num_prompts - num_repeat_prompts - num_gsm8k_prompts
    else:
        num_mm_prompts = 0
        num_gsm8k_prompts = num_prompts - num_repeat_prompts

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    set_random_seed(0)
    for _ in range(num_repeat_prompts):
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        prompts.append(
            [
                {
                    "role": "user",
                    "content": f"""
        please repeat the word '{word}' 10 times.
        give no other output than the word at least ten times in a row,
        in lowercase with spaces between each word and without quotes.
        """,
                }
            ]
        )
    prompts.extend(
        [{"role": "user", "content": prompt}]
        for prompt in _build_gsm8k_prompts(
            num_questions=num_gsm8k_prompts, num_shots=5
        )[0]
    )
    for _ in range(num_mm_prompts):
        placeholders = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
                },
            }
        ]
        prompt = [
            *placeholders,
            {"type": "text", "text": "The meaning of the image is"},
        ]
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


def get_instruct_coder_messages(n: int) -> list[Messages]:
    from vllm.benchmarks.datasets import InstructCoderDataset

    dataset = InstructCoderDataset(
        dataset_path="likaixin/InstructCoder", dataset_split="train"
    )
    prompts: Iterable[str] = dataset.sample_prompts(n=n)
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def greedy_sampling() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


def stochastic_sampling() -> SamplingParams:
    return SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=False)


def evaluate_llm_for_gsm8k(llm: LLM, expected_accuracy_threshold: float = 0.70) -> None:
    """Evaluate the LLM on GSM8K and check that accuracy is above a sanity threshold.

    The default threshold assumes the LLM uses the same target model as the "model_name"
    fixture, with max model len == 4096. Precomputed reference value is 75% to 80%
    on GSM8K with greedy decoding, so we check that it's above a sanity threshold of 70%
    to verify that the model is correct.
    """
    if expected_accuracy_threshold <= 0.0:
        print("Skipping GSM8K evaluation")
        return
    results = evaluate_gsm8k_offline(llm)
    accuracy = results["accuracy"]
    print(f"GSM8K accuracy: {accuracy:.3f}")
    assert accuracy >= expected_accuracy_threshold, (
        f"Expected GSM8K accuracy >= {expected_accuracy_threshold}, got {accuracy:.3f}"
    )


def assert_request_outputs_match(
    ref_outputs: Sequence[RequestOutput],
    spec_outputs: Sequence[RequestOutput],
    *,
    required_matches: int,
    context: str,
    max_mismatches: int = 3,
) -> None:
    """Assert a thresholded exact-text match with bounded failure diagnostics."""
    assert ref_outputs, f"{context}: no reference outputs"
    assert len(ref_outputs) == len(spec_outputs), (
        f"{context}: output count differs: "
        f"reference={len(ref_outputs)}, speculative={len(spec_outputs)}"
    )
    assert 0 <= required_matches <= len(ref_outputs), (
        f"{context}: invalid required_matches={required_matches} for "
        f"{len(ref_outputs)} outputs"
    )

    mismatches: list[str] = []
    matches = 0
    for index, (ref_output, spec_output) in enumerate(zip(ref_outputs, spec_outputs)):
        assert ref_output.outputs, (
            f"{context}: reference output {index} has no candidate"
        )
        assert spec_output.outputs, (
            f"{context}: speculative output {index} has no candidate"
        )
        ref_candidate = ref_output.outputs[0]
        spec_candidate = spec_output.outputs[0]
        if ref_candidate.text == spec_candidate.text:
            matches += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append(
                f"[{index}] ref_text={ref_candidate.text[:240]!r}, "
                f"spec_text={spec_candidate.text[:240]!r}\n"
                f"    ref_token_ids={list(ref_candidate.token_ids)[:64]}\n"
                f"    spec_token_ids={list(spec_candidate.token_ids)[:64]}"
            )

    print(
        f"{context}: exact text matches={matches}/{len(ref_outputs)} "
        f"(required={required_matches})"
    )
    mismatch_summary = "\n".join(mismatches) or "no mismatches captured"
    assert matches >= required_matches, (
        f"{context}: only {matches}/{len(ref_outputs)} outputs matched; "
        f"required at least {required_matches}. First mismatches:\n"
        f"{mismatch_summary}"
    )


def get_spec_decode_metric_value(metrics: Sequence[Metric], metric_name: str) -> float:
    """Get a spec-decode metric with an actionable error when stats are absent."""
    name2metric = {metric.name: metric for metric in metrics}
    metric = name2metric.get(metric_name)
    assert metric is not None, (
        f"Missing metric {metric_name!r}. Ensure disable_log_stats=False. "
        "Available spec-decode metrics: "
        f"{sorted(name for name in name2metric if 'spec_decode' in name) or ['<none>']}"
    )
    assert isinstance(metric, Counter | Gauge), f"{metric_name!r} has no scalar value"
    return float(metric.value)


def compute_acceptance_rate(
    metrics: list[Metric], prev_metrics: list[Metric] | None = None
) -> float:
    n_draft_toks = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_draft_tokens"
    )
    if n_draft_toks == 0:
        return float("nan")
    n_accepted_toks = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    if prev_metrics is not None:
        n_draft_toks -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_draft_tokens"
        )
        n_accepted_toks -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_accepted_tokens"
        )
        if n_draft_toks <= 0:
            return float("nan")
    return n_accepted_toks / n_draft_toks


def compute_acceptance_len(
    metrics: list[Metric], prev_metrics: list[Metric] | None = None
) -> float:
    n_drafts = get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
    n_accepted_toks = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    if n_drafts == 0:
        return 1
    if prev_metrics is not None:
        n_drafts -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_drafts"
        )
        n_accepted_toks -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_accepted_tokens"
        )
        if n_drafts <= 0:
            return 1
    return 1 + (n_accepted_toks / n_drafts)


# --- Test-only speculators ---------------------------------------------------
#
# Registered with `register_speculative_method` as `test_*` methods. A custom
# method must be registered in every process: the tests fork their engine and
# worker processes (VLLM_WORKER_MULTIPROC_METHOD=fork) after registering.


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


@functools.cache
def _tree_speculator_class() -> type:
    # The speculator imports the model runner, so it is built lazily: configs
    # are imported in every process, including the scheduler's.
    import numpy as np
    import torch

    from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
    from vllm.v1.worker.gpu.spec_decode.round import (
        PlanBound,
        PlanInputs,
        VerifyPlan,
    )

    class NgramTreeSpeculator(NgramGPUSpeculator):
        """Verifies the n-gram drafts as a tree: with ``branch`` the first draft
        gets a sibling holding another token, and the chain continues from the
        first draft. Without it the plan is the chain itself, laid out and
        committed as a tree (``is_chain=False``)."""

        branch = True

        def plan(self, inputs: PlanInputs) -> tuple[VerifyPlan, PlanBound] | None:
            if inputs.chain_only:
                return None
            assert self.resources is not None
            num_reqs = len(inputs.req_ids)
            num_cols = self.resources.max_candidates
            if not hasattr(self, "_parent"):
                shape = (self.req_states.max_num_reqs, num_cols)
                self._parent = torch.empty(shape, dtype=torch.int32, device=self.device)
                self._tokens = torch.empty(shape, dtype=torch.int32, device=self.device)
                self._count = torch.empty(
                    shape[0], dtype=torch.int32, device=self.device
                )
            rows = inputs.scheduled_rows.astype(np.int32)
            parent = np.tile(np.arange(-1, num_cols - 1, dtype=np.int32), (num_reqs, 1))
            # Row i holds draft i - 1 (row 0, the root, is ignored).
            src = np.tile(np.arange(-1, num_cols - 1), (num_reqs, 1)).clip(min=0)
            alt = np.zeros((num_reqs, num_cols), dtype=np.int64)
            if self.branch:
                tree = rows >= 4
                # Rows: root, d1, sibling of d1, d2 (child of d1), d3, ...
                parent[tree, 2] = 0
                parent[tree, 3] = 1
                src[tree, 2] = 0
                src[tree, 3:] -= 1
                alt[tree, 2] = 1
            drafts = self.req_states.draft_tokens[inputs.idx_mapping]
            src_gpu = torch.from_numpy(src).to(self.device, non_blocking=True)
            src_gpu = src_gpu.clamp(max=drafts.shape[1] - 1)
            tokens = drafts.gather(1, src_gpu)
            tokens = tokens + torch.from_numpy(alt).to(self.device, non_blocking=True)
            # Unproposed drafts are -1: any real token verifies the same.
            tokens = (
                tokens.clamp(min=0) % self.vllm_config.model_config.get_vocab_size()
            )
            self._parent[:num_reqs].copy_(torch.from_numpy(parent), non_blocking=True)
            self._tokens[:num_reqs].copy_(tokens.to(torch.int32))
            self._count[:num_reqs].copy_(torch.from_numpy(rows), non_blocking=True)
            plan = VerifyPlan(
                parent=self._parent[:num_reqs],
                count=self._count[:num_reqs],
                tokens=self._tokens[:num_reqs],
            )
            return plan, PlanBound()

    return NgramTreeSpeculator


@functools.cache
def _chain_as_tree_speculator_class() -> type:
    return type(
        "NgramChainAsTreeSpeculator", (_tree_speculator_class(),), {"branch": False}
    )


def __getattr__(name: str) -> Any:
    # Resolved by qualname in the worker (`speculator_class`).
    if name == "NgramTreeSpeculator":
        return _tree_speculator_class()
    if name == "NgramChainAsTreeSpeculator":
        return _chain_as_tree_speculator_class()
    raise AttributeError(name)


class NgramTreeConfig(NgramGPUSpeculatorConfig):
    """A 2-way branching speculator built on the n-gram drafts."""

    speculator_class = f"{__name__}.NgramTreeSpeculator"

    def resources(self, max_num_seqs: int) -> SpecResources:
        return replace(super().resources(max_num_seqs), max_branching=2)


class NgramChainAsTreeConfig(NgramTreeConfig):
    speculator_class = f"{__name__}.NgramChainAsTreeSpeculator"


TEST_SPECULATORS = {
    "test_ngram_budget_one": NgramBudgetOneConfig,
    "test_eagle3_budget_one": Eagle3BudgetOneConfig,
    "test_mtp_alternating_budget": MTPAlternatingBudgetConfig,
    "test_ngram_budget_one_on_rank_zero": NgramBudgetOneOnRankZeroConfig,
    "test_ngram_tree": NgramTreeConfig,
    "test_ngram_chain_as_tree": NgramChainAsTreeConfig,
}


def register_test_speculators() -> None:
    for name, config_cls in TEST_SPECULATORS.items():
        register_speculative_method(name, config_cls)
