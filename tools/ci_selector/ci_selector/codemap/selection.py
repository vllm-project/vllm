# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The answer, and the only place that writes to it.

`Selection` is what a run produces; everything here appends to one. The
question this module answers is whether a claim reaches a step, never why the
claim exists -- `classify.py` owns that and hands the finished claim over.

Each recorded reason carries its rule name and, when the record may weigh it,
the changed files behind it. The three lists stay index-aligned, so anything
reading them can pair a reason with its rule and its files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import hardware
from .claim import OUTPUT_RULES, Claim
from .pipeline.step import Step
from .pipeline.targets import StepTargets
from .state import PipelineData, RepoState


@dataclass
class Selection:
    # step_id -> reasons (auto-run steps only)
    selected: dict[str, list[str]] = field(default_factory=dict)
    # optional steps a rule hit: today these stay manual; shown, never run
    manual_hits: dict[str, list[str]] = field(default_factory=dict)
    # step_id -> the rule names behind those reasons, same order. Parallel
    # rather than a richer reason value: prose is for people, and the record
    # needs a key it can route on.
    selected_rules: dict[str, list[str]] = field(default_factory=dict)
    manual_rules: dict[str, list[str]] = field(default_factory=dict)
    # step_id -> per reason, the changed files the record may weigh, or None
    # when the reason is not droppable. Same order as selected_rules.
    selected_paths: dict[str, list[list[str] | None]] = field(default_factory=dict)
    # changed file -> the auto steps selected because of it. The inverse of
    # `selected`, and the only thing that answers "what does the map say about
    # this one file". A step in no value here is always selected anyway.
    selected_by_file: dict[str, list[str]] = field(default_factory=dict)
    # pipeline -> the changed file whose claim escalated it to run-all
    run_all_paths: dict[str, str] = field(default_factory=dict)
    manual_paths: dict[str, list[list[str] | None]] = field(default_factory=dict)
    # pipeline -> reason, when everything there runs
    run_all: dict[str, str] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    docs_only: bool = False
    # does the diff touch the docs build's dependency set (with tagged reasons)
    docs_affected: bool = False
    docs_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _record(
    sel: Selection,
    step: Step,
    reason: str,
    rule: str,
    paths: set[str] | None = None,
    by_file: str | None = None,
) -> None:
    if rule not in OUTPUT_RULES:
        raise ValueError(f"unpinned selection rule {rule!r}; add it to rules.RULES")
    manual = step.manual_only
    if by_file is not None and not manual:
        seen = sel.selected_by_file.setdefault(by_file, [])
        if step.step_id not in seen:
            seen.append(step.step_id)
    bucket = sel.manual_hits if manual else sel.selected
    rules = sel.manual_rules if manual else sel.selected_rules
    attrib = sel.manual_paths if manual else sel.selected_paths
    bucket.setdefault(step.step_id, []).append(reason)
    rules.setdefault(step.step_id, []).append(rule)
    # None, not empty: "no file answers for this" and "cannot be attributed at
    # all" must not read the same.
    attrib.setdefault(step.step_id, []).append(sorted(paths) if paths else None)


def _apply_preflight(state: RepoState, sel: Selection) -> None:
    pf = state.preflight
    for reason in pf.run_all_reasons:
        sel.notes.append(reason)
        for pdata in state.pipelines:
            sel.run_all.setdefault(pdata.config.name, reason)
    for pdata in state.pipelines:
        for step in pdata.steps:
            reason = pf.force_select.get(step.step_id)
            if reason:
                _record(sel, step, reason, "preflight")
    sel.notes.extend(pf.warnings)


def _apply_claim_to_pipeline(
    state: RepoState,
    sel: Selection,
    claim: Claim,
    pdata: PipelineData,
    path: str,
    paths: set[str] | None = None,
) -> None:
    """`path` names the reason strings; `paths` is what the record may weigh.
    They differ when a claim carries coverage for a file added alongside it."""
    paths = paths or {path}
    for step in pdata.steps:
        # Exclusivity answers "can this device run the file", which is about
        # inferred reach. A step whose own command collects the file imports it
        # regardless, and a bad import still fails that job, so collecting it
        # directly disarms the subtraction.
        if (
            path not in state.exclusive_disabled
            and hardware.device_excluded_for_path(path, step.device, step)
            and not _directly_collects(pdata.targets.get(step.step_id), path)
        ):
            continue
        if step.step_id in claim.step_ids:
            droppable = step.step_id in claim.droppable_step_ids
            _record(
                sel,
                step,
                f"{path}: {claim.detail}",
                claim.rule,
                paths if droppable else None,
                by_file=path,
            )
            continue
        if not claim.test_files:
            continue
        # The closure coverage of a device-named data file is ordinary pytest,
        # but a step on a different device loads its own config and not this
        # file, so scope that routing to the file's device. step_ids is left
        # alone: a declared dep is the generator's own trigger and must run.
        if claim.device_scope and hardware.device_scoped_out(step, claim.device_scope):
            continue
        st = pdata.targets.get(step.step_id)
        if st is None:
            continue
        hit = _targets_cover(st, claim.test_files)
        if hit:
            _record(
                sel,
                step,
                f"{path} -> {hit} -> {step.label}",
                claim.rule,
                paths if claim.droppable_test_files else None,
                by_file=path,
            )


def _targets_cover(st: StepTargets, test_files: set[str]) -> str | None:
    """The test file proving this step reads the claim, or None.

    Targets are walked in command order, but inside a directory target the
    match is the alphabetical first and not whichever the set yielded first.
    Set order for strings shifts with the interpreter's hash seed, so returning
    the first match made this function, and every reason string quoting it,
    differ between two runs on the same diff. `min` and not `sorted` because
    this runs per step and claim over very large closures.
    """
    for t in st.targets:
        if t.path.endswith(".py"):
            if t.path in test_files:
                return t.path
        else:
            prefix = t.path.rstrip("/") + "/"
            hit = min((f for f in test_files if f.startswith(prefix)), default=None)
            if hit is not None:
                return hit
    return None


def _directly_collects(st: StepTargets | None, path: str) -> bool:
    """True when this step's own command loads `path`: a named target, a file
    under a directory target, a scanned script, or a data argument.

    The directory leg is what _direct_step_refs lacks, and they stay separate
    because this exists to disarm a subtraction, not to add coverage.

    --ignore and --deselect are honoured here even though _targets_cover
    ignores them, because the two point opposite ways. Over-claiming coverage
    there only over-selects; over-claiming it HERE keeps a step the hardware
    rule was right to drop."""
    if st is None:
        return False
    if any(path == ig or path.startswith(ig.rstrip("/") + "/") for ig in st.ignored):
        return False
    if path in st.data_files or path in st.scripts_seen:
        return True
    for t in st.targets:
        if t.path == path:
            return True
        if not t.path.endswith(".py") and path.startswith(t.path.rstrip("/") + "/"):
            return True
    return False


def _apply_run_all(state: RepoState, sel: Selection) -> None:
    for pdata in state.pipelines:
        reason = sel.run_all.get(pdata.config.name)
        if not reason:
            continue
        cause = sel.run_all_paths.get(pdata.config.name)
        for step in pdata.steps:
            _record(sel, step, reason, "run-all", by_file=cause)


def _add_always_run(state: RepoState, sel: Selection) -> None:
    for pdata in state.pipelines:
        for step in pdata.steps:
            if step.always_runs:
                # Not _record: an always-run step is never manual. All three
                # lists must stay aligned, since consumers pair them by index.
                sel.selected.setdefault(step.step_id, []).append(
                    "always-run key shortcut (image-build*/AMD base)"
                )
                sel.selected_rules.setdefault(step.step_id, []).append("always-run")
                sel.selected_paths.setdefault(step.step_id, []).append(None)
