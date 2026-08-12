# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime preflight: fail-loud guards enforced at selection time.

The pytest suite pins these states clean at the developer's HEAD; production
runs against newer vLLM checkouts where nobody runs that suite. Every degraded
input that would silently weaken selection is surfaced here with a polarity:

- scoped escalation (force-select the affected step): unknown step fields,
  duplicate step ids, UNPARSABLE commands -- the distrusted step runs.
- global escalation (run-all): an empty core table or dead anchor, meaning the
  analyzer is blind to a whole coverage channel.
- surgical degrade: an unresolved engine entry module disables the worker-seam
  gate rather than silently dropping suites.
- per-file escalation: a changed path that failed to parse fails open.
- graph distrust: an unmodeled dynamic-import site means the closure may be
  missing edges, so rules that trust the graph over a step's declared deps stand
  down until it is classified.
- warning only: legitimately zero-target steps (rust, HPU/NPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .curated import ENGINE_ENTRY_MODULES, INFRA_DEVICES
from .hardware import family_of_device


@dataclass
class PreflightReport:
    run_all_reasons: list[str] = field(default_factory=list)
    force_select: dict[str, str] = field(default_factory=dict)  # step_id -> why
    seam_gate_ok: bool = True
    parse_error_paths: frozenset[str] = frozenset()
    # device strings the curated taxonomy cannot map: family_steps() is
    # incomplete, so the no-hardware zero-jobs rule must not fire
    unmapped_devices: frozenset[str] = frozenset()
    # files holding a dynamic import no parser models. Non-empty means the
    # import graph is demonstrably incomplete, which is the one condition
    # under which "the graph is authoritative" stops being true.
    unclassified_sites: frozenset[str] = frozenset()
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            not self.run_all_reasons
            and not self.force_select
            and self.seam_gate_ok
            and not self.parse_error_paths
            and not self.unclassified_sites
            and not self.warnings
        )


def _check_dynamic_sites(pf: PreflightReport, repo: Path, full) -> None:
    """The same audit `ci-validate dynamic-sites` runs, at selection time.

    Nearly free: the sites are collected during the graph build, and
    classification only re-reads the ~50 files that have one.

    Polarities differ because the failures differ. A collapsed walker means the
    detector itself is unreliable, which is the empty-table case -> run
    everything. An unclassified site means the graph is missing edges we cannot
    enumerate (that is what unclassified means), so the file itself fails open
    and the graph-over-declarations rules stand down, which is bounded rather
    than total.

    census_rot is deliberately NOT run here. Selection analyzes the checkout at
    a PR's merge base, which can be arbitrarily older than the census shipped in
    this package, and against an older tree "dead" and "missing" entries are the
    expected reading rather than rot -- escalating on it would run everything
    for every PR based before the newest census entry. Its home is the
    `dynamic-sites` check, which runs at head where the comparison is fair."""
    from .validate.dynamic_sites import classify_dynamic_sites

    sites = full.graph.dynamic_sites
    if not sites:
        pf.run_all_reasons.append(
            "preflight: the dynamic-import walker produced zero sites; "
            "the audit passes vacuously and unmodeled edges are invisible"
        )
        return
    unclassified = classify_dynamic_sites(repo, sites).unclassified
    if unclassified:
        pf.unclassified_sites = frozenset(s.file for s in unclassified)
        ordered = sorted(unclassified, key=lambda s: (s.file, s.lineno))
        shown = ", ".join(f"{s.file}:{s.lineno} ({s.func})" for s in ordered[:5])
        pf.warnings.append(
            f"preflight: {len(unclassified)} unmodeled dynamic-import site(s); "
            f"the graph may be missing edges, so declared deps are trusted over "
            f"it until they are classified or pragma'd: {shown}"
        )


def run_preflight(repo: Path, pipelines, full, load_report) -> PreflightReport:
    """repo: the checkout root; pipelines: PipelineData-shaped objects (.steps,
    .targets); full: the FullGraph; load_report: the yaml-load LoadReport."""
    pf = PreflightReport()
    # An escalation exists so a distrusted step cannot silently miss a failure.
    # A soft_fail step's failure never gates the merge, so escalating it buys
    # no recall and costs a job on every PR. Derived from the yaml, so a step
    # flipped back to hard-fail re-arms itself.
    soft_fail = {s.step_id for pdata in pipelines for s in pdata.steps if s.soft_fail}

    def escalate(sid: str, why: str) -> None:
        if sid not in soft_fail:
            pf.force_select[sid] = why

    for name, step_ids in load_report.unknown_fields.items():
        for sid in step_ids:
            escalate(sid, f"preflight: unknown step field '{name}' may gate execution")
    for sid in load_report.duplicate_ids:
        escalate(sid, "preflight: duplicate step id; target maps collided")

    zero_target: list[str] = []
    for pdata in pipelines:
        for step in pdata.steps:
            st = pdata.targets.get(step.step_id)
            if st is None:
                continue
            if st.unparsable:
                escalate(
                    step.step_id,
                    "preflight: unparsable command "
                    f"{st.unparsable[0]!r}; targets unknowable",
                )
                continue
            if step.always_runs or not step.commands:
                continue
            # Dangling pytest target (renamed / glob matched zero) = STALE
            # hole -> force-select so a newer checkout can't silently drop it.
            # Checked BEFORE the other-coverage skip: a step can hold a real
            # target (or merely a scanned script) alongside a stale one, and
            # skipping on that made the escalation unreachable for it.
            if st.dangling:
                escalate(
                    step.step_id,
                    f"preflight: pytest target dangling ({st.dangling[0]!r}); "
                    "coverage stale, running the step",
                )
                continue
            if st.targets or st.scripts_seen or st.data_files:
                continue
            zero_target.append(step.step_id)
    if zero_target:
        pf.warnings.append(
            f"preflight: {len(zero_target)} steps have no derivable targets "
            f"(non-pytest bodies): {', '.join(sorted(zero_target))}"
        )

    if full.graph.parse_errors:
        pf.parse_error_paths = frozenset(full.graph.parse_errors)
        pf.warnings.append(
            f"preflight: {len(full.graph.parse_errors)} files failed to "
            "parse (their edges are missing; changed ones fail open): "
            + ", ".join(sorted(full.graph.parse_errors)[:5])
        )

    _check_dynamic_sites(pf, repo, full)

    tables = {
        "model registry entries": full.registry.entries,
        "HF example ids": full.registry.hf_ids,
        "quant methods": full.quant.methods,
        "lazy parser tables": full.factories.parser_entries,
        # The four lazy tables merge into that one dict and their parser names
        # collide, so the row above only fires when all four die at once -- a
        # dead tokenizers anchor leaves its size unchanged. Guard each table on
        # its own pre-merge count. The merged row stays as the floor for the
        # case where the parser pass never runs and these rows are absent.
        **{
            f"lazy parser table {anchor}": count
            for anchor, count in full.factories.parser_table_counts.items()
        },
        "register_* entries": full.factories.register_entries,
        "attention enum": full.factories.enum_entries,
        "parser engine entries": full.factories.parser_engine_entries,
        "class table entries": full.factories.class_table_entries,
        "MODULE_ATTRS": full.factories.module_attrs,
    }
    for name, table in tables.items():
        if not table:
            pf.run_all_reasons.append(
                f"preflight: {name} parsed empty; the analyzer is blind to "
                "that coverage channel"
            )
    if full.factories.module_attrs and not full.factories.module_attr_resolved:
        pf.run_all_reasons.append(
            "preflight: MODULE_ATTRS parsed but zero aliases resolved; "
            "from-vllm import edges are dead"
        )
    if full.spawn.entrypoint_file is None:
        pf.run_all_reasons.append(
            "preflight: CLI entrypoint module unresolved; server-spawn edges are dead"
        )

    unmapped = {
        s.device
        for pdata in pipelines
        for s in pdata.steps
        if s.device
        and s.device not in INFRA_DEVICES
        and family_of_device(s.device) is None
    }
    if unmapped:
        pf.unmapped_devices = frozenset(unmapped)
        pf.warnings.append(
            "preflight: devices outside the curated taxonomy "
            f"({', '.join(sorted(unmapped))}); family routing is incomplete "
            "and the no-hardware rule is disabled"
        )

    unresolved = [m for m in ENGINE_ENTRY_MODULES if not full.index.resolve(m)]
    if unresolved:
        pf.seam_gate_ok = False
        pf.warnings.append(
            "preflight: engine entry modules unresolved "
            f"({', '.join(unresolved)}); worker-seam gate disabled "
            "(over-selecting)"
        )
    return pf
