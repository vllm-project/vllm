# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime preflight: loud guards enforced at selection time.

The pytest suite pins these clean at the developer's HEAD. Production runs
against newer vLLM checkouts where nobody runs that suite, so every degraded
input that would quietly weaken selection surfaces here with a direction:

- force one step: unknown step fields, duplicate ids, unparsable commands.
- run everything: an empty core table or a dead anchor, meaning we are blind to
  a whole coverage channel.
- degrade one gate: an unresolved engine entry module turns off the worker-seam
  gate instead of dropping suites.
- fail one file open: a changed path that would not parse.
- distrust the graph: an unmodeled dynamic import means the closure may be
  missing edges, so rules that trust the graph over a step's declared deps
  stand down until it is classified.
- warn only: steps that legitimately have no targets, and steps whose tests
  live inside their container image rather than in this checkout.

No step is exempt from being forced. A soft_fail step cannot gate the merge,
but people still read its result, so it is forced like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..handwritten import ENGINE_ENTRY_MODULES, INFRA_DEVICES
from .hardware import family_of_device


@dataclass
class PreflightReport:
    run_all_reasons: list[str] = field(default_factory=list)
    force_select: dict[str, str] = field(default_factory=dict)  # step_id -> why
    seam_gate_ok: bool = True
    parse_error_paths: frozenset[str] = frozenset()
    # devices the taxonomy cannot map, which makes family_steps() incomplete
    # and the no-hardware rule unsafe
    unmapped_devices: frozenset[str] = frozenset()
    # files holding a dynamic import no parser models. Non-empty means the
    # graph is provably incomplete, the one case where trusting it breaks down.
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


def _check_dynamic_sites(pf: PreflightReport, full) -> None:
    """The same audit the drift tests run, at selection time. Nearly free: the
    sites come out of the graph build and only the files holding one are reread.

    The two failures point different ways. A walker that found nothing is
    broken itself, so run everything. An unclassified site means edges we
    cannot list are missing, so that file fails open and the rules trusting the
    graph stand down, which is bounded rather than total.

    unused_external_entries is deliberately not run here. Selection reads the
    checkout at a PR's merge base, which can be far older than the list this
    package ships, and against an older tree an unused entry is expected rather
    than stale. Its home is the drift tests, which run at head."""
    from ..validate.dynamic_sites import classify_dynamic_sites

    sites = full.graph.dynamic_sites
    if not sites:
        pf.run_all_reasons.append(
            "preflight: the dynamic-import walker produced zero sites; "
            "the audit passes vacuously and unmodeled edges are invisible"
        )
        return
    unclassified = classify_dynamic_sites(sites, full.graph.table_files).unclassified
    if unclassified:
        pf.unclassified_sites = frozenset(s.file for s in unclassified)
        ordered = sorted(unclassified, key=lambda s: (s.file, s.lineno))
        shown = ", ".join(f"{s.file}:{s.lineno} ({s.func})" for s in ordered[:5])
        pf.warnings.append(
            f"preflight: {len(unclassified)} dynamic import(s) the graph cannot "
            f"follow, so tests reached only through them may not be selected: "
            f"{shown}. Fix by adding the file to DYNAMIC_IMPORT_FILES in "
            f"ci_selector/handwritten.py if it loads something outside the "
            f"repo, or by teaching a parser in codemap/graph/factories.py to "
            f"read the table it dispatches off."
        )


def run_preflight(repo: Path, pipelines, full, load_report) -> PreflightReport:
    """repo: the checkout root. pipelines: PipelineData-shaped objects. full:
    the FullGraph. load_report: the LoadReport from the yaml load."""
    pf = PreflightReport()

    # Every distrusted step is forced, soft_fail included: suppressing one
    # trades a signal people read for a saving nobody measured.
    def escalate(sid: str, why: str) -> None:
        pf.force_select[sid] = why

    for name, step_ids in load_report.unknown_fields.items():
        for sid in step_ids:
            escalate(sid, f"preflight: unknown step field '{name}' may gate execution")
    for sid in load_report.duplicate_ids:
        escalate(sid, "preflight: duplicate step id; target maps collided")

    zero_target: list[str] = []
    container_only: list[str] = []
    for pdata in pipelines:
        for step in pdata.steps:
            st = pdata.targets.get(step.step_id)
            if st is None:
                continue
            # Collected before the skips below, which a step with any other
            # coverage exits through. The point is that these tests are
            # unreachable from here whatever else the step runs.
            if st.container_tests:
                container_only.append(step.step_id)
            if st.unparsable:
                escalate(
                    step.step_id,
                    "preflight: unparsable command "
                    f"{st.unparsable[0]!r}; targets unknowable",
                )
                continue
            if step.always_runs or not step.commands:
                continue
            # A dangling pytest target is a stale hole, so force the step and
            # a newer checkout cannot quietly drop it. Checked before the
            # other-coverage skip, or a step holding one real target alongside
            # a stale one would never reach this.
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
    if container_only:
        pf.warnings.append(
            f"preflight: {len(container_only)} steps name tests that exist only "
            "inside their container image, so this checkout cannot map them: "
            + ", ".join(sorted(container_only))
        )

    if full.graph.parse_errors:
        pf.parse_error_paths = frozenset(full.graph.parse_errors)
        pf.warnings.append(
            f"preflight: {len(full.graph.parse_errors)} files failed to "
            "parse (their edges are missing; changed ones fail open): "
            + ", ".join(sorted(full.graph.parse_errors)[:5])
        )

    _check_dynamic_sites(pf, full)

    tables = {
        "model registry entries": full.registry.entries,
        "HF example ids": full.registry.hf_ids,
        "quant methods": full.quant.methods,
        "lazy parser tables": full.factories.parser_entries,
        # The lazy tables merge into that one dict and their parser names
        # collide, so the row above only fires if they all die at once. Guard
        # each on its own count. The merged row stays as the floor for when the
        # parser pass never runs and these rows are missing.
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
