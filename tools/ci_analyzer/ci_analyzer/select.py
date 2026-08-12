# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The selector: changed paths -> Buildkite steps, with reasons.

Rule NAMES are pinned in policy.RULES and emitted per step in
Selection.selected_rules, because narrowing downstream is only sound when
every rule that selected a step is one the narrower has evidence about.

SINGLE HOME OF RULE ORDER (names, not numbers -- numbers fork). A docs-only
short-circuit runs first, then table claims (registry files diffed entry-wise
pre-empt everything below), then per file the first matching claim wins in this
order:

  world -> buildkite chain -> no-hardware -> graph -> no-code -> status-A
  added-file family -> renamed-or-copied -> requirements-file -> release-ci ->
  exclusive-family scoped fail-open -> target-coverage -> package-data ->
  declared-deps -> terminal fail-open run-all.

Preflight escalations (force-select, global run-all, seam-gate disable) apply
after the per-file claims.

State is built at the diff BASE (harnesses + CLI worktree mode guarantee it).
At a head-built state added files are already graph-known, so the status-A rules
never fire (polarity-safe but less precise) and table scoping reads head-side
literals. The head-closure rule builds a SECOND graph at the head and degrades to
the fail-open chain if unreachable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from . import hardware, tablediff
from .curated import (
    INERT_CI_PREFIXES,
    LEGACY_CI_FILES,
    PACKAGE_ROOTS,
    RELEASE_PIPELINE_FILES,
)
from .docs import DocsDeps, build_docs_deps
from .externals import docker_image_inputs, release_pipeline_refs
from .gitdiff import diff_files
from .graph.build import FullGraph, build_full_graph
from .graph.registry import resolve_module_name
from .jobs.buildkite import load_pipeline_configs, load_steps
from .jobs.invoked import invoked_files, legacy_amd_invoked
from .jobs.model import LoadReport, PipelineConfig, Step
from .jobs.scripts import scan_script
from .jobs.testmap import StepTargets, map_step
from .keys import KeyIndex
from .policy import (
    OUTPUT_RULES,
    Claim,
    classify_world,
    docs_only,
    is_no_code,
    step_declares,
)
from .preflight import PreflightReport, run_preflight
from .repo import is_test_basename, is_test_file, test_file_catalog
from .worldtables import narrowable_world_paths


@dataclass
class DiffContext:
    """Refs and per-path statuses; present only when select() is given both
    ends of a real diff, which is what activates table-aware treatment and
    the status-A added-file rules (both assume state built at `base`)."""

    base: str
    head: str
    status: dict[str, str]  # path -> A/M/D/R/C/T
    renames: dict[str, str] = field(default_factory=dict)  # new path -> old path
    # EXTRA_WORLD_FILES paths whose change is confined to config-only tables,
    # so classify_world defers to CI instead of forcing analyzer-policy world.
    world_config_only: set[str] = field(default_factory=set)


@dataclass
class PipelineData:
    config: PipelineConfig
    steps: list[Step]
    targets: dict[str, StepTargets]  # step_id -> targets


@dataclass
class AnalyzerState:
    """Everything derivable from a checkout, reusable across diffs."""

    repo: Path
    pipelines: list[PipelineData]
    full: FullGraph
    catalog: list[str]
    load_report: LoadReport
    # test files at least one AUTO-RUN step invokes (orphans and coverage
    # reachable only via optional steps excluded): the zero-closure polarity
    # counts only these
    invoked: set[str] = field(default_factory=set)
    keys: KeyIndex = field(default_factory=KeyIndex)
    auto_step_ids: set[str] = field(default_factory=set)
    auto_covered_files: set[str] = field(default_factory=set)
    auto_prefixes: tuple[str, ...] = ()
    # test files only the legacy test-amd.yaml invokes (external pipeline)
    legacy_invoked: set[str] = field(default_factory=set)
    preflight: PreflightReport = field(default_factory=PreflightReport)
    # exclusive-namespace members with a live cross-family module-level
    # importer: their subtractive exclusion is disabled (fail-open)
    exclusive_disabled: set[str] = field(default_factory=set)
    # the docs build's derived file-dependency set (docs_affected signal)
    docs_deps: DocsDeps = field(default_factory=DocsDeps)
    # files referenced only by the release/nightly pipeline (select nothing)
    release_refs: frozenset[str] = frozenset()
    # repo file -> Dockerfile that COPY/ADDs it (relabel the run-all reason)
    docker_inputs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, repo: Path) -> AnalyzerState:
        report = LoadReport()
        pipelines = []
        for config in load_pipeline_configs(repo):
            steps = load_steps(repo, config, report)
            detect_duplicate_ids(steps, report)
            targets = {
                s.step_id: map_step(repo, s, script_scanner=scan_script) for s in steps
            }
            pipelines.append(PipelineData(config, steps, targets))
        full = build_full_graph(repo)
        state = cls(
            repo=repo,
            pipelines=pipelines,
            full=full,
            catalog=test_file_catalog(repo),
            load_report=report,
        )
        auto_targets = []
        for p in pipelines:
            for s in p.steps:
                if s.manual_only:
                    continue
                state.auto_step_ids.add(s.step_id)
                st = p.targets.get(s.step_id)
                if st is not None:
                    auto_targets.append(st)
        state.invoked = invoked_files(state.catalog, auto_targets)
        prefixes: set[str] = set()
        for st in auto_targets:
            state.auto_covered_files.update(st.data_files)
            state.auto_covered_files.update(st.scripts_seen)
            for t in st.targets:
                if t.path.endswith(".py"):
                    state.auto_covered_files.add(t.path)
                else:
                    prefixes.add(t.path.rstrip("/") + "/")
        state.auto_prefixes = tuple(sorted(prefixes))
        state.legacy_invoked = legacy_amd_invoked(repo, state.catalog)
        state.keys = KeyIndex.build(repo, full, pipelines)
        state.exclusive_disabled = set(
            hardware.exclusivity_violations(
                full.plain_reverse, full.index.file_to_module
            )
        )
        state.preflight = run_preflight(repo, pipelines, full, report)
        state.docs_deps = build_docs_deps(repo)
        state.release_refs = release_pipeline_refs(repo)
        state.docker_inputs = docker_image_inputs(repo)
        return state

    def family_steps(self, family: str) -> set[str]:
        return {
            s.step_id
            for p in self.pipelines
            for s in p.steps
            if hardware.step_in_family(s, family)
        }


@dataclass
class Selection:
    # step_id -> reasons (auto-run steps only)
    selected: dict[str, list[str]] = field(default_factory=dict)
    # optional steps a rule hit: today these stay manual; shown, never run
    manual_hits: dict[str, list[str]] = field(default_factory=dict)
    # step_id -> the RULE names behind those reasons, same order. A parallel
    # field rather than a richer reason value: prose is what a human reads,
    # and pass 2 needs a key it can route on.
    selected_rules: dict[str, list[str]] = field(default_factory=dict)
    manual_rules: dict[str, list[str]] = field(default_factory=dict)
    # pipeline -> reason, when everything there runs
    run_all: dict[str, str] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    docs_only: bool = False
    # does the diff touch the docs build's dependency set (with tagged reasons)
    docs_affected: bool = False
    docs_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def select(
    state: AnalyzerState,
    paths: list[str],
    *,
    base: str | None = None,
    head: str | None = None,
) -> Selection:
    sel = Selection()
    sel.docs_affected, sel.docs_reasons = state.docs_deps.docs_affected(paths)
    if docs_only(paths):
        sel.docs_only = True
        sel.notes.append("docs-only diff: generator emits no steps at all")
        return sel
    ctx = _diff_context(state, base, head)
    # Table files are classified first: their diffed claims may take over
    # the routing of files ADDED by the same diff (a new model file is
    # covered by its new registry entry instead of failing open).
    table_claims: dict[str, Claim] = {}
    covered_added: dict[str, str] = {}  # added path -> table path covering it
    if ctx is not None:
        for path in dict.fromkeys(paths):
            if path in tablediff.TABLE_FILES:
                claim, added = _classify_table(state, path, ctx)
                if claim is not None:
                    table_claims[path] = _apply_declarer_union(state, path, claim)
                    for a in added:
                        covered_added[a] = path
    for path in dict.fromkeys(paths):
        if path in table_claims:
            claim = table_claims[path]
        elif path in covered_added:
            claim = _apply_declarer_union(
                state,
                path,
                Claim(
                    "table-diff",
                    f"{path} is a newly registered module; coverage carried by "
                    f"{covered_added[path]}'s table claim",
                ),
            )
        else:
            claim = _classify(state, path, ctx)
        sel.claims.append(claim)
        for pipeline in claim.run_all:
            sel.run_all.setdefault(pipeline, f"{claim.rule}: {claim.detail}")
        for pdata in state.pipelines:
            _apply_claim_to_pipeline(state, sel, claim, pdata, path)
    for path in dict.fromkeys(paths):
        if path in state.exclusive_disabled:
            sel.notes.append(
                f"{path}: cross-family module-level importer exists; "
                "hardware exclusion disabled (fail-open)"
            )
    _apply_preflight(state, sel)
    _apply_run_all(state, sel)
    _add_always_run(state, sel)
    return sel


def _apply_preflight(state: AnalyzerState, sel: Selection) -> None:
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


def detect_duplicate_ids(steps: list[Step], report: LoadReport) -> None:
    seen: set[str] = set()
    for s in steps:
        if s.step_id in seen and s.step_id not in report.duplicate_ids:
            report.duplicate_ids.append(s.step_id)
        seen.add(s.step_id)


def _diff_context(
    state: AnalyzerState, base: str | None, head: str | None
) -> DiffContext | None:
    if not base or not head:
        return None
    try:
        files = diff_files(state.repo, base, head)
    except Exception:
        return None
    status = {f.path: f.status for f in files}
    renames = {f.path: f.old_path for f in files if f.old_path}
    for f in files:
        # Only a RENAME source vanishes; a COPY source still exists at head, so
        # marking it deleted would be a latent lie (nothing consumes "D" today).
        if f.old_path and f.status == "R":
            status.setdefault(f.old_path, "D")
    return DiffContext(
        base=base,
        head=head,
        status=status,
        renames=renames,
        world_config_only=narrowable_world_paths(state, base, head, set(status)),
    )


def _source_dep_steps(
    state: AnalyzerState, path: str, specific_only: bool = False
) -> set[str]:
    """Steps that declare `path` in their source_file_dependencies. For a
    file the import graph cannot reach (requirements, rust/, cmake), the
    step's own declaration is the ground truth -- it is the generator's
    selection mechanism. Mirror variants carry their own unioned deps.

    With specific_only, a step counts only when a dep entry MORE SPECIFIC than
    a catch-all prefix (bare `vllm/` or `tests/`) matches. On a GRAPH-KNOWN
    file the import graph is the authoritative coverage, so a blanket `vllm/`
    declaration adds only the CI config's own over-declaration -- keeping it
    caps realized savings at zero. Graph-blind files always use the full union
    (the declaration is their only signal), so this is a policy divergence:
    the analyzer trusts its graph over the generator's lazy blanket. See the
    catch-all-only-declarers note the wrapper adds and crosscheck's
    miss_catchall bucket."""
    return {
        s.step_id
        for p in state.pipelines
        for s in p.steps
        if step_declares(s.source_file_dependencies, path, specific_only)
    }


def _graph_known(state: AnalyzerState, path: str) -> bool:
    g = state.full.graph
    return (
        path in state.full.index.file_to_module
        or path in g.imports
        or path in g.reverse
    )


def _classify_requirements(state: AnalyzerState, path: str) -> Claim | None:
    """A requirements file: route to the UNION of the steps that declare it and
    its device family's jobs (the filename encodes the device). Broad
    requirements files are already claimed as `world` upstream, so only the
    narrow ones reach here. None (no declaring step, no device family) falls
    through to the terminal fail-open -- the safe default when unmappable."""
    family = hardware.family_of_path(path)
    fam_steps = state.family_steps(family) if family else set()
    step_ids = _source_dep_steps(state, path) | fam_steps
    if step_ids & state.auto_step_ids:
        return Claim(
            "requirements",
            f"{path}: declaring steps + {family or 'no'} device family",
            step_ids=step_ids,
        )
    if step_ids:
        # Manual-only coverage would silently select nothing: fall through
        # to the terminal fail-open instead.
        return None
    if family:
        return Claim(
            "requirements",
            f"{path} maps to {family}; that device runs only on an "
            "external/unmodeled pipeline; nothing to run",
        )
    return None


_ROOT_PREFIXES = tuple(f"{root}/" for root in PACKAGE_ROOTS)


def _classify_declared_deps(state: AnalyzerState, path: str) -> Claim | None:
    """Last chance before the terminal fail-open: route a file the graph is
    structurally blind to (rust/, cmake/cpu_extension.cmake) via the steps
    that declare it -- the generator's own mechanism, so this reproduces what
    real CI runs for the same diff. Fires only OUTSIDE the indexed package
    roots: inside them, graph-unknown is an anomaly owned by the zero-closure
    polarity, and dozens of steps declare a blanket vllm/ dep that would
    swallow every unmodelable vllm asset (tuning jsons, eval yamls). Requires
    an auto-run declarer, else falls through to run-all. The scoped-exclusive
    fail-open now consults this rule FIRST for family-exclusive paths; for
    those the apply-time device re-filter subtracts other-family declarers, so
    a family-exclusive path's claim is sometimes device-subtracted (a
    non-exclusive path reaching this rule directly is not). Known divergence:
    rust *.md hits no-code earlier while the generator would run the rust
    steps -- markdown cannot break a cargo build."""
    if path.startswith(_ROOT_PREFIXES):
        return None
    declarers = _source_dep_steps(state, path)
    if not declarers & state.auto_step_ids:
        return None
    detail = (
        f"{path}: routed to {len(declarers)} steps declaring it in "
        "source_file_dependencies"
    )
    step_ids = set(declarers)
    family = hardware.family_of_path(path)
    if family:
        # cpu.yaml's other CPU suites compile cpu_extension.cmake in-step
        # without declaring it; the family union covers that class.
        step_ids |= state.family_steps(family)
        detail += f" + {family} device family"
    return Claim("declared-deps", detail, step_ids=step_ids)


def _step_target_coverage(state: AnalyzerState, path: str) -> set[str]:
    """Steps whose targets cover a tests/benchmarks-side file, four ways: a
    directory target containing it; a .py-target whose (non-root) parent dir
    contains it (the lm-eval shape -- config yamls sit beside the step's test
    file); a direct data/script reference; and, for conftest/__init__ only,
    any target UNDER its directory (descendant-effect files)."""
    covering: set[str] = set()
    base = path.rsplit("/", 1)[-1]
    subtree = base in ("conftest.py", "__init__.py")
    file_dir = path.rsplit("/", 1)[0]
    dir_prefix = file_dir + "/"
    for p in state.pipelines:
        for sid, st in p.targets.items():
            if path in st.data_files or path in st.scripts_seen:
                covering.add(sid)
                continue
            for t in st.targets:
                if subtree and t.path.startswith(dir_prefix):
                    covering.add(sid)
                    break
                if not t.path.endswith(".py"):
                    if path.startswith(t.path.rstrip("/") + "/"):
                        covering.add(sid)
                        break
                elif "/" in t.path:
                    # The file must sit in the test's own directory or a DIRECT
                    # child -- a deep descendant of a shallow parent (tests/v1)
                    # is not a dependency of a test that merely lives there.
                    parent = t.path.rsplit("/", 1)[0]
                    if parent + "/" not in _ROOT_PREFIXES and parent in (
                        file_dir,
                        file_dir.rsplit("/", 1)[0],
                    ):
                        covering.add(sid)
                        break
    return covering


def _classify_testside(
    state: AnalyzerState, path: str, ride_along: frozenset[str] | set[str] = frozenset()
) -> Claim | None:
    """A tests/benchmarks-side file the import graph is blind to (an
    unimported .sh/.yaml/data file, an added __init__). Route by step-target
    coverage; an added test_*.py is exempt (a new uninvoked test keeps the
    run-all polarity of the added-test rule)."""
    if not path.startswith(("tests/", "benchmarks/")):
        return None
    if path.endswith(".py") and is_test_basename(path):
        return None
    covering = _step_target_coverage(state, path) | set(ride_along)
    if covering & state.auto_step_ids:
        return Claim(
            "target-coverage",
            f"{path} is a tests-side file outside the import graph; "
            f"{len(covering)} steps' targets cover its directory",
            step_ids=covering,
        )
    if covering:
        return Claim(
            "target-coverage",
            f"{path} is a tests-side file outside the import graph; covered "
            "only by manual-only steps; nothing to auto-run",
            step_ids=covering,
        )
    return Claim(
        "target-coverage",
        f"{path} is a tests-side file with no importer, no covering step "
        "target, and no step script/data reference; nothing to run",
    )


def _is_trivial_init(text: str) -> bool:
    """Empty or docstring-only after parsing (SPDX headers are comments)."""
    try:
        body = ast.parse(text).body
    except SyntaxError:
        return False
    if not body:
        return True
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def _auto_covered(state: AnalyzerState, test_files: set[str], script_files: set[str]):
    """(invoked tests, auto-run scripts) -- the polarity signal shared by the
    graph rule and the added/package rules below."""
    invoked = test_files & state.invoked
    auto_scripts = {
        f
        for f in script_files
        if f in state.auto_covered_files
        or (state.auto_prefixes and f.startswith(state.auto_prefixes))
    }
    return invoked, auto_scripts


def _classify_added_init(
    state: AnalyzerState, path: str, ctx: DiffContext
) -> Claim | None:
    """An added trivial __init__.py under vllm/: it only affects the
    importability of its package subtree, whose consumers are exactly the
    subtree's reverse closure. Non-trivial or unreadable-at-head -> fall
    through to the fail-open; a brand-new package (no base files) -> fall
    through (same-diff siblings carry their own claims)."""
    text = tablediff.git_show(state.repo, ctx.head, path)
    if text is None or not _is_trivial_init(text):
        return None
    pkg = path[: -len("__init__.py")]
    graph = state.full.graph
    base_files = [
        f for f in state.full.index.file_to_module if f.startswith(pkg) and f != path
    ]
    if not base_files:
        return None
    closure = graph.reverse_closure(set(base_files))
    test_files = {f for f in closure if is_test_file(f)}
    script_files = {f for f in closure if f.startswith(("examples/", "benchmarks/"))}
    invoked, auto_scripts = _auto_covered(state, test_files, script_files)
    if not invoked and not auto_scripts:
        return None
    return Claim(
        "added-trivial-init",
        f"{path} is a new trivial __init__ (empty/docstring-only at head); it "
        f"affects only its package subtree; routed to the subtree's reverse "
        f"closure ({len(base_files)} base files)",
        test_files=test_files | script_files,
    )


def _classify_package_data(state: AnalyzerState, path: str) -> Claim | None:
    """A non-Python asset under vllm/ (a tuning/config json) the graph cannot
    reach: route to the reverse closure of its OWNING package (the nearest
    ancestor dir with .py files -- the loader lives there) plus an additive
    device-family floor parsed from the filename. Stops before the bare vllm
    root: a root-level owning set would be a dressed-up run-all."""
    if not path.startswith("vllm/") or path.endswith(".py"):
        return None
    index = state.full.index
    graph = state.full.graph
    d = path.rsplit("/", 1)[0]
    owning: list[str] = []
    while "/" in d:
        owning = [f for f in index.file_to_module if f.rsplit("/", 1)[0] == d]
        if owning:
            break
        d = d.rsplit("/", 1)[0]
    if not owning:
        return None
    test_files: set[str] = set()
    script_files: set[str] = set()
    for f in owning:
        closure = graph.reverse_closure({f})
        test_files |= _seam_gated_tests(state, f, closure)
        script_files |= {
            c for c in closure if c.startswith(("examples/", "benchmarks/"))
        }
    filename = path.rsplit("/", 1)[-1]
    family = hardware.family_of_filename(filename)
    device_scope = hardware.device_prefix_of_filename(filename)
    # An h200 tuning file is a dependency only for h200 jobs, so scope the
    # additive family floor to the exact device the filename names (loader
    # convention).
    fam_steps = (
        {
            s.step_id
            for p in state.pipelines
            for s in p.steps
            if hardware.step_in_family(s, family)
            and not (device_scope and hardware.device_scoped_out(s, device_scope))
        }
        if family
        else set()
    )
    invoked, auto_scripts = _auto_covered(state, test_files, script_files)
    if not invoked and not auto_scripts and not (fam_steps & state.auto_step_ids):
        return None
    detail = (
        f"{path}: non-Python asset owned by {d}/; routed to the owning "
        "modules' reverse-closure tests"
    )
    if family:
        detail += f"; {family} device family from filename adds {len(fam_steps)} steps"
    if device_scope:
        detail += f"; scoped to device {device_scope}"
    return Claim(
        "package-data",
        detail,
        test_files=test_files | script_files,
        step_ids=fam_steps,
        device_scope=device_scope,
    )


def _head_graph(state: AnalyzerState, ctx: DiffContext):
    """The FullGraph at the diff HEAD, or None on ANY failure (missing
    objects, worktree trouble): the caller then falls through to the fail-open
    chain. Lazy import -- worktree imports select."""
    try:
        from .worktree import full_graph_for

        return full_graph_for(state.repo, ctx.head)
    except Exception:
        return None


def _covers_auto_step(state: AnalyzerState, path: str, test_files: set[str]) -> bool:
    """True iff at least one AUTO-RUN, non-device-excluded base step's targets
    cover a member -- mirrors _apply_claim_to_pipeline so a head-closure claim
    that would map to zero live steps falls through to fail-open instead of
    silently selecting nothing."""
    for p in state.pipelines:
        for step in p.steps:
            if step.step_id not in state.auto_step_ids:
                continue
            st = p.targets.get(step.step_id)
            if hardware.device_excluded_for_path(
                path, step.device, step
            ) and not _directly_collects(st, path):
                continue
            if st is not None and _targets_cover(st, test_files):
                return True
    return False


def _classify_added_head_closure(
    state: AnalyzerState, path: str, ctx: DiffContext
) -> Claim | None:
    """An added vllm/ .py the base graph cannot see: consult a graph built at
    HEAD, where its importers (including lazy/registry wiring and same-diff
    tests) exist, and map the resulting tests/scripts onto BASE steps. Empty
    closure or a zero-step mapping -> fall through to fail-open (an added file
    nothing reaches at head may be dynamically loaded; claiming 'nothing to
    run' would be the catastrophic polarity)."""
    head_full = _head_graph(state, ctx)
    if head_full is None:
        return None
    closure = head_full.graph.reverse_closure({path})
    cover = {f for f in closure if is_test_file(f)} | {
        f for f in closure if f.startswith(("examples/", "benchmarks/"))
    }
    if not cover or not _covers_auto_step(state, path, cover):
        return None
    return Claim(
        "added-head-closure",
        f"{path} is new; routed via its HEAD reverse closure "
        f"({len(cover)} test/script dependents mapped onto base steps)",
        test_files=cover,
    )


# Only the authoritative-nothing rules are exempt from the declarer union: a
# doc cannot break a build, and dead hardware runs nothing. release-ci is NOT
# here -- a release-pipeline script can also carry a live auto test (the Docker
# Build Metadata step declares docker-build-metadata-args.sh and executes it),
# so a release-ci claim must still pick up its genuine declarers.
_DEP_UNION_EXEMPT = frozenset({"no-code", "no-hardware"})


def _apply_declarer_union(state: AnalyzerState, path: str, claim: Claim) -> Claim:
    """Union the steps that declare `path` in source_file_dependencies -- the
    generator's own trigger, additive to whatever rule fired. Doing it at this
    single seam is what makes it a structural guarantee that no classifier can
    under-select by forgetting declarers, so every claim must pass through here,
    including the ones built outside `_classify`. Idempotent. Skipped for
    run-all claims (already maximal) and for _DEP_UNION_EXEMPT, where analyzer
    policy deliberately overrides a raw declared-deps match."""
    if claim.run_all or claim.rule in _DEP_UNION_EXEMPT:
        return claim
    # On a graph-known file the import graph is authoritative, so drop declarers
    # that matched ONLY via a catch-all `vllm/`/`tests/` prefix (the CI config's
    # own over-declaration). Graph-blind files keep the full union.
    #
    # That "authoritative" is the whole justification, and an unmodeled dynamic
    # import is direct evidence against it: edges are missing and we cannot say
    # which. So while any such site is unclassified, the blanket declarers come
    # back -- the fallback the analyzer is subtracting is exactly the net that
    # covers a hole it cannot see. Bounded (those declarers, not everything) and
    # self-clearing (classify or pragma the site and the savings return).
    suspended = bool(state.preflight.unclassified_sites)
    if _graph_known(state, path) and not suspended:
        declarers = _source_dep_steps(state, path, specific_only=True)
        omitted = len(_source_dep_steps(state, path) - declarers)
    else:
        declarers = _source_dep_steps(state, path)
        omitted = 0
    added = declarers - claim.step_ids
    if added:
        claim.step_ids |= added
        claim.detail += f"; +{len(added)} steps declare it as a source dep"
        if suspended and _graph_known(state, path):
            claim.detail += (
                " (catch-all declarers restored: unmodeled dynamic import sites "
                "mean the graph may be missing edges)"
            )
    if omitted:
        claim.detail += f"; {omitted} catch-all-only declarers omitted"
    return claim


def _classify(state: AnalyzerState, path: str, ctx: DiffContext | None) -> Claim:
    return _apply_declarer_union(state, path, _classify_inner(state, path, ctx))


def _classify_inner(state: AnalyzerState, path: str, ctx: DiffContext | None) -> Claim:
    configs = [p.config for p in state.pipelines]
    policy_world = ctx is None or path not in ctx.world_config_only
    claim = classify_world(path, configs, policy_world=policy_world)
    if claim:
        # A hardware path can be world for its OWN pipeline only (csrc/rocm
        # runs all of vllm_rocm_ci) while vllm_ci's mirror jobs, which run
        # on the image built from it, would be silently skipped: add the
        # device family's steps alongside the world claim. Declaring steps
        # ride along for the same reason (cmake/hipify.py is world for
        # vllm_rocm_ci only, but vllm_ci's torch-abi audit declares cmake/).
        family = hardware.family_of_path(path)
        if family:
            claim.step_ids |= state.family_steps(family)
        claim.step_ids |= _source_dep_steps(state, path)
        return claim
    if path.startswith(".buildkite/"):
        return _classify_buildkite(state, path, configs)
    # A file exclusive to a family with ZERO live steps (tpu today) has
    # provably nothing to run; the tether is re-derived every build, so a
    # tpu device appearing in any job yaml re-enables selection. Any device
    # the taxonomy cannot map disables the rule entirely: family_steps()
    # would be silently incomplete (fail-closed is the one polarity this
    # rule must never have).
    family = hardware.exclusive_family_of_path(path)
    if (
        family
        and path not in state.exclusive_disabled
        and not state.preflight.unmapped_devices
        and not state.family_steps(family)
    ):
        return Claim(
            "no-hardware",
            f"{path} executes only on {family} hardware; no live CI step "
            f"runs on {family}; nothing to run",
        )
    claim = _classify_graph(state, path)
    if claim:
        return claim
    if is_no_code(path):
        return Claim("no-code", f"{path} routes to no Buildkite jobs")
    # An ADDED plain test file under an existing step's directory target will
    # run in exactly those steps, so route it there instead of fail-open.
    # Strictly test_*.py: a conftest.py or __init__.py affects DESCENDANT
    # directories, which a containment check points away from, so it is routed
    # by _step_target_coverage's subtree leg further down instead.
    if ctx is not None and ctx.status.get(path) == "A" and path.endswith(".py"):
        # A conftest changes the fixtures of tests beneath it, so it keeps
        # fail-open only when pre-existing test files live under its directory.
        if path.rsplit("/", 1)[-1] == "conftest.py" and path.startswith("tests/"):
            dir_prefix = path.rsplit("/", 1)[0] + "/"
            if not any(f.startswith(dir_prefix) for f in state.catalog):
                return Claim(
                    "added-conftest",
                    f"{path} is a new conftest in a directory with no "
                    "pre-existing tests; coverage carried by the diff's own "
                    "added files",
                )
        # Nothing pre-existing can import an added file in a registered
        # package, and the package's coverage routes by its registered names,
        # so the new file inherits that routing.
        if path.startswith("vllm/"):
            keys = state.keys.for_file(path)
            if keys:
                graph = state.full.graph
                test_files = {
                    tf
                    for tf, lits in graph.string_literals.items()
                    if is_test_basename(tf) and not keys.isdisjoint(lits)
                }
                return Claim(
                    "added-in-claimed-package",
                    f"{path} is new inside a registered package; routed by "
                    f"its keys {sorted(keys)[:3]}",
                    test_files=test_files,
                    step_ids=state.keys.steps_naming(keys),
                )
            if path.rsplit("/", 1)[-1] == "__init__.py":
                init_claim = _classify_added_init(state, path, ctx)
                if init_claim is not None:
                    return init_claim
        is_test = is_test_file(path)
        is_benchmark = path.startswith("benchmarks/")
        if is_test or is_benchmark:
            owning = {
                sid
                for p in state.pipelines
                for sid, st in p.targets.items()
                for t in st.targets
                if not t.path.endswith(".py")
                and path.startswith(t.path.rstrip("/") + "/")
            }
            if owning:
                return Claim(
                    "added-test",
                    f"{path} is new and falls under existing steps' directory targets",
                    step_ids=owning,
                )
            if is_benchmark:
                # Nothing pre-existing can import a file that did not exist at
                # base, so a new benchmark no step invokes has nothing to run.
                return Claim(
                    "added-benchmark",
                    f"{path} is a new standalone benchmark no CI job invokes",
                )
        # Last resort for an added vllm/ file the earlier sub-rules missed
        # (not keyed, not a trivial init): reaching here means for_file was
        # empty, so no key routing is duplicated.
        if path.startswith("vllm/"):
            head_claim = _classify_added_head_closure(state, path, ctx)
            if head_claim is not None:
                return head_claim
    # A renamed/copied-into-existence path is unknown to the base graph, but
    # its content is the old path's (git -M pairs only >=50%-similar files;
    # heavier rewrites arrive as A+D and take the added-file rules). Classify
    # the OLD path (which exists at base) with ctx=None -- no added/rename rule
    # can re-fire, so no recursion even on a rename cycle -- and rebrand.
    # Head-side consumers of any genuinely new symbols are themselves M/A in
    # the diff with their own claims, so the base closure suffices.
    if ctx is not None and ctx.status.get(path) in ("R", "C"):
        old = ctx.renames.get(path)
        if old:
            sub = _classify(state, old, None)
            verb = "renamed" if ctx.status[path] == "R" else "copied"
            claim = Claim(
                "renamed",
                f"{path} {verb} from {old}; routed via its base closure "
                f"({sub.rule}: {sub.detail})",
                run_all=set(sub.run_all),
                step_ids=set(sub.step_ids),
                test_files=set(sub.test_files),
                divergent=set(sub.divergent),
            )
            family = hardware.family_of_path(path)
            if family:
                claim.step_ids |= state.family_steps(family)
            return claim
    if path.startswith("requirements/"):
        req = _classify_requirements(state, path)
        if req is not None:
            return req
    # A non-.buildkite release-pipeline script (tools/vllm-rocm/...) must be
    # zeroed BEFORE the scoped-exclusive fail-open, which would otherwise
    # swallow its rocm basename into the amd device family. A live-step
    # declarer disables this (it rejoins the test pipelines).
    if path in state.release_refs and not (
        _source_dep_steps(state, path) & state.auto_step_ids
    ):
        return Claim(
            "release-ci",
            f"{path} is referenced only by the release pipeline; no live "
            "test-pipeline jobs; nothing to run",
        )
    family = hardware.exclusive_family_of_path(path)
    if family and path not in state.exclusive_disabled:
        # A family-exclusive path OUTSIDE the package roots may carry the
        # generator's own routing (a blanket csrc/ source_file_dependencies):
        # consult the declarers first -- real CI runs exactly those plus the
        # family floor. The bare complement otherwise keeps every device=None
        # GPU suite and every unmapped-device step (rust cargo), none of which
        # run the file. Inside the roots the blanket vllm/ declarers would
        # swallow it (roots gate inside _classify_declared_deps); no auto
        # declarer keeps the complement (polarity). Rejected alternative:
        # mapping the 'cpu-medium' device to cpu in the FAMILY_DEVICE tables --
        # family_steps("cpu") would then pull the rust cargo steps into EVERY
        # cpu family floor.
        declared = _classify_declared_deps(state, path)
        if declared is not None:
            return declared
        # Unclaimed file in a hardware-exclusive namespace: fail open to its
        # device family; other devices provably cannot execute it.
        step_ids = {
            s.step_id
            for p in state.pipelines
            for s in p.steps
            if not hardware.device_excluded_for_path(path, s.device, s)
            or _directly_collects(p.targets.get(s.step_id), path)
        }
        return Claim(
            "fail-open",
            f"{path} is unclaimed; running its device family (scoped fail-open)",
            step_ids=step_ids,
        )
    testside = _classify_testside(state, path)
    if testside is not None:
        return testside
    pkg_data = _classify_package_data(state, path)
    if pkg_data is not None:
        return pkg_data
    declared = _classify_declared_deps(state, path)
    if declared is not None:
        return declared
    src = state.docker_inputs.get(path)
    detail = (
        f"{path} is a docker-image build input ({src} COPY); the CI image is "
        "rebuilt from it; running everything"
        if src
        else f"{path} is unclaimed by any rule; running everything"
    )
    return Claim("fail-open", detail, run_all={c.name for c in configs})


def _classify_buildkite(
    state: AnalyzerState, path: str, configs: list[PipelineConfig]
) -> Claim:
    """Ordered: live consumers first (config, step source, referenced
    script/data), THEN the legacy/inert zero-claims -- so a legacy or inert
    file that ever rejoins the live pipelines is claimed by its steps, not
    silenced -- then no-code, then the catch-all run-all."""
    for config in configs:
        if path == config.config_file:
            return Claim(
                "buildkite",
                f"{path} is {config.name}'s generator config",
                run_all={config.name},
            )
    step_ids = {
        s.step_id for p in state.pipelines for s in p.steps if s.source_file == path
    }
    if step_ids:
        return Claim("buildkite", f"{path} defines these steps", step_ids=step_ids)
    referencing = {
        sid
        for p in state.pipelines
        for sid, st in p.targets.items()
        if path in st.scripts_seen or path in st.data_files
    }
    if referencing:
        return Claim(
            "buildkite",
            f"{path} is used by these steps' commands",
            step_ids=referencing,
        )
    if path in LEGACY_CI_FILES:
        return Claim(
            "legacy-ci",
            f"{path} feeds only the retired external AMD pipeline; "
            "no live-pipeline jobs",
        )
    if path.startswith(INERT_CI_PREFIXES):
        return Claim(
            "inert-ci",
            f"{path} is in a CI tree no live pipeline consumes "
            "(external nightly/deprecated stub); nothing to run",
        )
    if path in RELEASE_PIPELINE_FILES or path in state.release_refs:
        detail = (
            f"{path} drives only the release/nightly publish pipeline"
            if path in RELEASE_PIPELINE_FILES
            else f"{path} is referenced only by the release pipeline"
        )
        return Claim("release-ci", f"{detail}; no live test-pipeline jobs")
    if is_no_code(path):
        return Claim("no-code", f"{path} routes to no Buildkite jobs")
    return Claim(
        "buildkite",
        f"{path} is unrecognized CI infra; running everything",
        run_all={c.name for c in configs},
    )


def _classify_graph(state: AnalyzerState, path: str) -> Claim | None:
    """The graph rule. Composed of four independent lookups so
    each stays auditable: direct step references, the (seam-gated) test
    closure, registered-key routing, and hardware-convention tagging."""
    if path in state.preflight.parse_error_paths:
        return Claim(
            "fail-open",
            f"{path} failed to parse; its edges are unknowable, running everything",
            run_all={p.config.name for p in state.pipelines},
        )
    if path in state.preflight.unclassified_sites:
        return Claim(
            "fail-open",
            f"{path} holds an unmodeled dynamic import; what it loads is "
            "unknowable, running everything",
            run_all={p.config.name for p in state.pipelines},
        )
    direct_steps = _direct_step_refs(state, path)
    graph = state.full.graph
    known = _graph_known(state, path)
    if not known and not direct_steps:
        return None

    closure = graph.reverse_closure({path})
    test_files = _seam_gated_tests(state, path, closure)
    # Steps run examples/benchmarks SCRIPTS as their test bodies too; a
    # closure member there counts as coverage.
    script_files = {f for f in closure if f.startswith(("examples/", "benchmarks/"))}
    keys, key_steps = _key_routed_steps(state, path, closure)

    # Polarity counts only coverage an AUTO-RUN step actually executes: a
    # file whose entire coverage is orphaned tests, or reachable only via
    # optional steps, must not silently under-run.
    invoked_tests = test_files & state.invoked
    auto_scripts = {
        f
        for f in script_files
        if f in state.auto_covered_files
        or (state.auto_prefixes and f.startswith(state.auto_prefixes))
    }
    if (
        not invoked_tests
        and not auto_scripts
        and not (direct_steps & state.auto_step_ids)
        and not (key_steps & state.auto_step_ids)
    ):
        claim = _zero_auto_coverage(state, path, test_files, direct_steps | key_steps)
        if claim is not None:
            return claim

    detail = f"{path} reaches {len(invoked_tests)} invoked test files"
    if key_steps:
        detail += (
            f"; registered key(s) {sorted(keys)[:3]} name it in {len(key_steps)} steps"
        )
    # Hardware-convention tagging exists for a SOURCE file whose compiled
    # kernels invisibly affect a family's jobs. A leaf-consumer file (a test,
    # benchmark, or example) has no such invisible reach -- nothing under vllm/
    # imports tests/, so its executing steps are exactly its target/script/data
    # coverage. A cpu-named test cannot affect an AMD job that never runs it.
    family = hardware.family_of_path(path)
    if family and not path.startswith(("tests/", "benchmarks/", "examples/")):
        hw_steps = state.family_steps(family)
        key_steps = key_steps | hw_steps
        detail += f"; {family} hardware-convention tagging adds {len(hw_steps)} steps"
    # A step that declares this path in source_file_dependencies via a SPECIFIC
    # (non-catch-all) prefix runs on its change no matter what the import graph
    # says -- additive coverage the closure alone misses once a member's closure
    # is deflated below its declarers (a demoted quant file still triggers the
    # evals that declare vllm/model_executor/layers/quantization). A bare
    # `vllm/` declarer is omitted here: the graph is authoritative on a
    # graph-known file (the wrapper reports the omission).
    dep_steps = _source_dep_steps(state, path, specific_only=True)
    if dep_steps:
        detail += f"; {len(dep_steps)} steps declare it as a source dep"
    return Claim(
        "graph",
        detail,
        test_files=test_files | script_files,
        step_ids=direct_steps | key_steps | dep_steps,
    )


def _zero_auto_coverage(
    state: AnalyzerState,
    path: str,
    test_files: set[str],
    manual_steps: set[str],
) -> Claim | None:
    """Dispatch for a graph-known file none of whose coverage auto-runs.
    None = fall through to the ordinary (empty) graph claim, which today
    covers only benchmarks/ and examples/ files. manual_steps (direct or
    key-routed steps, all manual-only here by construction) ride on every
    zero-claim so the manual_hits rendering contract survives."""
    run_all = {p.config.name for p in state.pipelines}
    root = state.full.index.installable_roots.get(path)
    if root:
        # Entry-point-loaded package: no import can reach it, but the steps
        # that pip-install it name its directory in their command text
        # (possibly relative to the tests/ working dir).
        needles = {root, root.removeprefix("tests/"), root.rsplit("/", 1)[-1]}
        steps = {
            sid
            for sid, text in state.keys.searchable.items()
            if any(n in text for n in needles)
        }
        if steps & state.auto_step_ids:
            return Claim(
                "graph",
                f"{path} is in installable package {root}; routed to steps "
                "whose commands reference it",
                step_ids=steps | manual_steps,
            )
        return Claim(
            "fail-open",
            f"{path} is in installable package {root} but no auto-run step "
            "references it; running everything",
            run_all=run_all,
        )
    if path.startswith("vllm/"):
        return Claim(
            "fail-open",
            f"{path} is in the graph but reaches zero auto-run coverage "
            "(zero-closure polarity); running everything",
            run_all=run_all,
        )
    if path.startswith("tests/"):
        # ONLY-legacy coverage (subset, not intersection: mixed coverage
        # means live tests exist somewhere and helpers must fail open).
        if test_files and test_files <= state.legacy_invoked:
            return Claim(
                "graph",
                f"{path}'s only coverage is legacy test-amd.yaml suites; "
                "no live-pipeline jobs",
                test_files=test_files,
                step_ids=manual_steps,
            )
        if is_test_basename(path):
            return Claim(
                "graph",
                f"{path} is invoked by no live step (orphan, see the "
                "uninvoked report); nothing to run",
                test_files=test_files,
                step_ids=manual_steps,
            )
        # A non-.py tests-side helper (an unimported .sh/.yaml/data file) whose
        # only graph presence is asset edges with no test closure: route by
        # step-target coverage instead of run-all. The `not test_files` guard
        # keeps a helper WITH a real (asset-edge) test closure on the fail-open.
        if not path.endswith(".py") and not test_files:
            claim = _classify_testside(state, path, ride_along=manual_steps)
            if claim is not None:
                return claim
        return Claim(
            "fail-open",
            f"{path} is a tests/ helper with zero auto-run coverage; "
            "running everything",
            run_all=run_all,
        )
    return None


def _classify_table(
    state: AnalyzerState, path: str, ctx: DiffContext
) -> tuple[Claim | None, set[str]]:
    """Diff a parsed table file between base and head; scope the claim to
    the changed entries. (None, set()) on any failed precondition -> the
    caller uses ordinary file-level classification."""
    base_text = tablediff.git_show(state.repo, ctx.base, path)
    head_text = tablediff.git_show(state.repo, ctx.head, path)
    if base_text is None or head_text is None:
        return None, set()
    diff = tablediff.diff_table(path, base_text, head_text)
    if diff is None:
        return None, set()

    graph = state.full.graph
    claim = Claim("table-diff", "")
    # Any table change at all affects the tests that iterate or import the
    # registries wholesale (sub-dict reshuffles included).
    if diff.texts_differ:
        for reg_file in tablediff.TABLE_FILES:
            claim.test_files |= {
                f for f in graph.reverse.get(reg_file, ()) if is_test_basename(f)
            }

    covered_added: set[str] = set()
    added_paths = {p for p, s in ctx.status.items() if s == "A"}
    for change in diff.changes:
        keys = {change.key}
        for parse in (diff.base, diff.head):
            keys |= parse.ids.get(change.key, set())
        keys |= state.full.registry.hf_ids.get(change.key, set())
        # Leaf tests naming the arch or any of its ids (base side).
        for test_file, literals in graph.string_literals.items():
            if is_test_basename(test_file) and not keys.isdisjoint(literals):
                claim.test_files.add(test_file)
        claim.step_ids |= state.keys.steps_naming(keys)
        claim.step_ids |= state.keys.steps_naming_raw(keys)
        if change.kind != "models":
            continue
        if change.change in ("removed", "modified"):
            mod = diff.base.modules.get(change.key)
            module_file = (
                state.full.index.resolve(resolve_module_name(mod)) if mod else None
            )
            if module_file:
                sub = _classify_graph(state, module_file)
                if sub is not None:
                    claim.test_files |= sub.test_files
                    claim.step_ids |= sub.step_ids
                    claim.run_all |= sub.run_all  # zero-closure propagates
        if change.change in ("added", "modified"):
            mod = diff.head.modules.get(change.key)
            if mod:
                covered_added |= _added_module_paths(mod, added_paths)

    summary = (
        ", ".join(f"{c.change} {c.key}" for c in diff.changes[:4])
        or "entries reshuffled"
    )
    claim.detail = (
        f"{path} table diff ({len(diff.changes)} entry changes: {summary}); "
        f"scoped to changed entries, all-arch tests always included"
    )
    return claim, covered_added


def _added_module_paths(mod: str, added_paths: set[str]) -> set[str]:
    """Diff-added files that a head-side table entry claims. Requires status
    A exactly (renames route via the rename-pairing rule). For a new
    vllm.models.* package, added files under the package dir are claimed too:
    nothing pre-existing can import a package that did not exist at base."""
    qualname = resolve_module_name(mod)
    base_path = qualname.replace(".", "/")
    candidates = {
        p for p in (f"{base_path}.py", f"{base_path}/__init__.py") if p in added_paths
    }
    if mod.startswith("vllm."):
        candidates |= {p for p in added_paths if p.startswith(base_path + "/")}
    return candidates


def _direct_step_refs(state: AnalyzerState, path: str) -> set[str]:
    """Steps that reference the file itself: as a target, a recursed script,
    or a data/config file."""
    return {
        sid
        for p in state.pipelines
        for sid, st in p.targets.items()
        if path in st.data_files
        or path in st.scripts_seen
        or any(t.path == path for t in st.targets)
    }


def _seam_gated_tests(state: AnalyzerState, path: str, closure: set[str]) -> set[str]:
    """Test files in the closure, with the worker-seam gate applied: tests
    reached ONLY by crossing a gated platform qualname edge (worker_cls
    dispatch) depend on the file only when they boot an engine."""

    def _tests(files: set[str]) -> set[str]:
        return {f for f in files if is_test_file(f)}

    graph = state.full.graph
    test_files = _tests(closure)
    if not graph.gated_edges or not state.preflight.seam_gate_ok:
        return test_files
    base_closure = graph.reverse_closure({path}, include_gated=False)
    seam_only = test_files - _tests(base_closure)
    if not seam_only:
        return test_files
    return _tests(base_closure) | (seam_only & state.full.engine_starting_tests())


def _key_routed_steps(
    state: AnalyzerState, path: str, closure: set[str]
) -> tuple[set[str], set[str]]:
    """Registered-key routing: if this file (or anything in its closure) is
    a string-registered module, steps naming its key depend on it (the nixl
    e2e jobs select the connector purely by name)."""
    keys: set[str] = set()
    for member in {path} | closure:
        keys |= state.keys.for_file(member)
    return keys, state.keys.steps_naming(keys)


def _apply_claim_to_pipeline(
    state: AnalyzerState,
    sel: Selection,
    claim: Claim,
    pdata: PipelineData,
    path: str,
) -> None:
    for step in pdata.steps:
        # Exclusivity answers "can this device execute the file", which is a
        # claim about inferred reach. A step whose own command collects the
        # file imports it regardless -- a rocm-named test in a directory a CUDA
        # job pytest-collects still fails that job on a bad import -- so direct
        # collection disarms the subtraction.
        if (
            path not in state.exclusive_disabled
            and hardware.device_excluded_for_path(path, step.device, step)
            and not _directly_collects(pdata.targets.get(step.step_id), path)
        ):
            continue
        if step.step_id in claim.step_ids:
            _record(sel, step, f"{path}: {claim.detail}", claim.rule)
            continue
        if not claim.test_files:
            continue
        # A device-named data file's graph-closure coverage is device-agnostic
        # pytest, but a step on a different known device loads its OWN config,
        # not this file -- so scope the owning-closure routing to the file's
        # device. step_ids (declarers + the pre-scoped floor) is left alone: a
        # declared dep is the generator's own trigger and must run.
        if claim.device_scope and hardware.device_scoped_out(step, claim.device_scope):
            continue
        st = pdata.targets.get(step.step_id)
        if st is None:
            continue
        hit = _targets_cover(st, claim.test_files)
        if hit:
            _record(sel, step, f"{path} -> {hit} -> {step.label}", claim.rule)


def _targets_cover(st: StepTargets, test_files: set[str]) -> str | None:
    for t in st.targets:
        if t.path.endswith(".py"):
            if t.path in test_files:
                return t.path
        else:
            prefix = t.path.rstrip("/") + "/"
            for f in test_files:
                if f.startswith(prefix):
                    return f
    return None


def _directly_collects(st: StepTargets | None, path: str) -> bool:
    """True when this step's OWN command loads `path`: a named target, a file
    under a directory target, a recursed script, or a data/config argument.

    The directory leg is what _direct_step_refs lacks (it matches exact paths
    only, and widening it would add steps on every claim). Keep them separate:
    this predicate exists to disarm a subtraction, not to add coverage.

    --ignore/--deselect are honoured here even though _targets_cover ignores
    them, because the two run in opposite directions: over-claiming coverage
    there only over-selects, while over-claiming it HERE keeps a step the
    hardware rule was right to drop. `pytest kernels/ --ignore=kernels/attention`
    never imports kernels/attention, so it is not proof of anything."""
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


def _record(sel: Selection, step: Step, reason: str, rule: str) -> None:
    if rule not in OUTPUT_RULES:
        raise ValueError(f"unpinned selection rule {rule!r}; add it to policy.RULES")
    manual = step.manual_only
    bucket = sel.manual_hits if manual else sel.selected
    rules = sel.manual_rules if manual else sel.selected_rules
    bucket.setdefault(step.step_id, []).append(reason)
    rules.setdefault(step.step_id, []).append(rule)


def _apply_run_all(state: AnalyzerState, sel: Selection) -> None:
    for pdata in state.pipelines:
        reason = sel.run_all.get(pdata.config.name)
        if not reason:
            continue
        for step in pdata.steps:
            _record(sel, step, reason, "run-all")


def _add_always_run(state: AnalyzerState, sel: Selection) -> None:
    for pdata in state.pipelines:
        for step in pdata.steps:
            if step.always_runs:
                # Deliberately not _record: an always-run step is never manual.
                sel.selected.setdefault(step.step_id, []).append(
                    "always-run key shortcut (image-build*/AMD base)"
                )
                sel.selected_rules.setdefault(step.step_id, []).append("always-run")
