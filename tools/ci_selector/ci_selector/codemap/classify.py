# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The rules: a changed path -> the Claim that says which steps it needs.

Reads `state.py` for what is in the tree; `selection.py` writes the answer.
Every claim carries its rule name, pinned in claim.RULES, because narrowing
downstream needs to know which rule selected a step.

THE ONE HOME OF RULE ORDER. A docs-only diff short-circuits first, then table
claims, then per file the first matching claim wins:

  image-input -> world -> buildkite chain -> no-hardware -> graph -> no-code -> status-A
  (graph answers as colocated-tests inside the import cycle, and outside it for
  a closure that has gone hub-like)
  added-file family -> renamed-or-copied -> rust -> requirements-file ->
  release-ci -> exclusive-family scoped fail-open -> target-coverage ->
  package-data -> native-tests -> declared-deps -> docker image-union
  deferral -> build-map-scoped fail-open -> inert floor -> terminal
  fail-open run-all.

Then `unions.py` adds what every path owes, then preflight escalations.

State is built at the diff BASE. At head the added files are already in the
graph, so the status-A rules would never fire.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from ..gitdiff import diff_files
from ..handwritten import (
    INERT_CI_PREFIXES,
    LEGACY_CI_FILES,
    PACKAGE_ROOTS,
    PR_PIPELINE,
    RELEASE_PIPELINE_FILES,
    REQUIREMENTS_BUILD_VALIDATED,
    RUST_GATE_ENV_VARS,
    RUST_PYO3_BRIDGE_FILE,
    RUST_TOOLCHAIN_FILES,
)
from . import build_map, colocation, hardware, native_ops, registry_diff, step_refs
from .claim import (
    Claim,
    classify_world,
    docs_only,
    is_no_code,
    matches_source_dependency,
)
from .externals import DOCKER_DIR
from .graph.model_registry import resolve_module_name
from .pipeline.buildkite import CI_DIR
from .pipeline.step import PipelineConfig
from .pipeline.targets import working_dir_to_repo_rel
from .repo import is_test_basename, is_test_file
from .selection import (
    Selection,
    _add_always_run,
    _apply_claim_to_pipeline,
    _apply_preflight,
    _apply_run_all,
    _directly_collects,
    _targets_cover,
)
from .state import (
    DiffContext,
    RepoState,
    _graph_known,
)
from .step_refs import (
    _direct_step_refs,
    _hardware_family_steps,
    _source_dep_steps,
    _source_dep_steps_ungated,
)
from .unions import (
    _apply_declarer_union,
    _apply_image_input_union,
    _build_map_allowed,
)
from .worktree import full_graph_for

PRECOMMIT_CONFIG = ".pre-commit-config.yaml"


def select(
    state: RepoState,
    paths: list[str],
    *,
    base: str | None = None,
    head: str | None = None,
) -> Selection:
    # Read once and discarded, so a typo in any CI_SELECTOR_* switch kills the
    # run instead of changing behaviour on whichever diff first hits a rule.
    colocation.mode()
    build_map.mode()
    native_ops.mode()
    step_refs.mode()
    sel = Selection()
    sel.docs_affected, sel.docs_reasons = state.docs_deps.docs_affected(paths)
    if docs_only(paths):
        sel.docs_only = True
        sel.notes.append("docs-only diff: generator emits no steps at all")
        return sel
    ctx = _diff_context(state, base, head)
    # Table files go first: their claims can take over the routing of files the
    # same diff ADDED, so a new model file rides on its new registry entry
    # instead of failing open.
    table_claims: dict[str, Claim] = {}
    covered_added: dict[str, str] = {}  # added path -> table path covering it
    if ctx is not None:
        for path in dict.fromkeys(paths):
            if path in registry_diff.TABLE_FILES:
                claim, added, gave_up = _classify_table(state, path, ctx)
                if claim is not None:
                    table_claims[path] = _apply_declarer_union(state, path, claim)
                    for a in added:
                        covered_added[a] = path
                else:
                    sel.notes.append(
                        f"{path}: no entry-level diff, {gave_up}. Routed by the "
                        "ordinary file rules instead, which reach more steps, "
                        "and any file this diff added that the table would have "
                        "carried falls back to its own rule."
                    )
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
            if pipeline not in sel.run_all:
                sel.run_all[pipeline] = f"{claim.rule}: {claim.detail}"
                sel.run_all_paths[pipeline] = path
        # A table claim covers the files it carries as much as the table.
        # evidence_paths replaces the path rather than joining it: the path
        # itself is never recorded, so keeping it would keep every step.
        carried = {a for a, t in covered_added.items() if t == path}
        weighed = set(claim.evidence_paths) or ({path} | carried)
        for pdata in state.pipelines:
            _apply_claim_to_pipeline(state, sel, claim, pdata, path, weighed)
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


def _diff_context(
    state: RepoState, base: str | None, head: str | None
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
        # Only a rename source vanishes. A copy source still exists at head, so
        # marking it deleted would be a quiet lie.
        if f.old_path and f.status == "R":
            status.setdefault(f.old_path, "D")
    return DiffContext(base=base, head=head, status=status, renames=renames)


def _classify_rust(state: RepoState, path: str) -> Claim:
    """rust/ routes by which shipped artifact the file's crate feeds.

    Three legs replace the docker-image widening, which this rule opts out of
    because that widening borrows whole-context images and balloons the answer.
    The legs are steps opting into the binary via the gate env vars; steps
    naming a workspace path in their own commands or scripts; steps on images
    that COPY the file in by name; and, for files reaching the PyO3 cdylib,
    the bridge file's whole claim, since that parser is imported with no env
    gate. Never droppable: a coverage row names Python functions and a .rs
    change is invisible to it.
    """
    ws = state.rust_workspace
    bucket = ws.bucket_of(path) or "root"
    gate_steps = state.keys.steps_naming_raw(set(RUST_GATE_ENV_VARS))
    gate_steps |= state.keys.steps_naming_raw({"rust/", *RUST_TOOLCHAIN_FILES})
    gate_steps &= state.auto_step_ids
    image_steps: set[str] = set()
    for df in state.artifacts.explicit_images_of(path):
        if hardware.family_of_path(df):
            image_steps |= state.artifacts.self_builders.get(df, set())
            image_steps |= state.artifacts.producers_of.get(df, set())
    image_steps -= {s for s in image_steps if s.startswith(f"{PR_PIPELINE}:")}
    image_steps &= state.auto_step_ids
    claim = Claim(
        "rust",
        f"{path}: {bucket} crate bucket; {len(gate_steps)} gate-env steps + "
        f"{len(image_steps)} hardware-image steps",
        step_ids=gate_steps | image_steps,
    )
    if bucket in ("cdylib", "root"):
        bridge = _classify(state, RUST_PYO3_BRIDGE_FILE, None)
        claim.step_ids |= bridge.step_ids
        claim.test_files |= bridge.test_files
        claim.run_all |= bridge.run_all
        claim.detail += f"; + the PyO3 bridge claim ({RUST_PYO3_BRIDGE_FILE})"
    return claim


def _classify_requirements(state: RepoState, path: str) -> Claim | None:
    """A requirements file: route to the steps that declare it plus its device
    family's jobs, since the filename names the device. A shared file gets its
    breadth from the docker-image widening instead, because every platform's
    image installs it. No declarer and no family falls through to the fail-open.

    A build-validated file (lint, dev) exists for tooling no test imports, so
    its honest reach is the declarers plus the always-run builds and its claim
    opts out of that widening. The manual-only fall-through applies there too.
    """
    if path in REQUIREMENTS_BUILD_VALIDATED:
        # Ignores the switch: this rule picks steps from declarations by
        # design. Silencing it sends the family-less files to run-all and
        # leaves the rest with nothing.
        declarers = _source_dep_steps_ungated(state, path)
        if declarers & state.auto_step_ids:
            return Claim(
                "requirements",
                f"{path}: build-validated, declaring steps only",
                step_ids=declarers,
                image_union_exempt=True,
            )
        return None
    family = hardware.requirements_family_of_path(path)
    fam_steps = state.family_steps(family) if family else set()
    step_ids = _source_dep_steps_ungated(state, path) | fam_steps
    if step_ids & state.auto_step_ids:
        return Claim(
            "requirements",
            f"{path}: declaring steps + {family or 'no'} device family",
            step_ids=step_ids,
        )
    if step_ids:
        # Manual-only coverage would select nothing at all, so fall through.
        return None
    if family:
        return Claim(
            "requirements",
            f"{path} maps to {family}; that device runs only on an "
            "external/unmodeled pipeline; nothing to run",
        )
    return None


# Directories too shallow and too shared for "sits beside a test target" to
# mean anything. `.buildkite/` joins the package roots because scripts live
# directly in it next to every pipeline yaml, and co-location there would make
# a yaml a dependency of whatever step runs one. Its SUBdirectories are fine
# and are where the real ownership is.
_ROOT_PREFIXES = tuple(f"{root}/" for root in PACKAGE_ROOTS) + (CI_DIR + "/",)


def _classify_declared_deps(state: RepoState, path: str) -> Claim | None:
    """Last chance before the fail-open: route a file the graph cannot see at
    all through the steps that declare it, which is the generator's own
    mechanism and so reproduces what real CI runs.

    Fires only outside the indexed package roots. Inside them a graph-unknown
    file is an anomaly the empty-closure direction owns, and the blanket
    `vllm/` declarers would swallow every asset we cannot model. Needs an
    auto-run declarer, else it falls through to run-all.

    Known difference from the generator: rust markdown hits no-code first and
    runs nothing, because markdown cannot break a cargo build."""
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
        # Other CPU suites compile the same file in-step without declaring it,
        # and the family union covers them.
        step_ids |= state.family_steps(family)
        detail += f" + {family} device family"
    return Claim("declared-deps", detail, step_ids=step_ids)


def _classify_native_tests(state: RepoState, path: str) -> Claim | None:
    """A kernel file routed by its own ops instead of by a declaration: the
    steps running the tests that name its ops, plus direct references, plus
    its device family. Declines without derived evidence, so a file with no
    ops keeps the fall-through below.

    This reaches everything the declared lists reached, except where the build
    map proves the file cannot compile into that step's family."""
    # The switch means "the op parse is not to be trusted"; routing must stand
    # down with it, same as the drop pass, or selection rides disarmed
    # evidence with no declared fallback behind it.
    if not state.native_ops.owns(path) or native_ops.mode() != "on":
        return None
    tests = state.native_ops.test_files_for(path)
    core: set[str] = set(_direct_step_refs(state, path))
    for tf in tests:
        core |= _steps_targeting(state, tf)
    if not core & state.auto_step_ids:
        return None
    step_ids = set(core)
    detail = (
        f"{path}: routed by its op joints ({len(tests)} op-referencing test "
        f"files -> {len(core)} steps)"
    )
    family = hardware.family_of_path(path)
    if family:
        step_ids |= state.family_steps(family)
        detail += f" + {family} device family ({len(step_ids - core)} more steps)"
    return Claim("native-tests", detail, step_ids=step_ids)


def _workdir_affinity_steps(state: RepoState, path: str) -> set[str]:
    """Steps whose declared working_dir covers `path`, meaning the whole tree
    is that job's. The Examples step spells its commands relative to the
    examples tree, so an example it does not invoke is still its business.
    Package roots and the default tests workdir are excluded."""
    out: set[str] = set()
    for p in state.pipelines:
        for s in p.steps:
            cwd = working_dir_to_repo_rel(s.working_dir)
            if not cwd or cwd + "/" in _ROOT_PREFIXES:
                continue
            if path.startswith(cwd + "/"):
                out.add(s.step_id)
    return out


def _steps_targeting(state: RepoState, path: str, *, siblings: bool = True) -> set[str]:
    """Steps whose targets cover a tests-side file, five ways: a directory
    target holding it, a .py target sitting beside it, a direct data or script
    reference, workdir affinity, and for a conftest or __init__ any target
    under its directory, since those affect everything below them.

    siblings=False drops the beside-a-.py-target leg, which is a guess at
    relatedness, not a reference; the inert veto needs an actual reference."""
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
                if t.path == path:
                    # A step invoking the file itself.
                    covering.add(sid)
                    break
                if not t.path.endswith(".py"):
                    if path.startswith(t.path.rstrip("/") + "/"):
                        covering.add(sid)
                        break
                elif siblings and "/" in t.path:
                    # The file must sit in the test's own directory or one
                    # level up. A deep descendant of a shallow parent is not a
                    # dependency of a test that merely lives there.
                    parent = t.path.rsplit("/", 1)[0]
                    if parent + "/" not in _ROOT_PREFIXES and parent in (
                        file_dir,
                        file_dir.rsplit("/", 1)[0],
                    ):
                        covering.add(sid)
                        break
    return covering | _workdir_affinity_steps(state, path)


def _reached_by_nothing(state: RepoState, path: str) -> bool:
    """True when no derived surface reaches the file: no step targets it, no
    key routes it, no image COPYs it, no step declares it, and its path or
    module name is in no step's command text. No job can then execute it.

    The declarer check ignores the trust switch (a generator trigger must
    never be silenced) but stays specific: the bare `vllm/` catch-all would
    match every vllm file. The command-text needle is the path or dotted
    module name, not a bare basename, which carries no identity."""
    if _steps_targeting(state, path, siblings=False):
        return False
    if state.keys.for_file(path):
        return False
    if path in state.docker_inputs:
        return False
    if _source_dep_steps_ungated(state, path, specific_only=True):
        return False
    needles = {path}
    if path.endswith(".py"):
        needles.add(path[: -len(".py")].replace("/", "."))
    return not any(
        n in text for text in state.keys.searchable.values() for n in needles
    )


def _named_in_invoked_tests(state: RepoState, path: str) -> bool:
    """Whether an auto-run test names the file in a string literal. The
    searchable text excludes directory-target test literals, so a by-name
    loader in such a test is only visible here."""
    stem = path.rsplit("/", 1)[-1].removesuffix(".py")
    dotted = path.removesuffix(".py").replace("/", ".")
    lits = state.full.graph.string_literals
    return any(
        (found := lits.get(tf)) and (stem in found or dotted in found)
        for tf in state.invoked
    )


def _classify_testside(
    state: RepoState, path: str, ride_along: frozenset[str] | set[str] = frozenset()
) -> Claim | None:
    """A leaf-side file the import graph cannot see: an unimported script,
    yaml or data file, or an added __init__. Routed by step-target coverage. An
    added test file is exempt, keeping the added-test rule's direction.

    `examples/` is included but does NOT stop here, and that asymmetry is the
    point. For tests and benchmarks, no covering step IS the answer: those
    trees exist to be run by a step. The Examples step names each example as a
    file target with no directory target, so the directory leg can never fire
    for them. The step's working_dir covers that tree instead, and a path even
    that cannot reach falls through to the rules below rather than zeroing.
    """
    if not path.startswith(("tests/", "benchmarks/", "examples/")):
        return None
    if path.endswith(".py") and is_test_basename(path):
        return None
    covering = _steps_targeting(state, path) | set(ride_along)
    if covering & state.auto_step_ids:
        step_ids = set(covering)
        detail = (
            f"{path} is a tests-side file outside the import graph; "
            f"{len(covering)} steps' targets cover its directory"
        )
        if path.startswith("examples/"):
            # A device-named example keeps its family, as the declared route
            # gave it.
            family = hardware.family_of_path(path)
            if family:
                step_ids |= state.family_steps(family)
                detail += f" + {family} device family"
        return Claim("target-coverage", detail, step_ids=step_ids)
    if covering:
        return Claim(
            "target-coverage",
            f"{path} is a tests-side file outside the import graph; covered "
            "only by manual-only steps; nothing to auto-run",
            step_ids=covering,
        )
    if path.startswith("examples/"):
        return None  # non-terminal: let the rules below route it
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


def _auto_run_hits(state: RepoState, test_files: set[str], script_files: set[str]):
    """(invoked tests, auto-run scripts): the direction signal the graph rule
    and the added/package rules share."""
    invoked = test_files & state.invoked
    auto_scripts = {
        f
        for f in script_files
        if f in state.auto_run_files
        or (state.auto_prefixes and f.startswith(state.auto_prefixes))
    }
    return invoked, auto_scripts


def _classify_added_init(state: RepoState, path: str, ctx: DiffContext) -> Claim | None:
    """An added empty __init__.py under vllm/. It only affects whether its
    package subtree imports, so the consumers are that subtree's reverse
    closure. Anything non-trivial, unreadable at head, or in a brand-new
    package falls through to the fail-open."""
    text = registry_diff.git_show(state.repo, ctx.head, path)
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
    invoked, auto_scripts = _auto_run_hits(state, test_files, script_files)
    if not invoked and not auto_scripts:
        return None
    return Claim(
        "added-trivial-init",
        f"{path} is a new trivial __init__ (empty/docstring-only at head); it "
        f"affects only its package subtree; routed to the subtree's reverse "
        f"closure ({len(base_files)} base files)",
        test_files=test_files | script_files,
    )


def _classify_package_data(state: RepoState, path: str) -> Claim | None:
    """A non-Python asset under vllm/ the graph cannot reach. Routed to the
    reverse closure of its owning package, meaning the nearest parent directory
    with Python in it, where the loader lives, plus a device-family floor read
    off the filename. Stops before the bare vllm root, which would be a
    run-all in disguise."""
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
        test_files |= _boot_gated_tests(state, f, closure)
        script_files |= {
            c for c in closure if c.startswith(("examples/", "benchmarks/"))
        }
    filename = path.rsplit("/", 1)[-1]
    family = hardware.family_of_filename(filename)
    device_scope = hardware.device_prefix_of_filename(filename)
    # A tuning file for one device matters only to that device's jobs, so
    # scope the family floor to the device the filename names.
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
    invoked, auto_scripts = _auto_run_hits(state, test_files, script_files)
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


def _head_graph(state: RepoState, ctx: DiffContext):
    """The FullGraph at the diff head, or None on any failure, which sends the
    caller to the fail-open chain."""
    try:
        return full_graph_for(state.repo, ctx.head)
    except Exception:
        return None


def _covers_auto_step(state: RepoState, path: str, test_files: set[str]) -> bool:
    """True when some auto-run, non-excluded base step's targets cover a
    member. Mirrors _apply_claim_to_pipeline, so a head-closure claim mapping
    to no live step falls through instead of selecting nothing."""
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
    state: RepoState, path: str, ctx: DiffContext
) -> Claim | None:
    """An added vllm/ file the base graph cannot see. Consult a graph built at
    HEAD, where its importers and same-diff tests exist, and map the resulting
    tests onto base steps. An empty closure falls through to the fail-open: a
    file nothing reaches at head may be loaded dynamically, and "nothing to
    run" is the one answer we cannot risk."""
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


def _classify(state: RepoState, path: str, ctx: DiffContext | None) -> Claim:
    claim = _apply_declarer_union(state, path, _classify_inner(state, path, ctx))
    claim = _apply_image_input_union(state, path, claim)
    return _apply_csrc_droppability(state, path, claim)


def _apply_csrc_droppability(state: RepoState, path: str, claim: Claim) -> Claim:
    """Let the record drop a csrc file's steps on wrapper evidence.

    Selection is untouched. Runs after both unions, so the narrowed step set
    is what may become droppable. The bet is that a kernel is only reached
    through its Python wrappers, so anything unresolved keeps: no ops, an op
    with no wrapper, or a path this diff added all grant nothing.

    Kept regardless: steps declaring the file, steps that build an image, and
    steps whose tests name one of its ops, since a test may call the op from a
    frame the recorder cannot see.
    """
    if claim.run_all or not state.native_ops.owns(path) or native_ops.mode() != "on":
        return claim
    proxies = state.native_ops.proxies_for(path)
    if not proxies:
        return claim
    # Ignores the switch: this can only keep a step, never pick or narrow one,
    # and a step CI itself triggers from a declaration is real.
    held: set[str] = set(_source_dep_steps_ungated(state, path))
    held |= {s for ss in state.artifacts.producers_of.values() for s in ss}
    held |= {s for ss in state.artifacts.self_builders.values() for s in ss}
    for test_file in state.native_ops.test_files_for(path):
        held |= _steps_targeting(state, test_file)
    droppable = (claim.step_ids & state.auto_step_ids) - held
    if not droppable:
        return claim
    claim.droppable_step_ids |= droppable
    claim.evidence_paths = frozenset(proxies)
    claim.detail += (
        f"; {len(droppable)} steps droppable on op-wrapper evidence "
        f"({len(proxies)} wrapper files)"
    )
    return claim


def _classify_image_input(state: RepoState, path: str) -> Claim | None:
    """An image-definition file: route to the steps that run on that image.

    Derived end to end. A step is a producer because another step's
    `depends_on` names it, the Dockerfile comes out of the producer's own build
    command, and the consumers are everything downstream. Runs before the world
    rule, so a Dockerfile becomes the steps that consume it instead of an
    undifferentiated run-all.

    Never droppable: a coverage row says nothing about which image a step runs
    on. Returns None for most of the repo, because the COPY graph cannot tell
    "changes the build" from "is in the image" (see pipeline/images.py), so
    only image DEFINITION files route here.
    """
    steps = state.artifacts.steps_for(path)
    if not steps:
        return None
    if not steps & state.auto_step_ids:
        return None  # manual-only coverage would silently select nothing
    producers = sorted(state.artifacts.producers_of[path])
    return Claim(
        "image-input",
        f"{path} defines the image built by {producers}; routed to that "
        f"image's {len(steps)} dependent steps",
        step_ids=steps,
    )


def _classify_inner(state: RepoState, path: str, ctx: DiffContext | None) -> Claim:
    configs = [p.config for p in state.pipelines]
    image = _classify_image_input(state, path)
    if image is not None:
        return image
    claim = classify_world(path, configs)
    if claim:
        # The world claim escalates every pipeline on its own, so these two
        # add nothing today. Kept because the rule is an ordering device: a
        # member a narrower rule would claim must still carry its hardware
        # family and its declarers.
        family = hardware.family_of_path(path)
        if family:
            claim.step_ids |= state.family_steps(family)
        claim.step_ids |= _source_dep_steps(state, path)
        return claim
    if path.startswith(".buildkite/"):
        return _classify_buildkite(state, path, configs)
    # A file exclusive to a family with no live steps has nothing to run. The
    # link is re-derived every build, so that family appearing in any job yaml
    # switches selection back on. A device the taxonomy cannot map turns the
    # rule off entirely, since family_steps() would be quietly incomplete.
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
    lint = _classify_lint_only(state, path)
    if lint is not None:
        return lint
    # An added plain test file under an existing directory target runs in
    # exactly those steps. Test files only: a conftest or __init__ affects the
    # directories BELOW it, which a containment check points away from, so
    # those go through _steps_targeting's subtree leg further down.
    if ctx is not None and ctx.status.get(path) == "A" and path.endswith(".py"):
        # A conftest changes the fixtures of the tests beneath it, so it only
        # keeps the fail-open when such tests already exist.
        if path.rsplit("/", 1)[-1] == "conftest.py" and path.startswith("tests/"):
            dir_prefix = path.rsplit("/", 1)[0] + "/"
            if not any(f.startswith(dir_prefix) for f in state.catalog):
                return Claim(
                    "added-conftest",
                    f"{path} is a new conftest in a directory with no "
                    "pre-existing tests; coverage carried by the diff's own "
                    "added files",
                )
        # Nothing that already existed can import an added file, and the
        # package routes by its registered names, so the new file inherits it.
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
            owning = _steps_targeting(state, path)
            if owning:
                return Claim(
                    "added-test",
                    f"{path} is new and falls under existing steps' targets",
                    step_ids=owning,
                )
            if is_benchmark:
                # Nothing can import a file that did not exist at base, so a
                # new benchmark no step invokes has nothing to run.
                return Claim(
                    "added-benchmark",
                    f"{path} is a new standalone benchmark no CI job invokes",
                )
            # No step's command sweeps its directory, so no job can run it. An
            # added step yaml escalates its pipeline through the job_dirs guard.
            return Claim(
                "added-test",
                f"{path} is a new test in a directory no step sweeps; "
                "no job can run it yet",
            )
        # An added vllm/ file with no keys and no head closure reaching live
        # coverage has unknown reach and fails open. Inheriting the package
        # directory's closure was tried and over-selects (its files sit in the
        # import cycle); diff-side import parsing is the queued sound route.
        if path.startswith("vllm/"):
            head_claim = _classify_added_head_closure(state, path, ctx)
            if head_claim is not None:
                return head_claim
    # A renamed or copied path is unknown to the base graph, but its content is
    # the old path's, since git only pairs similar files and a heavier rewrite
    # arrives as an add plus a delete. So classify the OLD path, which exists at
    # base, with no ctx: no added or rename rule can fire again, which rules out
    # recursion even on a rename cycle.
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
                device_scope=sub.device_scope,
                droppable_step_ids=set(sub.droppable_step_ids),
                droppable_test_files=sub.droppable_test_files,
                image_union_exempt=sub.image_union_exempt,
            )
            family = hardware.family_of_path(path)
            if family:
                # Hardware tagging is not function-attributable, so these stay
                # out of `droppable_step_ids`.
                claim.step_ids |= state.family_steps(family)
            return claim
    if state.rust_workspace.owns(path):
        return _classify_rust(state, path)
    if path.startswith("requirements/"):
        req = _classify_requirements(state, path)
        if req is not None:
            return req
    # A release-pipeline script outside .buildkite has to be zeroed before the
    # scoped fail-open, which would otherwise read its name as a device family.
    # A live-step declarer turns this off, since the file rejoined the tests.
    if path in state.release_refs and not (
        # Ignores the switch: this only ever says "the file is still tested",
        # so silencing it would invent empty answers.
        _source_dep_steps_ungated(state, path) & state.auto_step_ids
    ):
        return Claim(
            "release-ci",
            f"{path} is referenced only by the release pipeline; no live "
            "test-pipeline jobs; nothing to run",
        )
    family = hardware.exclusive_family_of_path(path)
    if family and path not in state.exclusive_disabled:
        # A family-exclusive path outside the package roots may carry the
        # generator's own routing, so consult the declarers first: real CI runs
        # exactly those plus the family floor. Otherwise the complement keeps
        # every device-less GPU suite and every unmapped-device step, none of
        # which run the file. No auto declarer keeps the complement.
        declared = _classify_declared_deps(state, path)
        if declared is not None:
            return declared
        # Unclaimed file in a hardware-exclusive namespace: fail open to its
        # own device family, since no other device can run it.
        step_ids = {
            s.step_id
            for p in state.pipelines
            for s in p.steps
            if not hardware.device_excluded_for_path(path, s.device, s)
            or _directly_collects(p.targets.get(s.step_id), path)
        }
        detail = f"{path} is unclaimed; running its device family (scoped fail-open)"
        # The complement keeps every GPU suite carrying no device name, but a
        # source cannot break a suite whose build never compiles it. Steps that
        # invoke the file directly are kept either way.
        fams = state.build_map.families.get(path)
        if fams and not state.preflight.unmapped_devices and build_map.mode() == "on":
            direct = {
                sid
                for p in state.pipelines
                for sid, st in p.targets.items()
                if _directly_collects(st, path)
            }
            step_ids = (step_ids & _build_map_allowed(state, fams)) | direct
            detail += f"; build-map scoped to {sorted(fams)}"
        return Claim("fail-open", detail, step_ids=step_ids)
    testside = _classify_testside(state, path)
    if testside is not None:
        return testside
    pkg_data = _classify_package_data(state, path)
    if pkg_data is not None:
        return pkg_data
    native = _classify_native_tests(state, path)
    if native is not None:
        return native
    declared = _classify_declared_deps(state, path)
    if declared is not None:
        return declared
    # Last chance to ask the build graph before escalating: the image union
    # bails on a run_all claim, so escalating here would pre-empt an answer it
    # had. Scoped to `docker/` because almost every other path reaching this
    # line has an empty union anyway.
    if path.startswith(DOCKER_DIR + "/") and (
        state.artifacts.steps_for_input(path) & state.auto_step_ids
    ):
        return Claim(
            "fail-open",
            f"{path} is unclaimed by any rule, but the build DAG knows which "
            "images it is copied into; deferring to the image-input union",
        )
    # An unclaimed file the build map has families for cannot break a family
    # it never compiles into, so the complement is scoped the way the
    # exclusive-family branch scopes it. This drops the no-device steps
    # (torch-stable-abi-audit included) for cpu/amd-only files — the lane-2
    # containment measurement endorsed exactly that class. A step_ids claim
    # here also lets the coverage stage work on the rest of the diff, where
    # the old run_all vetoed every drop PR-wide.
    fams = state.build_map.families.get(path)
    if fams and not state.preflight.unmapped_devices and build_map.mode() == "on":
        direct = {
            sid
            for p in state.pipelines
            for sid, st in p.targets.items()
            if _directly_collects(st, path)
        }
        step_ids = (
            {s.step_id for p in state.pipelines for s in p.steps}
            & _build_map_allowed(state, fams)
        ) | direct
        if step_ids:
            return Claim(
                "fail-open",
                f"{path} is unclaimed by any rule; running the device "
                f"families that compile it (build-map scoped to {sorted(fams)})",
                step_ids=step_ids,
            )
    # A file no derived surface reaches: only the always-run image builds
    # touch it, so nothing beyond the floor can run it. Package roots are
    # excluded because an added or ctx-less vllm/tests path lands here too and
    # must keep the fail-open.
    if not path.startswith(_ROOT_PREFIXES) and _reached_by_nothing(state, path):
        return Claim(
            "inert",
            f"{path} is reached by no derived surface (step targets, keys, "
            "docker inputs, declarers, command text); nothing beyond the "
            "floor can execute it",
        )
    src = state.docker_inputs.get(path)
    detail = (
        f"{path} is a docker-image build input ({src} COPY); the CI image is "
        "rebuilt from it; running everything"
        if src
        else f"{path} is unclaimed by any rule; running everything"
    )
    return Claim("fail-open", detail, run_all={c.name for c in configs})


def _classify_buildkite(
    state: RepoState, path: str, configs: list[PipelineConfig]
) -> Claim:
    """Ordered: live consumers first, then the legacy and inert zero-claims, so
    a retired file that ever rejoins the live pipelines is claimed by its steps
    instead of being silenced. Then no-code, then the catch-all run-all."""
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
    # `_steps_targeting`, not a scripts_seen/data_files check: this leg used to
    # miss a file that is a step's pytest TARGET, reached through its
    # `working_dir`, and everything it missed fell to the terminal run-all. The
    # same helper covers the config yamls beside such a target.
    referencing = _steps_targeting(state, path)
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
    # An added step yaml (loaded ones matched source_file above) escalates
    # every pipeline whose job_dirs include it -- one dir can feed two.
    job_dir_configs = {
        config.name
        for config in configs
        if any(path.startswith(d.rstrip("/") + "/") for d in config.job_dirs)
    }
    if job_dir_configs:
        return Claim(
            "buildkite",
            f"{path} is under the job_dirs of {sorted(job_dir_configs)}; a "
            "step file the base did not load escalates the pipelines it joins",
            run_all=job_dir_configs,
        )
    # The generator's own escalation trigger for a pipeline. Read only to
    # escalate, never to route or drop, so the hand list stays out of both.
    # Before the docker floor: if the generator runs everything on this file,
    # the always-run builds are not its whole test.
    pattern_configs = {
        config.name
        for config in configs
        if any(matches_source_dependency(p, path) for p in config.run_all_patterns)
        and not any(
            matches_source_dependency(p, path) for p in config.run_all_exclude_patterns
        )
    }
    if pattern_configs:
        return Claim(
            "buildkite",
            f"{path} matches the run_all_patterns of {sorted(pattern_configs)}; "
            "the generator escalates those pipelines on it",
            run_all=pattern_configs,
        )
    if state.docker_inputs.get(path):
        return Claim(
            "inert",
            f"{path} is executed at image build "
            f"({state.docker_inputs[path]} COPY); the always-run builds "
            "are its test",
        )
    # A file directly under .buildkite may be generator input; only a
    # subdirectory file can rest at the floor.
    if "/" in path.removeprefix(CI_DIR + "/") and _reached_by_nothing(state, path):
        return Claim(
            "inert",
            f"{path} is reached by no derived surface and no generator "
            "pattern; nothing beyond the floor can execute it",
        )
    return Claim(
        "buildkite",
        f"{path} is unrecognized CI infra; running everything",
        run_all={c.name for c in configs},
    )


def colocation_routes(state: RepoState, path: str) -> bool:
    """Whether selection answers `path` with a colocated-tests claim.

    The one home of that question for callers outside the rule, so their floor
    and this rule cannot drift apart.
    """
    claim = _classify_graph(state, path)
    return claim is not None and claim.rule == "colocated-tests"


def _classify_graph(
    state: RepoState, path: str, *, inherit: bool = True
) -> Claim | None:
    """The graph rule: four separate lookups so each stays checkable. Direct
    step references, the test closure, registered-key routing, and hardware
    tagging by name. `inherit=False` blocks a second hop when
    _inherit_table_coverage recurses in."""
    if path in state.preflight.parse_error_paths:
        return Claim(
            "fail-open",
            f"{path} failed to parse; its edges are unknowable, running everything",
            run_all={p.config.name for p in state.pipelines},
        )
    # The only protection left for an unmodeled dynamic import, and it covers
    # one case: the site file itself changed. Files the site loads in secret
    # get ordinary selection and may under-run until it is classified, which
    # the check and the preflight warning both say loudly.
    #
    # Measured, so nobody rebuilds it: adding "every step that reaches the site
    # file" looks like the honest bound and is useless, because reverse
    # reachability here has collapsed. Nearly every file reaches nearly every
    # other, so that union is most of the pipeline for every site, worse than
    # the catch-all declarers it replaced. Narrowing needs knowing what the
    # site loads, which is a parser's job and not a graph walk's.
    if path in state.preflight.unclassified_sites:
        return Claim(
            "fail-open",
            f"{path} holds an unmodeled dynamic import; what it loads is "
            "unknowable, running everything",
            run_all={p.config.name for p in state.pipelines},
        )
    # After the two guards above, never before: both say the graph is known to
    # be incomplete here, and co-location would be trusting it anyway.
    colocated = colocation._classify_colocated_tests(state, path)
    if colocated is not None:
        return colocated
    direct_steps = _direct_step_refs(state, path)
    graph = state.full.graph
    known = _graph_known(state, path)
    if not known and not direct_steps:
        return None

    closure = graph.reverse_closure({path})
    test_files = _boot_gated_tests(state, path, closure)
    # Steps run example and benchmark scripts as test bodies too, so a closure
    # member there counts as coverage.
    script_files = {f for f in closure if f.startswith(("examples/", "benchmarks/"))}
    keys, key_steps = _key_routed_steps(state, path, closure)

    # Only coverage an auto-run step actually executes counts. A file whose
    # tests are all orphaned, or reachable only through optional steps, must
    # not quietly under-run.
    invoked_tests = test_files & state.invoked
    auto_scripts = {
        f
        for f in script_files
        if f in state.auto_run_files
        or (state.auto_prefixes and f.startswith(state.auto_prefixes))
    }
    if (
        not invoked_tests
        and not auto_scripts
        and not (direct_steps & state.auto_step_ids)
        and not (key_steps & state.auto_step_ids)
    ):
        if inherit:
            claim = _inherit_table_coverage(state, path, direct_steps | key_steps)
            if claim is not None:
                return claim
        claim = _nothing_auto_runs(state, path, test_files, direct_steps | key_steps)
        if claim is not None:
            return claim

    detail = f"{path} reaches {len(invoked_tests)} invoked test files"
    if key_steps:
        detail += (
            f"; registered key(s) {sorted(keys)[:3]} name it in {len(key_steps)} steps"
        )
    family, hw_steps = _hardware_family_steps(state, path)
    # Pinned before the hw union below, which is not droppable.
    inferred_steps = direct_steps | key_steps
    if family:
        key_steps = key_steps | hw_steps
        detail += f"; {family} hardware-convention tagging adds {len(hw_steps)} steps"
    # A step declaring this path through a specific prefix runs on its change
    # whatever the graph says. That is coverage the closure alone misses once a
    # file's closure shrinks below its declarers. A bare `vllm/` declarer is
    # left out, since the graph is the better answer on a file it knows.
    dep_steps = _source_dep_steps(state, path, specific_only=True)
    if dep_steps:
        detail += f"; {len(dep_steps)} steps declare it as a source dep"
    # An examples script no step invokes still belongs to its tree's job.
    # Not droppable, like the step it replaces.
    affinity = _workdir_affinity_steps(state, path)
    claim = Claim(
        "graph",
        detail,
        test_files=test_files | script_files,
        step_ids=direct_steps | key_steps | dep_steps | affinity,
        # hw_steps is subtracted, not just left out: it stands for compiled
        # reach nothing records, so a step it holds stays held.
        droppable_step_ids=(inferred_steps | dep_steps) - hw_steps - affinity,
        droppable_test_files=True,
    )
    # Here and not earlier, so _nothing_auto_runs claims (rule "graph" but
    # grep-built, not closure-built) can never reach the hub gate.
    return colocation._colocated_hub(state, path, claim) or claim


def _inherit_table_coverage(
    state: RepoState, path: str, manual_steps: set[str]
) -> Claim | None:
    """A registry-named file with no coverage of its own inherits the coverage
    of the table that names it.

    The registry parsers add edges only from tests to targets, so a member no
    test names by key has no in-edges and would run everything. What can load
    it is exactly the registry's own consumers, which the table's claim already
    carries. One hop: any table whose own coverage is empty or run-all
    disqualifies the inheritance, falling back to _nothing_auto_runs.
    """
    tables = state.full.table_of().get(path)
    if not tables:
        return None
    test_files: set[str] = set()
    step_ids: set[str] = set(manual_steps)
    droppable: set[str] = set()
    # A step one table holds non-droppably must stay held even if another
    # table's leg marks it droppable, or its hardware coverage could be dropped.
    held: set[str] = set(manual_steps)
    inherited: list[str] = []
    for table in sorted(tables):
        if table == path:
            continue
        sub = _classify_graph(state, table, inherit=False)
        if sub is None or sub.run_all:
            return None
        if not (sub.test_files & state.invoked) and not (
            sub.step_ids & state.auto_step_ids
        ):
            return None
        test_files |= sub.test_files
        step_ids |= sub.step_ids
        droppable |= sub.droppable_step_ids
        held |= sub.step_ids - sub.droppable_step_ids
        inherited.append(table)
    if not inherited:
        return None
    return Claim(
        "graph",
        f"{path} reaches zero auto-run coverage of its own; inheriting the "
        f"coverage of registry {', '.join(inherited)} that names it",
        test_files=test_files,
        step_ids=step_ids,
        droppable_step_ids=(droppable & step_ids) - held,
        droppable_test_files=True,
    )


def _nothing_auto_runs(
    state: RepoState,
    path: str,
    test_files: set[str],
    manual_steps: set[str],
) -> Claim | None:
    """For a file the graph knows but whose coverage never auto-runs. None
    falls through to the ordinary empty graph claim. manual_steps ride along on
    every zero-claim so the manual_hits rendering still works."""
    run_all = {p.config.name for p in state.pipelines}
    root = state.full.index.installable_roots.get(path)
    if root:
        # Loaded through an entry point, so no import reaches it, but the
        # steps that pip-install it name its directory in their commands.
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
        # Floor only when nothing at all names it, including invoked-test
        # literals. Any name-hit keeps the fail-open for the loader shapes the
        # graph cannot see.
        if _reached_by_nothing(state, path) and not _named_in_invoked_tests(
            state, path
        ):
            return Claim(
                "graph",
                f"{path} reaches zero auto-run coverage and nothing names it "
                "(steps, keys, declarers, invoked-test literals); nothing "
                "can run it",
                test_files=test_files,
                step_ids=manual_steps,
            )
        return Claim(
            "fail-open",
            f"{path} is in the graph but reaches zero auto-run coverage "
            "(empty-closure direction); running everything",
            run_all=run_all,
        )
    if path.startswith("tests/"):
        # Legacy coverage ONLY. A subset and not an intersection: mixed
        # coverage means live tests exist and the helper must fail open.
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
                f"{path} is invoked by no live step (orphan); nothing to run",
                test_files=test_files,
                step_ids=manual_steps,
            )
        # A non-Python helper reaching a test only through asset edges: route
        # by step-target coverage. One with a test closure carries it below.
        if not path.endswith(".py") and not test_files:
            claim = _classify_testside(state, path, ride_along=manual_steps)
            if claim is not None:
                return claim
        # No live job loads this helper; its optional coverage rides along as
        # manual hits and nothing auto-runs.
        return Claim(
            "graph",
            f"{path} is a tests/ helper with zero auto-run coverage; its "
            "coverage rides along as manual hits, nothing auto-runs it",
            test_files=test_files,
            step_ids=manual_steps,
        )
    return None


def _classify_table(
    state: RepoState, path: str, ctx: DiffContext
) -> tuple[Claim | None, set[str], str]:
    """Diff a parsed table between base and head and scope the claim to the
    changed entries.

    On failure the claim is None and the caller falls back to ordinary
    file-level classification, which is broader. The third element says what
    went wrong so the caller can report that, because a skipped table diff and
    a table with no changes otherwise read exactly the same.
    """
    base_text = registry_diff.git_show(state.repo, ctx.base, path)
    head_text = registry_diff.git_show(state.repo, ctx.head, path)
    if base_text is None or head_text is None:
        ref = ctx.base if base_text is None else ctx.head
        return None, set(), f"could not read it at {ref} (git show failed or timed out)"
    diff = registry_diff.diff_table(path, base_text, head_text)
    if diff is None:
        return None, set(), "the table did not parse"

    graph = state.full.graph
    claim = Claim("table-diff", "")
    # Any table change at all affects the tests that walk or import the whole
    # registry, reshuffles included.
    if diff.texts_differ:
        for reg_file in registry_diff.TABLE_FILES:
            claim.test_files |= {
                f for f in graph.reverse.get(reg_file, ()) if is_test_basename(f)
            }

    covered_added: set[str] = set()
    added_paths = {p for p, s in ctx.status.items() if s == "A"}
    held: set[str] = set()
    for change in diff.changes:
        keys = {change.key}
        for parse in (diff.base, diff.head):
            keys |= parse.ids.get(change.key, set())
        keys |= state.full.registry.hf_ids.get(change.key, set())
        # Base-side tests naming the arch or any of its ids.
        for test_file, literals in graph.string_literals.items():
            if is_test_basename(test_file) and not keys.isdisjoint(literals):
                claim.test_files.add(test_file)
        named = state.keys.steps_naming(keys) | state.keys.steps_naming_raw(keys)
        claim.step_ids |= named
        held |= named
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
                    claim.droppable_step_ids |= sub.droppable_step_ids
                    held |= sub.step_ids - sub.droppable_step_ids
                    claim.run_all |= sub.run_all  # empty-closure propagates
        if change.change in ("added", "modified"):
            mod = diff.head.modules.get(change.key)
            if mod:
                covered_added |= _added_module_paths(mod, added_paths)

    # A step another leg kept without droppability must not become droppable
    # through a sibling arch's closure.
    claim.droppable_step_ids -= held

    summary = (
        ", ".join(f"{c.change} {c.key}" for c in diff.changes[:4])
        or "entries reshuffled"
    )
    claim.detail = (
        f"{path} table diff ({len(diff.changes)} entry changes: {summary}); "
        f"scoped to changed entries, all-arch tests always included"
    )
    return claim, covered_added, ""


def _added_module_paths(mod: str, added_paths: set[str]) -> set[str]:
    """Files the diff added that a head-side table entry claims. Added only,
    since a rename goes through the rename rule. For a new model package the
    files under it are claimed too, because nothing that already existed can
    import a package that did not exist at base."""
    qualname = resolve_module_name(mod)
    base_path = qualname.replace(".", "/")
    candidates = {
        p for p in (f"{base_path}.py", f"{base_path}/__init__.py") if p in added_paths
    }
    if mod.startswith("vllm."):
        candidates |= {p for p in added_paths if p.startswith(base_path + "/")}
    return candidates


_LINT_ONLY: dict[Path, frozenset[str]] = {}


def _lint_only_files(state: RepoState) -> frozenset[str]:
    """Repo files pre-commit runs, read out of its own config, so a new hook
    script is covered the day it lands. Empty on anything unexpected, which
    leaves the caller's fail-open in place."""
    if state.repo in _LINT_ONLY:
        return _LINT_ONLY[state.repo]
    found: set[str] = set()
    try:
        raw = (state.repo / PRECOMMIT_CONFIG).read_text()
        config = yaml.safe_load(raw) or {}
        for repo_block in config.get("repos") or []:
            for hook in repo_block.get("hooks") or []:
                entry = hook.get("entry")
                if not isinstance(entry, str):
                    continue
                for token in entry.split():
                    if (state.repo / token).is_file():
                        found.add(token)
    except Exception:
        # Do NOT memoize a failure. The key is the repo path, and a cached
        # worktree reuses its path for a different commit after eviction, so one
        # unreadable config would pin "no lint-only files" for every later tree
        # there.
        return frozenset()
    _LINT_ONLY[state.repo] = frozenset(found)
    return _LINT_ONLY[state.repo]


def _classify_lint_only(state: RepoState, path: str) -> Claim | None:
    """The pre-commit config and the scripts it runs, when CI runs neither.

    pre-commit is a lint gate, not a test job. Where no step invokes it,
    editing a hook script cannot change what any test does, and the fail-open
    would otherwise run the whole pipeline on a lint tweak.

    Narrow on purpose: only files pre-commit's own config names, and only when
    nothing in CI references them. Living under `tools/` earns nothing here.
    """
    lint_files = _lint_only_files(state)
    if not lint_files:
        return None
    if path != PRECOMMIT_CONFIG and path not in lint_files:
        return None
    if _direct_step_refs(state, path) or path in state.docker_inputs:
        return None
    # Ignores the switch: a declared hook script must never be silenced into
    # "nothing to run".
    if _source_dep_steps_ungated(state, path):
        return None
    # If a step runs pre-commit itself, the config and its hooks belong to
    # that step and not to nothing.
    runners = {
        sid
        for p in state.pipelines
        for sid, st in p.targets.items()
        if PRECOMMIT_CONFIG in st.data_files or PRECOMMIT_CONFIG in st.scripts_seen
    }
    if runners:
        return Claim(
            "no-code",
            f"{path} is pre-commit configuration; running the steps that invoke it",
            step_ids=runners,
        )
    return Claim(
        "no-code",
        f"{path} is pre-commit configuration and no Buildkite step runs "
        "pre-commit; nothing to run",
    )


def _boot_gated_tests(state: RepoState, path: str, closure: set[str]) -> set[str]:
    """Test files in the closure with the boot-edge gate applied: a test
    reached only by crossing a gated platform edge depends on the file only if
    it boots an engine."""

    def _tests(files: set[str]) -> set[str]:
        return {f for f in files if is_test_file(f)}

    graph = state.full.graph
    test_files = _tests(closure)
    if not graph.boot_edges or not state.preflight.boot_gate_ok:
        return test_files
    base_closure = graph.reverse_closure({path}, include_boot=False)
    boot_only = test_files - _tests(base_closure)
    if not boot_only:
        return test_files
    return _tests(base_closure) | (boot_only & state.full.engine_starting_tests())


def _key_routed_steps(
    state: RepoState, path: str, closure: set[str]
) -> tuple[set[str], set[str]]:
    """If this file or anything in its closure is registered under a string
    key, the steps naming that key depend on it. Some e2e jobs pick their
    backend by name and never import it."""
    keys: set[str] = set()
    for member in {path} | closure:
        keys |= state.keys.for_file(member)
    return keys, state.keys.steps_naming(keys)
