# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Everything derivable from a checkout, built once and reused across diffs.

No policy lives here. This module answers "what is in this tree" -- the parsed
pipelines, the import graph, the test catalog, preflight, the docs deps, the
image DAG -- and `classify.py` decides what any of it means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..handwritten import PATH_TOKEN_FAMILIES
from . import hardware
from .build_map import BuildMap
from .docs_deps import DocsDeps, build_docs_deps
from .externals import copy_inputs, docker_image_inputs, release_pipeline_refs
from .graph.build import FullGraph, build_full_graph
from .guards import PreflightReport, run_preflight
from .native_ops import NativeOps
from .pipeline.buildkite import load_pipeline_configs, load_steps
from .pipeline.images import ArtifactGraph, add_image_inputs, build_artifact_graph
from .pipeline.invoked_tests import invoked_files, legacy_amd_invoked
from .pipeline.scripts import scan_script
from .pipeline.step import LoadReport, PipelineConfig, Step
from .pipeline.targets import StepTargets, map_step
from .registered_names import KeyIndex
from .repo import TestIndex, build_test_index, is_test_file, test_file_catalog
from .rust_workspace import RustWorkspace


@dataclass
class DiffContext:
    """Refs and per-path statuses, present only when select() gets both ends of
    a real diff. That is what turns on table-aware treatment and the added-file
    rules, both of which assume state built at `base`."""

    base: str
    head: str
    status: dict[str, str]  # path -> A/M/D/R/C/T
    renames: dict[str, str] = field(default_factory=dict)  # new path -> old path


@dataclass
class PipelineData:
    config: PipelineConfig
    steps: list[Step]
    targets: dict[str, StepTargets]  # step_id -> targets


@dataclass
class RepoState:
    """Everything derivable from a checkout, reusable across diffs."""

    repo: Path
    pipelines: list[PipelineData]
    full: FullGraph
    catalog: list[str]
    load_report: LoadReport
    # test files at least one auto-run step invokes, so orphans and
    # optional-only coverage are out
    invoked: set[str] = field(default_factory=set)
    keys: KeyIndex = field(default_factory=KeyIndex)
    auto_step_ids: set[str] = field(default_factory=set)
    auto_run_files: set[str] = field(default_factory=set)
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
    # which steps build which container image, and who consumes it
    artifacts: ArtifactGraph = field(default_factory=ArtifactGraph)
    # crate closures of the two shipped rust artifacts (the rust rule's map)
    rust_workspace: RustWorkspace = field(default_factory=RustWorkspace)
    # csrc/cmake path -> the device families that compile it
    build_map: BuildMap = field(default_factory=BuildMap)
    # csrc path -> its ops -> the Python wrappers calling them
    native_ops: NativeOps = field(default_factory=NativeOps)

    @classmethod
    def build(cls, repo: Path) -> RepoState:
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
            state.auto_run_files.update(st.data_files)
            state.auto_run_files.update(st.scripts_seen)
            for t in st.targets:
                if t.path.endswith(".py"):
                    state.auto_run_files.add(t.path)
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
        state.artifacts = build_artifact_graph(repo, pipelines)
        state.rust_workspace = RustWorkspace.build(repo)
        state.build_map = BuildMap.build(repo)
        state.native_ops = NativeOps.build(repo, state.catalog)
        dockerfiles = {d for fs in state.artifacts.defined_by.values() for d in fs}
        in_files, in_dirs, blanket = copy_inputs(repo, dockerfiles)
        add_image_inputs(
            repo,
            state.artifacts,
            in_files,
            in_dirs,
            blanket,
            lambda f: _graph_known(state, f),
            hardware.family_of_path,
        )
        return state

    def test_index(self) -> TestIndex:
        """Every test file in the tree, by directory and by basename.

        Index and catalog unioned because neither contains the other: the
        catalog is a `test_*.py` glob that picks up files no module index can
        name, and the index holds test modules the glob's SKIP_DIRS drop.
        """
        if not hasattr(self, "_test_index"):
            files = frozenset(
                f
                for f in set(self.full.index.file_to_module) | set(self.catalog)
                if is_test_file(f)
            )
            self._test_index = build_test_index(files)
        return self._test_index

    def direct_test_importers(self) -> dict[str, frozenset[str]]:
        """file -> the test files importing it, depth 1 only.

        Not a closure: inside the import cycle the transitive version reaches
        everything and says nothing, while a literal `from vllm.x.y import z`
        edge still names one file.

        Demoted edges are kept, unlike the closure walks. Callers only ever
        union this in, so an edge selection has disowned can widen the answer
        and never narrow it.
        """
        if not hasattr(self, "_direct_test_importers"):
            found: dict[str, set[str]] = {}
            for test in self.test_index().files:
                for target in self.full.graph.imports.get(test, ()):
                    found.setdefault(target, set()).add(test)
            self._direct_test_importers = {
                target: frozenset(tests) for target, tests in found.items()
            }
        return self._direct_test_importers

    def family_steps(self, family: str) -> set[str]:
        return {
            s.step_id
            for p in self.pipelines
            for s in p.steps
            if hardware.step_in_family(s, family)
        }

    def family_partition(
        self,
    ) -> tuple[dict[str, frozenset[str]], frozenset[str], frozenset[str]]:
        """(per token family, their union, the auto steps in no family).

        The remainder is the main image and its steps. Cached on the instance,
        not on id(), so a collected state cannot alias another's answer.
        """
        if not hasattr(self, "_family_partition"):
            per = {
                fam: frozenset(self.family_steps(fam))
                for _tokens, fam in PATH_TOKEN_FAMILIES
            }
            union = frozenset().union(*per.values())
            self._family_partition = (per, union, frozenset(self.auto_step_ids) - union)
        return self._family_partition


def detect_duplicate_ids(steps: list[Step], report: LoadReport) -> None:
    seen: set[str] = set()
    for s in steps:
        if s.step_id in seen and s.step_id not in report.duplicate_ids:
            report.duplicate_ids.append(s.step_id)
        seen.add(s.step_id)


def _graph_known(state: RepoState, path: str) -> bool:
    g = state.full.graph
    return (
        path in state.full.index.file_to_module
        or path in g.imports
        or path in g.reverse
    )
