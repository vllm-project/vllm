# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full graph assembly: imports + wall parsers + lazy-edge finalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..curated import CONFTESTS_NOT_ENGINE_STARTING, ENGINE_ENTRY_MODULES
from ..repo import ModuleIndex, build_module_index
from .assets import AssetParse, add_asset_edges
from .dispatch import DispatchParse, add_demotion_edges
from .factories import FactoryParse, add_factory_edges
from .imports import ImportGraph, build_graph
from .platform import PlatformParse, add_platform_edges
from .registry import (
    QuantParse,
    RegistryParse,
    add_quant_method_edges,
    add_registry_edges,
    string_keyed_claims,
)
from .spawn import SpawnParse, add_spawn_edges


@dataclass
class FullGraph:
    index: ModuleIndex
    graph: ImportGraph
    registry: RegistryParse
    quant: QuantParse
    factories: FactoryParse
    dispatch: DispatchParse
    platform: PlatformParse
    spawn: SpawnParse
    assets: AssetParse
    # module-level AST reverse edges, snapshotted BEFORE wall parsers add
    # synthesized edges: the exclusivity invariant reads real imports only
    plain_reverse: dict[str, set[str]] = field(default_factory=dict)

    def all_claims(self) -> set[str]:
        """String-keyed targets across every wall parser plus the demotion
        pass: files whose coverage routes by key, not by import edge."""
        return (
            string_keyed_claims(self.registry, self.index)
            | self.quant.claims
            | self.factories.claims
            | self.dispatch.claims
        )

    def engine_starting_tests(self) -> set[str]:
        """Tests that actually boot an engine: they import an engine
        entrypoint (LLM, CLI, api_server, engine core) or take the
        vllm_runner fixture. Gates worker-seam reachability in selection."""
        if not hasattr(self, "_engine_tests"):
            entry_files = {
                f for m in ENGINE_ENTRY_MODULES if (f := self.index.resolve(m))
            }
            tests = set(self.graph.engine_fixture_files)
            # Conftests that import an entrypoint (server fixtures etc.)
            # make every test beneath them engine-starting; the conftest
            # auto-load edges are exactly the descendant tests.
            engine_conftests = {
                f
                for f, imports in self.graph.imports.items()
                if f.startswith("tests/")
                and f.endswith("/conftest.py")
                and f not in CONFTESTS_NOT_ENGINE_STARTING
                and imports & entry_files
            }
            for file, imports in self.graph.imports.items():
                if not file.startswith("tests/"):
                    continue
                if imports & entry_files or imports & engine_conftests:
                    tests.add(file)
            self._engine_tests = tests
        return self._engine_tests


def build_full_graph(repo: Path) -> FullGraph:
    index = build_module_index(repo)
    graph = build_graph(repo, index)
    plain_reverse = {dst: set(srcs) for dst, srcs in graph.reverse.items()}
    registry = add_registry_edges(repo, index, graph)
    quant = add_quant_method_edges(repo, index, graph)
    factories = add_factory_edges(repo, index, graph)
    # Members already keyed by a typed registration (registry/quant/attention
    # enum): Ext C may demote a type-only import of one of these, routing it by
    # its own key.
    pre_claims = string_keyed_claims(registry, index) | quant.claims | factories.claims
    # Demote config-key-guarded eager plugin imports (proposers, pooling
    # runner): its claims join the set so platform-gating skips them and
    # finalize_lazy_edges drops lazy imports into them, like the other
    # string-keyed registries.
    dispatch = add_demotion_edges(repo, index, graph, preclaimed=frozenset(pre_claims))
    claims = pre_claims | dispatch.claims
    platform = add_platform_edges(repo, index, graph, claimed=claims)
    spawn = add_spawn_edges(repo, index, graph)
    assets = add_asset_edges(repo, graph)
    graph.finalize_lazy_edges(claims)
    return FullGraph(
        index=index,
        graph=graph,
        registry=registry,
        quant=quant,
        factories=factories,
        dispatch=dispatch,
        platform=platform,
        spawn=spawn,
        assets=assets,
        plain_reverse=plain_reverse,
    )
