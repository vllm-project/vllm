# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full graph assembly: imports + edge parsers + lazy-edge finalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...handwritten import (
    CONFTESTS_NOT_ENGINE_STARTING,
    ENGINE_ENTRY_MODULES,
    REGISTRY_FILE,
)
from ..repo import ModuleIndex, build_module_index
from .assets import AssetParse, add_asset_edges
from .cycles import ImportCycle, dominant_cycle
from .demote import DispatchParse, add_demotion_edges
from .factories import FactoryParse, add_factory_edges
from .imports import ImportGraph, build_graph
from .model_registry import (
    QUANT_INIT_FILE,
    QuantParse,
    RegistryParse,
    add_quant_method_edges,
    add_registry_edges,
    resolve_module_name,
    string_keyed_claims,
)
from .platform import PlatformParse, add_platform_edges
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
    # module-level AST reverse edges, snapshotted BEFORE edge parsers add
    # synthesized edges: the exclusivity invariant reads real imports only
    plain_reverse: dict[str, set[str]] = field(default_factory=dict)

    def all_claims(self) -> set[str]:
        """String-keyed targets across every edge parser plus the demotion
        pass: files whose coverage routes by key, not by import edge."""
        return (
            string_keyed_claims(self.registry, self.index)
            | self.quant.claims
            | self.factories.claims
            | self.dispatch.claims
        )

    def table_of(self) -> dict[str, frozenset[str]]:
        """Target file -> the table files that name it.

        A registered member with no test naming its key has no in-edges; this
        is the one hop back to the table whose consumers can load it. Merged
        here because quant and the model registry keep separate parse results.
        """
        if not hasattr(self, "_table_of"):
            merged: dict[str, set[str]] = {
                t: set(tables) for t, tables in self.factories.table_of.items()
            }
            for target in self.quant.methods.values():
                if target != QUANT_INIT_FILE:
                    merged.setdefault(target, set()).add(QUANT_INIT_FILE)
            for mod, _cls in self.registry.entries.values():
                resolved = self.index.resolve(resolve_module_name(mod))
                if resolved and resolved != REGISTRY_FILE:
                    merged.setdefault(resolved, set()).add(REGISTRY_FILE)
            self._table_of = {t: frozenset(s) for t, s in merged.items()}
        return self._table_of

    def import_cycle(self) -> ImportCycle:
        """The dominant import cycle, computed once.

        Cached here and not on ImportGraph, whose `add_edge` clears caches on
        every call; a FullGraph never changes after it is built.
        """
        if not hasattr(self, "_import_cycle"):
            self._import_cycle = dominant_cycle(self.graph)
        return self._import_cycle

    def engine_starting_tests(self) -> set[str]:
        """Tests that actually boot an engine: they import an engine
        entrypoint (LLM, CLI, api_server, engine core) or take the
        vllm_runner fixture. Gates boot-edge reachability in selection."""
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
