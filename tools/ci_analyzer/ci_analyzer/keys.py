# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered-key index: string names that select code without importing it.

E2E jobs pin connectors, backends, quant methods, and models by string in argv,
env blocks, and config files. This index maps every registered key to the module
it names, and every step to the keys its commands, scripts, env, config files, or
target test files mention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from .curated import RAW_KEY_MIN_LEN as _RAW_KEY_MIN_LEN
from .curated import SUBSTRING_KEY_MIN_LEN as _SUBSTRING_KEY_MIN_LEN
from .graph.build import FullGraph
from .graph.registry import resolve_module_name

# Typed contexts. A parser key selects code only when passed via a parser
# flag/config field; a register key (connector/backend/engine) only as a
# quoted or assigned value ('"NixlConnector"', 'KV_CONNECTOR=${X:-Nixl..}').
# This is what keeps 'ipc' from matching docker's --ipc=host and
# 'granite' from matching ibm-granite/ model ids. quant/enum/arch/hf_id
# keys stay untyped: their legitimate contexts (eval configs, backend
# matrices) cannot be enumerated without risking eval coverage.
_PARSER_FLAG_RE = (
    r"(?:--tool-call-parser|--reasoning-parser|--tokenizer-mode"
    r"|tool_call_parser|reasoning_parser|tokenizer_mode)"
)


def _typed_pattern(key: str, mechanism: str) -> re.Pattern | None:
    esc = re.escape(key)
    if mechanism == "parser":
        # Flag adjacency, OR a quoted/assigned/shell-default value context:
        # eval harnesses pin via env indirection (TOOL_CALL_PARSER="${X:-openai}").
        # Path fragments (entrypoints/openai) and model-id substrings match neither.
        return re.compile(
            _PARSER_FLAG_RE + r"[\"']?[ =:]+[\"']?" + esc + r"\b"
            r"|(?::-|[\"'=])\s*\\?[\"']?" + esc + r"\b"
        )
    if mechanism == "register":
        return re.compile(r"(?::-|[\"'=:])\s*\\?[\"']?" + esc + r"\b")
    return None


def _strip_comment_lines(text: str) -> str:
    """Shell/YAML comment lines never execute and must not route keys
    ('# Default: TRITON_ATTN on ROCm' selected a step)."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _is_scalar_literal(key: str) -> bool:
    """A gating literal that parses as a JSON non-string is a truthiness test,
    not a config selector: `VLLM_..._FORCE_FIRST_CONFIG in ("1","true")` mints
    `1` and `true`, which then match `-tp=1`, `sleep 1` and `|| true` across
    the pipeline.

    This replaces a step-fanout threshold that deleted any dispatch key naming
    more than N steps. That bar could not tell a common word from a popular
    genuine key, and deleting a genuine one severed the only route the member
    had left (demotion having already cut its import edge), silently. Measured
    over every key at HEAD, this rejects exactly the same set. json, not yaml:
    yaml would also swallow on/off/yes/no, which are plausible config values."""
    try:
        return not isinstance(json.loads(key), str)
    except ValueError:
        return False


@dataclass
class KeyIndex:
    # registered-module file or package dir -> string keys naming it
    keyed_modules: dict[str, set[str]] = field(default_factory=dict)
    # step_id -> registered keys appearing in the step's searchable text
    step_keys: dict[str, set[str]] = field(default_factory=dict)
    # step_id -> the full searchable text (commands + scripts + env +
    # config files + target-test literals), kept so ad-hoc keys that were
    # not registered at build time (table-diff head-side ids) can be matched
    searchable: dict[str, str] = field(default_factory=dict)
    # key -> mechanism (register|parser|enum|quant|arch|hf_id|class_table)
    key_mechanism: dict[str, str] = field(default_factory=dict)
    # gating literal -> why the mint deliberately declined to register it.
    # dropped-edges reads this to tell "refused on purpose" from "should have
    # been routable and is not", which is the shape of a real routing gap.
    refused: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, repo: Path, full: FullGraph, pipelines) -> KeyIndex:
        index = cls()
        entries = _registered_entries(full)
        for key, (module_file, mechanism) in entries.items():
            index.key_mechanism[key] = mechanism
            index.keyed_modules.setdefault(module_file, set()).add(key)
            if module_file.endswith("/__init__.py"):
                pkg = module_file[: -len("__init__.py")]
                index.keyed_modules.setdefault(pkg, set()).add(key)
        # Severed (claimed) demoted members route by their gating literals:
        # eval/PD-accuracy configs select `method: eagle` by string without
        # importing, and the member's deflated closure no longer reaches those
        # steps. Scoped to claimed members (a broad member is already covered
        # by its own closure); a typed registration owns any colliding word.
        for (_, member), lits in full.dispatch.demotions.items():
            if member not in full.dispatch.claims:
                continue
            for lit in lits:
                mech = index.key_mechanism.get(lit)
                if mech and _typed_pattern(lit, mech):
                    index.refused.setdefault(lit, f"typed-owned ({mech})")
                    continue
                if _is_scalar_literal(lit):
                    index.refused.setdefault(lit, "non-string scalar")
                    continue
                index.key_mechanism.setdefault(lit, "dispatch")
                index.keyed_modules.setdefault(member, set()).add(lit)
        all_keys = {k for ks in index.keyed_modules.values() for k in ks}
        typed = {
            k: pat
            for k in all_keys
            if (pat := _typed_pattern(k, index.key_mechanism[k]))
        }
        untyped = all_keys - set(typed)
        substring_keys = {
            k for k in untyped if "/" in k or len(k) >= _SUBSTRING_KEY_MIN_LEN
        }
        patterns = {
            key: re.compile(rf"\b{re.escape(key)}\b")
            for key in untyped - substring_keys
        }
        literals = full.graph.string_literals
        for pdata in pipelines:
            steps_by_id = {s.step_id: s for s in pdata.steps}
            for sid, st in pdata.targets.items():
                target_literals: set[str] = set()
                for t in st.targets:
                    if t.path.endswith(".py"):
                        target_literals |= literals.get(t.path, set())
                haystack = st.haystack
                step = steps_by_id.get(sid)
                if step and step.env:
                    haystack += "\n" + "\n".join(
                        f"{k}={v}" for k, v in step.env.items()
                    )
                for data_file in st.data_files:
                    haystack += _config_text(repo, data_file, full.graph.parse_errors)
                stripped = _strip_comment_lines(haystack)
                hits = match_keys(
                    substring_keys, patterns, typed, target_literals, stripped
                )
                if hits:
                    index.step_keys[sid] = hits
                # Store the comment-stripped text: steps_naming_raw searches
                # this, so a model id or package name living only in a shell
                # comment must not route a step (as step_keys also uses `stripped`).
                index.searchable[sid] = (
                    stripped + "\n" + "\n".join(sorted(target_literals))
                )
        return index

    def for_file(self, path: str) -> set[str]:
        keys: set[str] = set()
        for module_or_pkg, ks in self.keyed_modules.items():
            if path == module_or_pkg or (
                module_or_pkg.endswith("/") and path.startswith(module_or_pkg)
            ):
                keys |= ks
        return keys

    def steps_naming(self, keys: set[str]) -> set[str]:
        return {sid for sid, step_keys in self.step_keys.items() if keys & step_keys}

    def steps_naming_raw(self, keys: set[str]) -> set[str]:
        """Substring search for keys unknown at index-build time (table-diff
        head-side archs/ids). Specific strings only (CamelCase, org/id).
        The bar is deliberately LOWER than _SUBSTRING_KEY_MIN_LEN: raw keys
        run only for the handful of diffed entries in one table claim
        (recall-biased, bounded blast radius) while registered keys match
        persistently across every step (precision-biased); 18 real archs
        are 8-11 chars, so raising this bar under-selects renamed/added
        archs."""
        candidates = {k for k in keys if "/" in k or len(k) >= _RAW_KEY_MIN_LEN}
        return {
            sid
            for sid, text in self.searchable.items()
            if any(k in text for k in candidates)
        }


def match_keys(
    substring_keys: set[str],
    patterns: dict[str, re.Pattern],
    typed: dict[str, re.Pattern],
    target_literals: set[str],
    haystack: str,
) -> set[str]:
    """A key hits a step via an exact literal in the step's own target test
    files, or via its mechanism-appropriate context in the (comment-
    stripped) command haystack."""
    hits = {k for k in substring_keys if k in target_literals or k in haystack}
    hits |= {
        key
        for key, pat in patterns.items()
        if key in target_literals or pat.search(haystack)
    }
    hits |= {
        key
        for key, pat in typed.items()
        if key in target_literals or pat.search(haystack)
    }
    return hits


def _registered_entries(full: FullGraph) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for table, mechanism in (
        (full.factories.register_entries, "register"),
        (full.factories.parser_entries, "parser"),
        (full.factories.enum_entries, "enum"),
        (full.factories.class_table_entries, "class_table"),
        (full.quant.methods, "quant"),
    ):
        for key, module_file in table.items():
            entries[key] = (module_file, mechanism)
    for arch, (mod, _cls) in full.registry.entries.items():
        resolved = full.index.resolve(resolve_module_name(mod))
        if resolved is None:
            continue
        entries[arch] = (resolved, "arch")
        for hf_id in full.registry.hf_ids.get(arch, ()):
            entries[hf_id] = (resolved, "hf_id")
    return entries


def _config_text(repo: Path, data_file: str, errors: list[str] | None = None) -> str:
    """A config file's text, plus one level of indirection: a list file
    (models-small.txt) enumerates per-model yaml configs whose contents
    carry the actual model ids and backend flags. Unreadable files are
    recorded into `errors` (graph.parse_errors) so preflight sees the
    degraded haystack instead of a silent ''."""
    try:
        text = (repo / data_file).read_text()
    except (UnicodeDecodeError, OSError):
        if errors is not None and data_file not in errors:
            errors.append(data_file)
        return ""
    out = "\n" + text
    parent = data_file.rsplit("/", 1)[0]
    for line in text.splitlines():
        name = line.strip()
        if not name or name.startswith("#") or "/" in name:
            continue
        if name.endswith((".yaml", ".yml", ".json", ".txt")):
            try:
                out += "\n" + (repo / parent / name).read_text()
            except (UnicodeDecodeError, OSError):
                nested = f"{parent}/{name}"
                if errors is not None and nested not in errors:
                    errors.append(nested)
    return out
