# SPDX-License-Identifier: Apache-2.0
"""Intermediate state tracker for interactive configuration.

``InteractiveState`` holds the partially-configured benchmark parameters,
can be initialized from CLI args or a saved JSON file, and can be
converted to the ``argparse.Namespace`` that ``run_engine_bench()`` expects.
"""

# Standard
from typing import Any
import argparse
import json

# First Party
from lmcache.cli.commands.bench.engine_bench.interactive.schema import (
    ALL_ITEMS,
    PHASE_GENERAL,
    PHASE_REQUIRED,
    PHASE_WORKLOAD,
    ConfigItem,
)

# Keys that exist on argparse.Namespace but are NOT part of the interactive
# config item registry (operational flags, handled separately).
_OUTPUT_KEYS = ("output_dir", "seed", "no_csv", "export_csv", "json", "quiet")

# Keys used only during the interactive flow, never serialized or
# passed to the orchestrator.
_INTERACTIVE_ONLY_KEYS = {"has_lmcache"}

# Keys excluded from exported JSON configs.  These are either
# environment-specific (engine_url, lmcache_url) or interactive-only.
_EXPORT_EXCLUDED_KEYS = _INTERACTIVE_ONLY_KEYS | {"engine_url", "lmcache_url"}

# Mapping from ConfigItem.key to the argparse attribute name when they differ.
# Most keys match directly; these are the exceptions.
_KEY_TO_ATTR: dict[str, str] = {
    "kv_cache_volume": "kv_cache_volume",
    "tokens_per_gb_kvcache": "tokens_per_gb_kvcache",
}

# argparse attribute names where the CLI default is None (meaning "not set"),
# versus attributes where a non-None argparse default is the real default
# (e.g., kv_cache_volume defaults to 100.0).
_ARGPARSE_NONE_MEANS_UNSET = {
    "engine_url",
    "workload",
    "model",
    "lmcache_url",
    "tokens_per_gb_kvcache",
}


class InteractiveState:
    """Tracks which config items have been set and their values.

    Keys present in ``_values`` are considered "set".  Missing keys are
    "unset" and will either be prompted for or filled with defaults.
    """

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def is_set(self, key: str) -> bool:
        return key in self._values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def unset(self, key: str) -> None:
        """Remove a key so it counts as unset again (used when stepping back)."""
        self._values.pop(key, None)

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values)

    # ------------------------------------------------------------------
    # Readiness checks
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True when all required items (whose conditions are met) have values."""
        for item in ALL_ITEMS:
            if not item.required:
                continue
            if not self._condition_met(item):
                continue
            if not self.is_set(item.key):
                return False
        return True

    def get_missing_required(self) -> list[ConfigItem]:
        """Return phase-1 items that still need user input.

        Includes both required items that are unset, and non-required
        phase-1 items (like ``lmcache_url``) that are relevant because
        a downstream required item (``tokens_per_gb_kvcache``) is unset.
        """
        missing: list[ConfigItem] = []
        for item in ALL_ITEMS:
            if item.phase != PHASE_REQUIRED:
                continue
            if not self._condition_met(item):
                continue
            if self.is_set(item.key):
                continue
            if item.required:
                missing.append(item)
            elif item.key in ("has_lmcache", "lmcache_url") and not self.is_set(
                "tokens_per_gb_kvcache"
            ):
                # Only ask about LMCache when tokens_per_gb is needed
                missing.append(item)
        return missing

    def get_general_items(self) -> list[ConfigItem]:
        """Return phase-2 (general) items that are not yet set."""
        return [
            item
            for item in ALL_ITEMS
            if item.phase == PHASE_GENERAL
            and not self.is_set(item.key)
            and self._condition_met(item)
        ]

    def get_workload_items(self) -> list[ConfigItem]:
        """Return phase-3 (workload-specific) items whose conditions are met."""
        return [
            item
            for item in ALL_ITEMS
            if item.phase == PHASE_WORKLOAD and self._condition_met(item)
        ]

    def has_unconfigured_general(self) -> bool:
        """True if there are general items the user hasn't explicitly set."""
        return len(self.get_general_items()) > 0

    def has_workload_items(self) -> bool:
        """True if there are workload-specific items to configure."""
        return len(self.get_workload_items()) > 0

    def workload_items_all_default(self) -> bool:
        """True if no workload-specific items have been explicitly set."""
        for item in ALL_ITEMS:
            if item.phase != PHASE_WORKLOAD:
                continue
            if not self._condition_met(item):
                continue
            if self.is_set(item.key):
                return False
        return True

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def fill_defaults(self) -> None:
        """Set all unset items (whose conditions are met) to their defaults."""
        for item in ALL_ITEMS:
            if self.is_set(item.key):
                continue
            if not self._condition_met(item):
                continue
            if item.default is not None:
                self._values[item.key] = item.default

    # ------------------------------------------------------------------
    # Conversion: CLI args ↔ InteractiveState
    # ------------------------------------------------------------------

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "InteractiveState":
        """Build state from parsed CLI arguments.

        Only sets values that the user explicitly provided (i.e., not the
        argparse default).  This lets us distinguish "user set
        ``--kv-cache-volume 100``" from "user didn't touch it".
        """
        state = cls()
        for item in ALL_ITEMS:
            if item.key in _INTERACTIVE_ONLY_KEYS:
                continue
            attr = _KEY_TO_ATTR.get(item.key, item.key)
            value = getattr(args, attr, None)
            if value is None:
                continue
            # For keys where argparse default is None, any non-None value
            # means the user set it.
            if attr in _ARGPARSE_NONE_MEANS_UNSET:
                state._values[item.key] = value
                continue
            # For keys with real argparse defaults, we can't easily tell
            # if the user typed --kv-cache-volume 100 vs it being the
            # default.  We mark it as "set" only if it differs from the
            # schema default.  This is imperfect but good enough — the
            # worst case is we re-prompt for a value the user explicitly
            # set to the default.
            if item.default is not None and value == item.default:
                continue
            state._values[item.key] = value

        # Derive has_lmcache from lmcache_url if provided via CLI
        if state.is_set("lmcache_url"):
            state._values["has_lmcache"] = True

        return state

    def to_namespace(self) -> argparse.Namespace:
        """Convert to ``argparse.Namespace`` compatible with ``run_engine_bench``.

        Fills defaults for any unset items, then builds the namespace
        with the attribute names that ``parse_args_to_config()`` and
        ``create_workload()`` expect.
        """
        self.fill_defaults()
        ns = argparse.Namespace()

        # Map state keys to namespace attributes
        for item in ALL_ITEMS:
            if item.key in _INTERACTIVE_ONLY_KEYS:
                continue
            attr = _KEY_TO_ATTR.get(item.key, item.key)
            # Only fall back to schema default if the item's condition is met.
            # This prevents lmcache_url's default from leaking when
            # has_lmcache is not set.
            if item.key in self._values:
                value = self._values[item.key]
            elif self._condition_met(item):
                value = item.default
            else:
                value = None
            setattr(ns, attr, value)

        # Output settings (not in the interactive registry)
        ns.output_dir = self._values.get("output_dir", ".")
        ns.seed = self._values.get("seed", 42)
        ns.no_csv = self._values.get("no_csv", False)
        ns.json = self._values.get("export_json", False)
        ns.quiet = self._values.get("quiet", False)
        ns.bench_target = "engine"

        # Ensure format/output attrs exist for create_metrics
        if not hasattr(ns, "format"):
            ns.format = None
        if not hasattr(ns, "output"):
            ns.output = None

        return ns

    # ------------------------------------------------------------------
    # Conversion: JSON ↔ InteractiveState
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for config export.

        Excludes environment-specific keys (``engine_url``,
        ``lmcache_url``) and interactive-only keys so the exported
        config is portable and works without an LMCache server.
        """
        self.fill_defaults()
        return {k: v for k, v in self._values.items() if k not in _EXPORT_EXCLUDED_KEYS}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "InteractiveState":
        """Load from a saved config JSON dict."""
        state = cls()
        for key, value in data.items():
            state._values[key] = value
        return state

    def save_json(self, path: str) -> None:
        """Export the current state to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
            f.write("\n")

    @classmethod
    def load_json(cls, path: str) -> "InteractiveState":
        """Load state from a JSON config file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_json(data)

    def merge_cli_args(self, args: argparse.Namespace) -> None:
        """Merge CLI args on top of existing state (CLI args win)."""
        cli_state = InteractiveState.from_cli_args(args)
        self._values.update(cli_state._values)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_lines(self) -> list[tuple[str, str]]:
        """Return ``(label, value_str)`` pairs for the config summary."""
        lines: list[tuple[str, str]] = []
        for item in ALL_ITEMS:
            if item.key in _INTERACTIVE_ONLY_KEYS:
                continue
            if not self._condition_met(item):
                continue
            value = self._values.get(item.key, item.default)
            if value is None or value == "":
                if item.key == "model":
                    display = "(auto-detect)"
                elif item.key == "lmcache_url":
                    continue  # skip empty lmcache_url
                else:
                    display = "(not set)"
            else:
                display = str(value)
            lines.append((item.display_name, display))
        return lines

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _condition_met(self, item: ConfigItem) -> bool:
        """Check whether an item's condition is satisfied."""
        if item.condition is None:
            return True
        return item.condition(self._values)
