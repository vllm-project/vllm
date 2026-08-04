# SPDX-License-Identifier: Apache-2.0
"""Interactive configuration flow for ``lmcache bench engine``.

Entry point: ``run_interactive(args)`` — walks the user through
missing configuration, offers gates for optional settings, and
returns a complete ``argparse.Namespace`` ready for the orchestrator.
"""

# Standard
from typing import cast
import argparse
import sys

# First Party
from lmcache.cli.commands.bench.engine_bench.interactive.schema import (
    ConfigItem,
)
from lmcache.cli.commands.bench.engine_bench.interactive.state import (
    InteractiveState,
)
from lmcache.cli.commands.bench.engine_bench.interactive.terminal import (
    BOLD,
    CYAN,
    GO_BACK,
    RESET,
    YELLOW,
    prompt_bool,
    prompt_choice,
    prompt_number,
    prompt_text,
)

__all__ = ["run_interactive"]


# Config items whose text input is a URL and accepts port/host shorthand.
_URL_KEYS = {"engine_url", "lmcache_url"}


def _normalize_url(value: str) -> str:
    """Expand shorthand URL input into a fully-qualified URL.

    A bare port (``8000``) becomes ``http://localhost:8000``; a bare host or
    ``host:port`` (``localhost:8000``) gains an ``http://`` scheme.  Values
    that already carry a scheme (``://``) and the empty string are returned
    unchanged.
    """
    value = value.strip()
    if not value or "://" in value:
        return value
    if value.isdigit():
        return f"http://localhost:{value}"
    return f"http://{value}"


# ---------------------------------------------------------------------------
# Prompt dispatcher
# ---------------------------------------------------------------------------


def _prompt_for_item(item: ConfigItem, allow_back: bool) -> object:
    """Prompt the user for a single config item based on its type.

    Returns the entered value, or :data:`GO_BACK` when ``allow_back`` is True
    and the user asks to step back to the previous question.
    """
    if item.input_type == "text":
        return prompt_text(
            item.display_name,
            item.description,
            default=item.default if item.default is not None else "",
            allow_back=allow_back,
        )
    if item.input_type == "int":
        return prompt_number(
            item.display_name,
            item.description,
            default=item.default,
            number_type=int,
            allow_back=allow_back,
        )
    if item.input_type == "float":
        return prompt_number(
            item.display_name,
            item.description,
            default=item.default,
            number_type=float,
            allow_back=allow_back,
        )
    if item.input_type == "bool":
        return prompt_bool(
            item.display_name,
            item.description,
            default=bool(item.default) if item.default is not None else True,
            allow_back=allow_back,
        )
    if item.input_type == "choice":
        return prompt_choice(
            item.display_name,
            item.description,
            choices=item.choices,
            default=item.default if item.default is not None else "",
            allow_back=allow_back,
        )
    raise ValueError(f"Unknown input_type {item.input_type!r} for {item.key}")


# ---------------------------------------------------------------------------
# Gate prompt
# ---------------------------------------------------------------------------


def _prompt_gate(section_name: str, detail: str, allow_back: bool) -> object:
    """Ask the user whether to configure a section or skip with defaults.

    Returns ``"configure"``, ``"use defaults"``, or :data:`GO_BACK`.
    """
    return prompt_choice(
        section_name,
        f"Would you like to configure {detail}?\n  Defaults will be used if you skip.",
        choices=[
            ("use defaults", "Skip, use defaults"),
            ("configure", "Yes, configure"),
        ],
        default="use defaults",
        allow_back=allow_back,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(state: InteractiveState) -> None:
    """Print a formatted configuration summary."""
    print()
    print(f"{BOLD}{'─' * 50}{RESET}")
    print(f"{BOLD} Configuration Summary{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")
    for label, value in state.summary_lines():
        padding = max(0, 26 - len(label))
        print(f"  {label}:{' ' * padding}{CYAN}{value}{RESET}")
    print(f"{BOLD}{'─' * 50}{RESET}")


# ---------------------------------------------------------------------------
# Action prompt
# ---------------------------------------------------------------------------


def _prompt_action(allow_back: bool) -> object:
    """Ask the user to start the benchmark, export config, or quit.

    Returns ``"start"``, ``"export"``, ``"quit"``, or :data:`GO_BACK`.
    """
    return prompt_choice(
        "What would you like to do?",
        "",
        choices=[
            ("start", "Start benchmark"),
            ("export", "Export configuration for later use and exit"),
            ("quit", "Quit without running"),
        ],
        default="start",
        allow_back=allow_back,
    )


def _resolve_before_export(state: InteractiveState) -> None:
    """Resolve tokens_per_gb_kvcache and model before exporting.

    If the user provided an LMCache URL, query the server to get
    ``tokens_per_gb_kvcache`` so the exported config is standalone.
    If the model is empty and an engine URL is available, auto-detect it.
    """
    # First Party
    from lmcache.cli.commands.bench.engine_bench.config import (
        auto_detect_model,
        resolve_tokens_per_gb,
    )

    engine_url = state.get("engine_url", "")
    model = state.get("model", "")

    # Auto-detect model if empty
    if not model and engine_url:
        try:
            model = auto_detect_model(engine_url)
            state.set("model", model)
        except RuntimeError as e:
            print(f"  {YELLOW}Warning: could not auto-detect model: {e}{RESET}")

    # Resolve tokens_per_gb from LMCache if needed
    lmcache_url = state.get("lmcache_url", "")
    if lmcache_url and not state.is_set("tokens_per_gb_kvcache"):
        try:
            tokens = resolve_tokens_per_gb(lmcache_url, model)
            state.set("tokens_per_gb_kvcache", tokens)
        except RuntimeError as e:
            print(
                f"  {YELLOW}Warning: could not resolve "
                f"tokens_per_gb from LMCache: {e}{RESET}"
            )


def _handle_export(state: InteractiveState) -> None:
    """Prompt for filename, resolve values, save JSON, and exit."""
    _resolve_before_export(state)
    # No allow_back here, so the return is always a plain string.
    filename = cast(
        str,
        prompt_text(
            "Export filename",
            "",
            default="bench_config.json",
        ),
    )
    state.save_json(filename)
    print()
    print(f"  {CYAN}Saved to {filename}{RESET}")
    print(
        f"  {BOLD}Replay with:{RESET} "
        f"{CYAN}lmcache bench engine "
        f"--engine-url <URL> --config {filename}{RESET}"
    )
    print()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# Heading text for each section gate, keyed by section name.
_GATE_TEXT = {
    "general": ("General settings", "general settings (model, KV cache volume, etc.)"),
}


def _next_question(
    state: InteractiveState, gates: dict[str, str]
) -> ConfigItem | str | None:
    """Determine the next question to ask.

    Returns a ``ConfigItem`` to prompt for a value, a section name
    (``"general"`` / ``"workload"``) to prompt the section gate, or None when
    configuration is complete.  The result is derived purely from current
    state plus the gate decisions already made, so stepping back (which
    un-sets the most recent answer) naturally re-presents the question that
    produced it.
    """
    missing = state.get_missing_required()
    if missing:
        return missing[0]

    if state.has_unconfigured_general() and "general" not in gates:
        return "general"
    if gates.get("general") == "configure":
        general = state.get_general_items()
        if general:
            return general[0]

    workload = [i for i in state.get_workload_items() if not state.is_set(i.key)]
    if workload and "workload" not in gates:
        return "workload"
    if gates.get("workload") == "configure" and workload:
        return workload[0]

    return None


def _gate_text(section: str, state: InteractiveState) -> tuple[str, str]:
    """Return the ``(section_name, detail)`` heading for a gate prompt."""
    if section == "workload":
        return (
            f"Workload settings ({state.get('workload', 'workload')})",
            "workload-specific settings",
        )
    return _GATE_TEXT[section]


def _gather_config(state: InteractiveState) -> bool:
    """Drive the question loop until the user starts, exports, or quits.

    Walks required items, the general/workload gates, and the summary, while
    letting the user step back to revise any earlier answer.  Gate decisions
    are tracked outside ``state`` so they never leak into the exported config.

    Returns True to start the benchmark, False to quit.  Exits the process
    directly on export.
    """
    gates: dict[str, str] = {}
    # Trail of committed answers, for stepping back.  Item keys are stored as
    # the key itself; gate decisions as ``"gate:<section>"``.
    trail: list[str] = []

    def step_back() -> None:
        marker = trail.pop()
        if marker.startswith("gate:"):
            gates.pop(marker[len("gate:") :], None)
        else:
            state.unset(marker)

    while True:
        question = _next_question(state, gates)

        if question is None:
            _print_summary(state)
            action = _prompt_action(allow_back=bool(trail))
            if action is GO_BACK:
                step_back()
                continue
            if action == "export":
                _handle_export(state)  # exits
            return action == "start"

        if isinstance(question, ConfigItem):
            value = _prompt_for_item(question, allow_back=bool(trail))
            if value is GO_BACK:
                step_back()
                continue
            if question.key in _URL_KEYS and isinstance(value, str):
                value = _normalize_url(value)
            state.set(question.key, value)
            trail.append(question.key)
        else:
            name, detail = _gate_text(question, state)
            choice = _prompt_gate(name, detail, allow_back=bool(trail))
            if choice is GO_BACK:
                step_back()
                continue
            gates[question] = str(choice)
            trail.append(f"gate:{question}")


def run_interactive(args: argparse.Namespace) -> argparse.Namespace:
    """Run the interactive configuration flow.

    Walks the user through missing required items, offers gates for
    general and workload-specific settings, shows a summary, and
    returns a complete ``argparse.Namespace``.  At any prompt the user can
    step back to the previous question (``<``) or exit (Ctrl-C / Ctrl-D, or
    the Quit action).

    Args:
        args: Partially-populated CLI args (some may be None).

    Returns:
        A fully-populated ``argparse.Namespace`` ready for the
        benchmark orchestrator.
    """
    state = InteractiveState.from_cli_args(args)

    print()
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD} lmcache bench engine — Interactive Setup{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"  {YELLOW}Type < to go back · Ctrl-C to exit{RESET}")

    try:
        start = _gather_config(state)
    except (KeyboardInterrupt, EOFError):
        print()
        print(f"  {YELLOW}Setup cancelled.{RESET}")
        sys.exit(0)

    if not start:
        print()
        print(f"  {YELLOW}Quit without running.{RESET}")
        sys.exit(0)

    # Carry over output settings from original CLI args
    ns = state.to_namespace()
    for attr in ("output_dir", "seed", "no_csv", "json", "quiet", "format", "output"):
        cli_val = getattr(args, attr, None)
        if cli_val is not None:
            setattr(ns, attr, cli_val)

    return ns
