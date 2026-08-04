# SPDX-License-Identifier: Apache-2.0
"""Terminal UI primitives for interactive configuration.

Provides simple prompt functions for collecting user input:

- ``prompt_text`` — free-form text with optional default
- ``prompt_number`` — numeric input with type validation
- ``prompt_bool`` — Y/N confirmation
- ``prompt_choice`` — arrow-key selection (falls back to numbered list on non-TTY)

Every prompt accepts ``allow_back``.  When enabled, the user steps back to the
previous question by typing ``<`` — in every prompt, typed or arrow-key — in
which case the prompt returns the :data:`GO_BACK` sentinel instead of a value.
"""

# Standard
import sys


class _GoBack:
    """Sentinel type returned by prompts when the user asks to step back."""


# Returned by any prompt (when ``allow_back=True``) to mean "go to the
# previous question".  Compare with ``is`` — there is only one instance.
GO_BACK = _GoBack()

# Token the user types to go back, in any text/number/bool prompt and in the
# non-TTY choice fallback.
BACK_TOKEN = "<"

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
CURSOR_UP = "\033[A"


def _is_tty() -> bool:
    return hasattr(sys.stdin, "fileno") and sys.stdin.isatty()


def _print_back_hint() -> None:
    """Print the dim 'type < to go back' hint used by typed prompts."""
    print(f"  {DIM}(type {BACK_TOKEN} to go back){RESET}")


# ---------------------------------------------------------------------------
# prompt_text
# ---------------------------------------------------------------------------


def prompt_text(
    label: str,
    description: str = "",
    default: str = "",
    allow_back: bool = False,
) -> str | _GoBack:
    """Prompt for free-form text input.

    Args:
        label: The config item name shown as a heading.
        description: One-line explanation shown below the label.
        default: Default value; shown in brackets, returned on empty Enter.
        allow_back: If True, typing ``<`` returns :data:`GO_BACK`.

    Returns:
        The user's input, the default if they pressed Enter, or
        :data:`GO_BACK` if they asked to step back.
    """
    print()
    print(f"{BOLD}{label}{RESET}")
    if description:
        print(f"  {description}")
    if allow_back:
        _print_back_hint()
    if default:
        prompt = f"  {DIM}[default: {default}]{RESET} {GREEN}>{RESET} "
    else:
        prompt = f"  {GREEN}>{RESET} "
    value = input(prompt).strip()
    if allow_back and value == BACK_TOKEN:
        return GO_BACK
    return value if value else default


# ---------------------------------------------------------------------------
# prompt_number
# ---------------------------------------------------------------------------


def prompt_number(
    label: str,
    description: str = "",
    default: float | int | None = None,
    number_type: type = int,
    allow_back: bool = False,
) -> float | int | _GoBack:
    """Prompt for a numeric value with validation.

    Args:
        label: The config item name.
        description: One-line explanation.
        default: Default value; returned on empty Enter.  None means required.
        number_type: ``int`` or ``float``.
        allow_back: If True, typing ``<`` returns :data:`GO_BACK`.

    Returns:
        The parsed number, or :data:`GO_BACK` if the user asked to step back.
    """
    print()
    print(f"{BOLD}{label}{RESET}")
    if description:
        print(f"  {description}")
    if allow_back:
        _print_back_hint()

    while True:
        if default is not None:
            prompt = f"  {DIM}[default: {default}]{RESET} {GREEN}>{RESET} "
        else:
            prompt = f"  {GREEN}>{RESET} "
        raw = input(prompt).strip()
        if allow_back and raw == BACK_TOKEN:
            return GO_BACK
        if not raw and default is not None:
            return default
        try:
            return number_type(raw)
        except (ValueError, TypeError):
            type_name = "integer" if number_type is int else "number"
            print(f"  {YELLOW}Please enter a valid {type_name}.{RESET}")


# ---------------------------------------------------------------------------
# prompt_bool
# ---------------------------------------------------------------------------


def prompt_bool(
    label: str,
    description: str = "",
    default: bool = True,
    allow_back: bool = False,
) -> bool | _GoBack:
    """Prompt for a yes/no confirmation.

    Shows ``[Y/n]`` or ``[y/N]`` depending on the default.
    Enter alone accepts the default.

    Args:
        label: The config item name.
        description: One-line explanation.
        default: Default value when user presses Enter.
        allow_back: If True, typing ``<`` returns :data:`GO_BACK`.

    Returns:
        True for yes, False for no, or :data:`GO_BACK` if the user asked to
        step back.
    """
    print()
    print(f"{BOLD}{label}{RESET}")
    if description:
        print(f"  {description}")
    if allow_back:
        _print_back_hint()

    hint = "[default: Y] (Y/n)" if default else "[default: N] (y/N)"
    while True:
        raw = input(f"  {DIM}{hint}{RESET} {GREEN}>{RESET} ").strip().lower()
        if allow_back and raw == BACK_TOKEN:
            return GO_BACK
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(f"  {YELLOW}Please enter Y or N.{RESET}")


# ---------------------------------------------------------------------------
# prompt_choice — arrow-key selection
# ---------------------------------------------------------------------------


def _read_key() -> str:
    """Read a single keypress from stdin in raw mode.

    Returns:
        ``"up"``, ``"down"``, ``"enter"``, or the literal character.
    """
    # Standard
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Raw mode disables signal generation, so translate Ctrl-C / Ctrl-D
        # into the usual exceptions the caller already expects.
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        if ch == "\r" or ch == "\n":
            return "enter"
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "escape"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _render_choices(
    choices: list[tuple[str, str]],
    selected: int,
) -> list[str]:
    """Build display lines for the choice selector."""
    lines: list[str] = []
    for i, (value, desc) in enumerate(choices):
        if i == selected:
            marker = f"{CYAN}● {value}{RESET}"
        else:
            marker = f"  {value}"
        # Pad value to align descriptions
        padding = max(0, 18 - len(value))
        line = f"  {marker}{' ' * padding} {DIM}{desc}{RESET}"
        lines.append(line)
    return lines


def prompt_choice(
    label: str,
    description: str = "",
    choices: list[tuple[str, str]] = [],  # noqa: B006
    default: str = "",
    allow_back: bool = False,
) -> str | _GoBack:
    """Prompt user to select from a list using arrow keys.

    Each choice is a ``(value, description)`` tuple.  The description is
    shown next to the value as a brief explanation.

    Falls back to numbered input on non-TTY stdin.

    Args:
        label: The config item name.
        description: One-line explanation.
        choices: List of ``(value, one_line_description)`` tuples.
        default: Pre-selected value.  Defaults to the first choice.
        allow_back: If True, pressing ``<`` returns :data:`GO_BACK`.

    Returns:
        The selected value string, or :data:`GO_BACK` if the user asked to
        step back.
    """
    if not choices:
        raise ValueError("choices must not be empty")

    # Find default index
    selected = 0
    for i, (val, _desc) in enumerate(choices):
        if val == default:
            selected = i
            break

    print()
    print(f"{BOLD}{label}{RESET}")
    if description:
        print(f"  {description}")

    # Non-TTY fallback: numbered list
    if not _is_tty():
        return _prompt_choice_fallback(choices, selected, allow_back)

    nav = "Use ↑↓ to navigate, Enter to select"
    nav += f", {BACK_TOKEN} to go back." if allow_back else "."
    print(f"  {DIM}{nav}{RESET}")
    print()

    # Initial render
    lines = _render_choices(choices, selected)
    for line in lines:
        print(line)

    while True:
        key = _read_key()
        if key == "up":
            selected = (selected - 1) % len(choices)
        elif key == "down":
            selected = (selected + 1) % len(choices)
        elif key == "enter" or (key == BACK_TOKEN and allow_back):
            # Clear and re-render final state
            num_lines = len(choices)
            sys.stdout.write(f"\r{CURSOR_UP * num_lines}")
            lines = _render_choices(choices, selected)
            for line in lines:
                print(f"{CLEAR_LINE}{line}")
            return GO_BACK if key == BACK_TOKEN else choices[selected][0]
        else:
            continue

        # Redraw
        num_lines = len(choices)
        sys.stdout.write(f"\r{CURSOR_UP * num_lines}")
        lines = _render_choices(choices, selected)
        for line in lines:
            print(f"{CLEAR_LINE}{line}")


def _prompt_choice_fallback(
    choices: list[tuple[str, str]],
    default_index: int,
    allow_back: bool = False,
) -> str | _GoBack:
    """Numbered fallback for non-TTY environments."""
    for i, (val, desc) in enumerate(choices):
        marker = "*" if i == default_index else " "
        print(f"  {marker} {i + 1}) {val}  — {desc}")
    if allow_back:
        _print_back_hint()
    while True:
        raw = input(f"  [default: {default_index + 1}] {GREEN}>{RESET} ").strip()
        if allow_back and raw == BACK_TOKEN:
            return GO_BACK
        if not raw:
            return choices[default_index][0]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        except ValueError:
            pass
        print(f"  {YELLOW}Enter a number 1-{len(choices)}.{RESET}")
