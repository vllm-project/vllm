# SPDX-License-Identifier: Apache-2.0
"""Startup banner shown when LMCache starts serving.

The banner is printed at most once per process, to ``stderr``, by the
``lmcache`` CLI and by the vLLM connector integrations (scheduler role
only, so tensor-parallel deployments print a single banner). Setting the
``LMCACHE_DISABLE_BANNER=1`` environment variable suppresses it.
"""

# Standard
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    # Standard
    from typing import TextIO

try:
    # First Party
    from lmcache import _version

    _LMCACHE_VERSION = getattr(_version, "__version__", "unknown")
    _LMCACHE_COMMIT = getattr(_version, "__commit_id__", "")
except ImportError:  # pragma: no cover - version file is generated at build time
    _LMCACHE_VERSION = "unknown"
    _LMCACHE_COMMIT = ""

DISABLE_BANNER_ENV = "LMCACHE_DISABLE_BANNER"

LMCACHE_WEBSITE = "https://lmcache.ai/"
LMCACHE_RECIPES = "https://docs.lmcache.ai/recipes"
LMCACHE_LINKEDIN = "https://www.linkedin.com/company/lmcache-lab"

# Solarized palette, 24-bit ANSI escapes (TTY only): "LM" in bold italic
# orange (#cb4b16), "Cache" in cyan (#2aa198).
_LM_STYLE = "\x1b[1;3;38;2;203;75;22m"
_CACHE_STYLE = "\x1b[38;2;42;161;152m"
_DIM_STYLE = "\x1b[2m"
_RESET = "\x1b[0m"

# Figlet "standard" font, split into the two color groups.
_LM_ART = (
    " _     __  __ ",
    "| |   |  \\/  |",
    "| |   | |\\/| |",
    "| |___| |  | |",
    "|_____|_|  |_|",
)
_CACHE_ART = (
    "  ____           _          ",
    " / ___|__ _  ___| |__   ___ ",
    "| |   / _` |/ __| '_ \\ / _ \\",
    "| |__| (_| | (__| | | |  __/",
    " \\____\\__,_|\\___|_| |_|\\___|",
)
_RIGHT_TEXT_GAP = "     "

_banner_printed = False


def _banner_disabled() -> bool:
    """Return whether ``LMCACHE_DISABLE_BANNER`` is set to a truthy value."""
    return os.getenv(DISABLE_BANNER_ENV, "").strip().lower() in ("1", "true", "yes")


def _render_banner(colored: bool) -> str:
    """Render the banner text.

    Args:
        colored: Whether to wrap the logo in ANSI color escapes.

    Returns:
        The multi-line banner: the LMCache logo with the version (and
        commit id when available), website, recipes, and LinkedIn links
        on its right, and a final line describing the
        ``LMCACHE_DISABLE_BANNER`` opt-out. A blank line surrounds the
        banner on each side to set it apart from adjacent log output.
    """
    lm_style = _LM_STYLE if colored else ""
    cache_style = _CACHE_STYLE if colored else ""
    dim_style = _DIM_STYLE if colored else ""
    reset = _RESET if colored else ""

    version = f"LMCache v{_LMCACHE_VERSION}"
    if _LMCACHE_COMMIT:
        version += f" ({_LMCACHE_COMMIT[:9]})"
    right_text = {
        1: version,
        2: f"Website:  {LMCACHE_WEBSITE}",
        3: f"Recipes:  {LMCACHE_RECIPES}",
        4: f"LinkedIn: {LMCACHE_LINKEDIN}",
    }
    lines = [""]
    for row, (lm_part, cache_part) in enumerate(zip(_LM_ART, _CACHE_ART, strict=True)):
        line = f"{lm_style}{lm_part}{reset} {cache_style}{cache_part}{reset}"
        if row in right_text:
            line += _RIGHT_TEXT_GAP + right_text[row]
        lines.append(line)
    lines.append(f"{dim_style}Set {DISABLE_BANNER_ENV}=1 to hide this banner.{reset}")
    lines.append("")
    return "\n".join(lines)


def print_banner_once(stream: "TextIO") -> None:
    """Print the LMCache startup banner to ``stream`` at most once.

    The banner shows the LMCache logo, version, and website, followed by
    a hint describing the ``LMCACHE_DISABLE_BANNER`` opt-out. ANSI colors
    are used only when ``stream`` is a TTY. Subsequent calls in the same
    process are no-ops, as are all calls when ``LMCACHE_DISABLE_BANNER``
    is set to ``1``/``true``/``yes``.

    Args:
        stream: Destination text stream. Callers should pass
            ``sys.stderr`` so the banner never interferes with
            machine-readable stdout output.
    """
    global _banner_printed
    if _banner_printed or _banner_disabled():
        return
    _banner_printed = True
    stream.write(_render_banner(stream.isatty()) + "\n")
    stream.flush()
