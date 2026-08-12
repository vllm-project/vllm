# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort scan of shell scripts reached from live step commands.

Only scripts a live step invokes get read, so a script nothing runs cannot
manufacture targets. Inside a body we look for the same test shapes the YAML
parser knows (pytest lines, python drivers, nested bash) and follow cd linearly.
Everything else is ignored: these scripts are mostly docker and setup logic.

Giving up is always recorded as a dangling target, never a silent return.
Targets found here feed state.invoked, so a script we could not read would make
the tests it runs look like nothing invokes them.
"""

from __future__ import annotations

import regex as re

# How many scripts deep a `bash x.sh` chain is followed. Not the cycle guard:
# scripts_seen already scans each script once per step. Deepest chain today is 2.
MAX_DEPTH = 3

PYTEST_LINE_RE = re.compile(r"(?:^|[;&(\s])(?:python3?\s+-m\s+)?pytest\s+(.*)")
# Any .sh path literal, not just `bash x.sh`: scripts often assign the script
# to a var first (SCRIPT=".../x.sh"; bash "$SCRIPT"), which bash-only misses.
SH_PATH_RE = re.compile(r"[\w./-]+\.sh\b")
# Tolerates quotes and a "${VAR}/" prefix (PROXY_CMD="python3 ${ROOT}/x.py").
# resolve_path strips the prefix, so the driver still resolves to a real file.
PYTHON_LINE_RE = re.compile(r'(?:^|[;&(\s])python3?\s+"?(\S+?\.py)\b')
CD_RE = re.compile(r"(?:^|[;&(\s])cd\s+(\S+)")
SEP_RE = re.compile(r"&&|\|\||;")


def scan_script(script: str, parser, depth: int = 0) -> None:
    """Feed a script body's test shapes into the invoking step's CommandParser.

    Targets found here land in that step's StepTargets tagged via=<script>. A
    script's runtime cwd is not knowable statically, so resolve_path falls back
    from the tracked cwd to the repo root.
    """
    if depth >= MAX_DEPTH:
        parser.out.dangling.append(script)
        return
    path = parser.repo / script
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        parser.out.dangling.append(script)
        return
    parser.out.haystack += "\n" + text
    saved_cwd = parser.cwd
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = CD_RE.search(line)
        if m and "$" not in m.group(1):
            parser.chdir(m.group(1))
        # Split so every chained `pytest` is seen. The .sh loop below stays on
        # the whole line, so `pytest a && bash next.sh` still carries next.sh.
        for segment in SEP_RE.split(line):
            m = PYTEST_LINE_RE.search(segment)
            if m:
                args = _tokenize(m.group(1))
                if args is not None:
                    parser.parse_pytest(args, via=script)
        for token in SH_PATH_RE.findall(line):
            nested = parser.resolve_path(token)
            if nested is None:
                # `"$(dirname "$0")/x.sh"`: fall back to a sibling of this script.
                sibling = f"{script.rsplit('/', 1)[0]}/{token.rsplit('/', 1)[-1]}"
                if (parser.repo / sibling).is_file():
                    nested = sibling
            if nested and nested not in parser.out.scripts_seen:
                parser.out.scripts_seen.append(nested)
                scan_script(nested, parser, depth + 1)
        # Runs even when the line also named a script: `bash setup.sh && python3
        # tests/driver.py` still has a real driver target.
        m = PYTHON_LINE_RE.search(line)
        if m:
            resolved = parser.resolve_path(m.group(1))
            if resolved:
                parser.out.add_target(resolved, "script", via=script)
    parser.cwd = saved_cwd


def _tokenize(argstr: str) -> list[str] | None:
    import shlex

    # Cut at separators, pipes and redirects so a trailing `&& bash next.sh` is
    # not read as a pytest positional.
    argstr = re.split(r"&&|;|\||2>|>", argstr)[0].rstrip("\\ ")
    try:
        return shlex.split(argstr)
    except ValueError:
        return None
