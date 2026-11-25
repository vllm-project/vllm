# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def _run_compiled(launcher, *args) -> None:
    """Reuse the compiled callable of a factory-style host launcher."""
    import flydsl.compiler as flyc
    from flydsl.utils import env

    if env.compile.compile_only:
        flyc.compile(launcher, *args)
        return

    compiled = getattr(launcher, "_cf", None)
    if compiled is not None:
        compiled(*args)
        return

    # flyc.compile executes once; do not launch its result again on this call.
    launcher._cf = flyc.compile(launcher, *args)
