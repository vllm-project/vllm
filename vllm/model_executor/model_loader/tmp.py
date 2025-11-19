# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import sys
from types import FrameType, FunctionType, ModuleType
from typing import Any

# Types that we should *print* but absolutely never recurse into.
BLACKLIST_TYPES = (
    FrameType,
    ModuleType,
    FunctionType,
    type(sys),
)


def safe_repr(obj: Any) -> str:
    """Return a safe repr that cannot recurse or throw."""
    try:
        return repr(obj)
    except Exception:
        return f"<unreprable {type(obj).__name__}>"


def print_referrers(obj: Any, max_depth: int = 2):
    """Print all referrers to an object up to a specified depth."""
    visited: set[int] = set()

    def _walk(o: Any, depth: int):
        oid = id(o)
        if oid in visited:
            return
        visited.add(oid)

        if depth > max_depth:
            return

        referrers = gc.get_referrers(o)

        print(len(referrers))
        for ref in referrers:
            if ref is referrers:
                continue  # avoid trivial self-list reference

            indent = "    " * depth
            print(f"{indent}- {type(ref).__name__}: {safe_repr(ref)}")

            # If it's a blacklist type, print it but DO NOT recurse into it
            if isinstance(ref, BLACKLIST_TYPES):
                continue

            # Otherwise, recurse
            _walk(ref, depth + 1)

    print(f"Root object ({type(obj).__name__}): {safe_repr(obj)}")
    _walk(obj, 1)
