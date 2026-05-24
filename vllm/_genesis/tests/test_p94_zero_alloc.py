# SPDX-License-Identifier: Apache-2.0
"""TDD test for P94 — backport of vllm#41043
prepare_next_token_ids_padded zero-alloc.

Eliminates GPU->CPU .tolist() sync, list-comp Python objects, and
np.array(...) allocation in the spec-decode hot path.
"""
from __future__ import annotations

import re


from vllm._genesis.wiring.spec_decode.patch_94_spec_decode_zero_alloc import (
    P94_OLD,
    P94_NEW,
    GENESIS_P94_MARKER,
)


def test_p94_old_has_anti_pattern_tolist():
    """The old code we replace must contain the .tolist() sync."""
    pat = re.compile(r"\(.*num_tokens_no_spec.*\)\.tolist\(\)")
    assert pat.search(P94_OLD), (
        "P94_OLD anchor missing .tolist() pattern — anchor obsolete"
    )


def test_p94_new_has_no_tolist_in_executable_code():
    """The replacement must NOT contain .tolist() in actual code (only OK
    inside Python string literals or comments).
    """
    # Strip Python comments (lines starting with # after optional indent
    # inside the emit string)
    code_lines = [
        line for line in P94_NEW.split("\\n")
        if not line.lstrip().startswith("#") and not line.startswith('    "        #')
    ]
    code_only = "\\n".join(code_lines)
    assert ".tolist()" not in code_only, (
        f"P94_NEW still contains .tolist() in executable code:\n{code_only}"
    )


def test_p94_new_has_no_np_array_alloc():
    """The replacement must NOT contain `np.array(` in executable code
    (only OK inside comments).
    """
    code_lines = [
        line for line in P94_NEW.split("\\n")
        if not line.lstrip().startswith("#") and not line.startswith('    "        #')
    ]
    code_only = "\\n".join(code_lines)
    assert "np.array(" not in code_only, (
        f"P94_NEW still contains np.array(...) allocation in executable "
        f"code:\n{code_only}"
    )


def test_p94_new_has_in_place_loop():
    """The replacement must use the direct-loop pattern."""
    pat_for = re.compile(r"for i in range\(num_reqs\):")
    pat_assign = re.compile(r"self\.backup_next_token_ids\.np\[i\] = requests\[")
    assert pat_for.search(P94_NEW), "P94_NEW missing `for i in range(num_reqs)`"
    assert pat_assign.search(P94_NEW), (
        "P94_NEW missing in-place assignment `self.backup_next_token_ids.np[i] = requests[...]`"
    )


def test_p94_marker_versioned():
    """The marker should embed a version so re-applies don't no-op when we
    bump the patch."""
    assert "v7.62" in GENESIS_P94_MARKER, (
        f"P94 marker {GENESIS_P94_MARKER!r} should embed v7.62.x"
    )
    assert "vllm#41043" in GENESIS_P94_MARKER, (
        "P94 marker should reference upstream PR for drift detection"
    )
