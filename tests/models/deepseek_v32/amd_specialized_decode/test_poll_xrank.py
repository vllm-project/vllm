# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank poll-expiry flag (``poll_xrank``).

A rank whose bounded wait expires sets its word in rank 0's symmetric buffer, so the
output rank sees every rank's expiries.
"""

import inspect
import itertools
import os

import regex as re

from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.layout import layout


def test_poll_xrank_layout():
    """The flag block is appended after every other symmetric region (no offset
    moves), one int32 per rank; the target address has an 8-byte scratch slot."""
    for S, npes in itertools.product((1, 2, 4, 5, 6, 8, 10, 12), (1, 2, 4, 8)):
        scr, sym = layout(S, 8, npes, 2048)
        old_end = 2 * sym["ffn"]
        assert sym["poll_xrank"] == old_end and sym["poll_xrank"] % 4 == 0
        assert all(
            v <= old_end
            for k, v in sym.items()
            if not k.startswith("_") and k != "poll_xrank"
        )
        assert sym["_bytes"] >= sym["poll_xrank"] + 4 * npes
        assert scr["poll_xrank_addr"] % 8 == 0
        assert scr["_bytes"] >= scr["poll_xrank_addr"] + 8


def test_poll_xrank_ir_fences(tmp_path, monkeypatch, import_kernel):
    """In the TP8 LLVM IR, every expired-wait flag store is followed by a release
    fence before any other store (compile only, needs FlyDSL)."""
    monkeypatch.setenv("FLYDSL_COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    kernel = import_kernel("glm.kernel")
    launch = kernel.build_glm5_monokernel(
        1,
        8,
        8,
        2048,
        launches_per_step=1,
        inter=256,
        poll_limit=1000,
        poll_early_out=True,
    )
    params = inspect.signature(launch._original_func).parameters
    n = len([p for p in params if p != "stream"])
    # pointer arguments, then rank and layer
    launch._compile_only(*[0x1000 * (i + 1) for i in range(n - 2)], 0, 0, stream=0)
    dumps = [
        os.path.join(r, f)
        for r, _, fs in os.walk(tmp_path)
        for f in fs
        if f == "21_llvm_ir.ll"
    ]
    assert dumps
    with open(dumps[0]) as fh:
        ir = fh.read()
    flag = re.compile(r"store atomic i32 1, ptr addrspace\(1\) %\S+ monotonic")
    later_store = re.compile(r"\bstore\b|buffer\.store|\bfence\b")
    sites = 0
    for block in re.split(r"\n(?=\d+:|[A-Za-z_.][\w.]*:)", ir):
        lines = block.splitlines()
        for i, ln in enumerate(lines):
            if flag.search(ln):
                sites += 1
                nxt = [x for x in lines[i + 1 :] if later_store.search(x)]
                assert nxt and re.search(r"fence release\s*$", nxt[0].strip())
    assert sites > 0 and sites == len(re.findall(r"fence release\s*\n", ir))
