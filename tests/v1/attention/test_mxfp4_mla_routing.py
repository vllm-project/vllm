# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prove which read kernel GLM-5.3-Flash actually reaches.

This exists because the plan for this work asserted that the shipping tree
routes prefill and decode to *separate* kernels, and that
``_validate_dsv4_sparse_dims`` accepting ``(512, 0)`` means the decode kernels
are live for NoPE -- implying three kernels needed an MXFP4 branch.

That is wrong, and reading the code is weaker evidence than exercising the
predicate. ``_use_rocm_sparse_triton`` guards a single
``rocm_sparse_attn_prefill`` call, so prefill and single-token decode share
``_sparse_attn_prefill_ragged_kernel``. The
``_sparse_attn_decode_*`` kernels address a 576-byte ``fp8_ds_mla`` paged row
and are reached only from the DeepSeek-V4 model path.

CPU-only: these are predicate and call-graph facts, not kernel behaviour.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pathlib

# Resolve the sources through the imported package rather than the working
# directory. A cwd-relative path silently inspects whichever tree the test was
# launched from -- or fails outright elsewhere -- while these assertions are
# about the code that actually gets imported.
_VLLM_SPEC = importlib.util.find_spec("vllm")
assert _VLLM_SPEC is not None and _VLLM_SPEC.origin is not None
VLLM_ROOT = pathlib.Path(_VLLM_SPEC.origin).parent
BACKEND = VLLM_ROOT / "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
OPS = VLLM_ROOT / "v1/attention/ops/rocm_aiter_mla_sparse.py"

# GLM-5.3-Flash: kv_lora_rank=512, qk_rope_head_dim=0, so head_size == 512.
HEAD_SIZE = 512
KV_LORA_RANK = 512


def _route(kv_cache_dtype, head_size=HEAD_SIZE, **kw) -> bool:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        _use_rocm_sparse_triton,
    )

    meta = dict(num_prefills=0, num_decodes=0, num_decode_tokens=0, max_query_len=1)
    meta.update(kw)
    return _use_rocm_sparse_triton(
        kv_cache_dtype=kv_cache_dtype,
        head_size=head_size,
        kv_lora_rank=KV_LORA_RANK,
        **meta,
    )


DECODE = dict(num_decodes=8, num_decode_tokens=8, max_query_len=1)
PREFILL = dict(num_prefills=2, max_query_len=128)


def test_mxfp4_decode_and_prefill_take_the_triton_route():
    for step in (DECODE, PREFILL):
        assert _route("mxfp4_mla", **step) is True
        assert _route("auto", **step) is True


def test_rope_bearing_geometry_is_refused():
    """DeepSeek shapes (576 = 512 + 64) must not take this route."""
    assert _route("mxfp4_mla", head_size=576, **DECODE) is False
    assert _route("auto", head_size=576, **PREFILL) is False


def test_both_predicates_feed_the_prefill_op_only():
    """The load-bearing claim: one call site, and it is the prefill op.

    Parse the impl's forward and assert the branch guarded by the predicate
    calls ``rocm_sparse_attn_prefill`` and never
    ``rocm_sparse_attn_decode``.
    """
    src = BACKEND.read_text()
    tree = ast.parse(src)

    guarded_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_names = {
            n.func.id
            for n in ast.walk(node.test)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "_use_rocm_sparse_triton" not in test_names:
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                guarded_calls.add(call.func.id)

    assert "rocm_sparse_attn_prefill" in guarded_calls, guarded_calls
    assert "rocm_sparse_attn_decode" not in guarded_calls, (
        "the decode op is reachable from the predicate branch; the decode "
        "kernels would then need an MXFP4 branch too"
    )


def test_every_forward_mla_return_carries_an_lse_slot():
    """forward_impl unpacks ``attn_out, lse``; a bare tensor fails at startup.

    The native mxfp4 branch returns early, so it must honour the same
    ``(output, lse)`` contract as the fall-through path.
    """
    tree = ast.parse(BACKEND.read_text())
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_mla"
    )
    returns = [n for n in ast.walk(forward) if isinstance(n, ast.Return)]
    assert len(returns) >= 3, len(returns)
    for ret in returns:
        assert isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 2, (
            f"line {ret.lineno}: _forward_mla must return (output, lse)"
        )


def test_decode_op_is_not_referenced_by_this_backend_at_all():
    src = BACKEND.read_text()
    assert "rocm_sparse_attn_decode" not in src, (
        "this backend references the decode op; re-check whether GLM-5.3-Flash "
        "can reach the _sparse_attn_decode_* kernels"
    )


def test_decode_kernels_belong_to_deepseek_v4():
    """Locate the decode op's only caller, to show whose path it is."""
    callers = [
        p
        for p in VLLM_ROOT.rglob("*.py")
        if "rocm_sparse_attn_decode(" in p.read_text(errors="ignore") and p != OPS
    ]
    assert callers, "expected at least one caller of the decode op"
    assert all("deepseek_v4" in p.as_posix() for p in callers), [
        p.as_posix() for p in callers
    ]


def test_decode_kernels_still_assume_a_576_byte_row():
    """Why they cannot serve a 272-byte MXFP4 row even if reached.

    They address ``pos_in_block * 576``, the fp8_ds_mla paged layout, rather
    than the flat per-slot stride the prefill kernel uses.
    """
    src = OPS.read_text()
    assert "pos_in_block * 576" in src


def test_prefill_kernel_has_the_mxfp4_branch():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    body = inspect.getsource(_rocm_sparse_attn_prefill_ragged_triton)
    assert "kv_is_mxfp4" in body
    assert "KV_IS_MXFP4" in body
    # And the launcher must refuse a narrow query.
    assert "bf16 query" in body


# The tests above establish that the decode path is unreachable *for this
# backend*. It is still reachable from DeepSeek-V4, and the hazard there is that
# an MXFP4 cache is uint8 exactly like fp8_ds_mla, so the decode op's dtype
# asserts pass and the row is silently reinterpreted. These cover the guard that
# turns that silent misread into an error.


def test_guard_rejects_a_packed_mxfp4_cache():
    import pytest
    import torch

    from vllm.v1.attention.ops.mxfp4_mla import row_bytes
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _reject_mxfp4_cache

    # 272 bytes for a 512-wide latent: 256 E2M1 + 16 E8M0.
    assert row_bytes(KV_LORA_RANK) == 272
    cache = torch.zeros((4, 64, row_bytes(KV_LORA_RANK)), dtype=torch.uint8)

    with pytest.raises(NotImplementedError, match="packed MXFP4"):
        _reject_mxfp4_cache(cache, KV_LORA_RANK, "extra")


def test_guard_allows_an_fp8_ds_mla_cache():
    """The guard must not break the path it is defending.

    fp8_ds_mla is also uint8, so a guard keyed on dtype would reject this and
    take out DeepSeek-V4. Only the MXFP4 row width may trip it.
    """
    import torch

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _reject_mxfp4_cache

    # The fp8_ds_mla paged row the decode kernels actually address.
    for row in (576, 656):
        cache = torch.zeros((4, 64, row), dtype=torch.uint8)
        _reject_mxfp4_cache(cache, KV_LORA_RANK, "extra")

    # A bf16 cache is not uint8 and must pass through untouched.
    _reject_mxfp4_cache(
        torch.zeros((4, 64, KV_LORA_RANK), dtype=torch.bfloat16),
        KV_LORA_RANK,
        "extra",
    )


def test_decode_op_installs_the_guard_on_both_caches():
    """Both the SWA and the extra cache must be checked, not just one."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode

    body = inspect.getsource(rocm_sparse_attn_decode)
    assert body.count("_reject_mxfp4_cache") == 2
    assert "_reject_mxfp4_cache(swa_k_cache" in body
