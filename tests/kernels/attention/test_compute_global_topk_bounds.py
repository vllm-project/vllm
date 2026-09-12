# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: compute_global_topk_indices_and_lens dual-axis bounds.

Drives the shipped DeepSeek-V4 / V4.1 kernels. Invalid req_idx or block_idx
must become -1, be excluded from topk_lens, and never alias onto another
request's block-table row.
"""

from __future__ import annotations

import pytest
import torch

BLOCK_SIZE = 64


def _import_compute(version: str):
    """Import shipped kernel, bypassing heavy model package __init__ when needed."""
    import importlib
    import importlib.util
    import os
    import sys
    import types

    module_name = f"vllm.models.{version}.common.ops.cache_utils"
    if module_name in sys.modules and hasattr(
        sys.modules[module_name], "compute_global_topk_indices_and_lens"
    ):
        return sys.modules[module_name].compute_global_topk_indices_and_lens

    try:
        mod = importlib.import_module(module_name)
        return mod.compute_global_topk_indices_and_lens
    except Exception:
        # Fresh source checkout without compiled extensions / heavy inits.
        import vllm

        root = os.path.join(os.path.dirname(vllm.__file__), "models")
        version_root = os.path.join(root, version)

        def ensure_ns(fullname, path):
            if fullname in sys.modules and hasattr(sys.modules[fullname], "__path__"):
                return
            m = types.ModuleType(fullname)
            m.__path__ = [path]
            m.__file__ = os.path.join(path, "__init__.py")
            m.__package__ = fullname
            sys.modules[fullname] = m

        ensure_ns("vllm.models", root)
        ensure_ns(f"vllm.models.{version}", version_root)
        ensure_ns(f"vllm.models.{version}.common", os.path.join(version_root, "common"))
        ensure_ns(
            f"vllm.models.{version}.common.ops",
            os.path.join(version_root, "common", "ops"),
        )
        path = os.path.join(version_root, "common", "ops", "cache_utils.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.compute_global_topk_indices_and_lens


def _run(compute, topk_indices, token_to_req, block_table, is_valid):
    out, lens = compute(
        topk_indices, token_to_req, block_table, BLOCK_SIZE, is_valid
    )
    if topk_indices.is_cuda:
        torch.cuda.synchronize()
    return out, lens


@pytest.fixture(params=["deepseek_v4", "deepseek_v4_1"])
def compute(request):
    return _import_compute(request.param)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestComputeGlobalTopkBounds:
    def test_exact_valid_metadata_unchanged(self, compute):
        """Cases 1 + 11: valid entries produce expected slots; unchanged."""
        device = torch.device("cuda")
        topk = torch.tensor([[0, 64, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == 10 * BLOCK_SIZE + 0
        assert out[0, 1].item() == 11 * BLOCK_SIZE + 0
        assert out[0, 2].item() == -1
        assert lens[0].item() == 2

    def test_req_idx_equal_rows(self, compute):
        """Case 3: req_idx == rows → invalid."""
        device = torch.device("cuda")
        topk = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([1], dtype=torch.int32, device=device)  # rows=1
        block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0].tolist() == [-1, -1, -1, -1]
        assert lens[0].item() == 0

    def test_req_idx_greater_than_rows(self, compute):
        """Case 4: req_idx > rows → invalid."""
        device = torch.device("cuda")
        topk = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([5], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert all(v == -1 for v in out[0].tolist())
        assert lens[0].item() == 0

    def test_req_idx_negative(self, compute):
        """Case 5: req_idx < 0 → invalid."""
        device = torch.device("cuda")
        topk = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([-1], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert all(v == -1 for v in out[0].tolist())
        assert lens[0].item() == 0

    def test_block_idx_equal_stride(self, compute):
        """Case 6: block_idx == logical cols → invalid."""
        device = torch.device("cuda")
        # logical cols=2; local_idx=2*64 → block_indices=2 == cols
        topk = torch.tensor([[128, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == -1
        assert lens[0].item() == 0

    def test_block_idx_greater_than_stride(self, compute):
        """Case 7: block_idx > logical cols → invalid."""
        device = torch.device("cuda")
        topk = torch.tensor([[256, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == -1
        assert lens[0].item() == 0

    def test_block_idx_negative_via_neg_local(self, compute):
        """Case 8: negative local_idx (block_idx path) → invalid sentinel."""
        device = torch.device("cuda")
        topk = torch.tensor([[-5, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == -1
        assert lens[0].item() == 0

    def test_mixed_valid_invalid_and_topk_lens(self, compute):
        """Cases 9+10: mixed row OOB; invalid → -1; excluded from topk_lens."""
        device = torch.device("cuda")
        # token0 req0 valid; token1 req1 OOB against rows=1
        topk = torch.tensor(
            [[0, -1, -1, -1], [0, -1, -1, -1]], dtype=torch.int32, device=device
        )
        token_to_req = torch.tensor([0, 1], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True, True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == 10 * BLOCK_SIZE
        assert lens[0].item() == 1
        assert all(v == -1 for v in out[1].tolist())
        assert lens[1].item() == 0
        # Must not alias token1 onto row 0 (would yield 640 and lens=1).
        assert out[1, 0].item() != 10 * BLOCK_SIZE

    def test_column_oob_mixed_with_valid(self, compute):
        """Column OOB entry discarded; valid sibling preserved."""
        device = torch.device("cuda")
        topk = torch.tensor([[0, 128, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == 10 * BLOCK_SIZE
        assert out[0, 1].item() == -1
        assert lens[0].item() == 1

    def test_cg_padded_block_table_rows(self, compute):
        """Case 12: padded block_table rows; only real req indices gather."""
        device = torch.device("cuda")
        topk = torch.tensor(
            [[0, -1, -1, -1], [64, -1, -1, -1]], dtype=torch.int32, device=device
        )
        token_to_req = torch.tensor([0, 1], dtype=torch.int32, device=device)
        # Padded to 2 rows for CG; row1 is unused padding content.
        block_table = torch.tensor(
            [[10, 11, 12, 13], [90, 91, 92, 93]], dtype=torch.int32, device=device
        )
        is_valid = torch.tensor([True, True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == 10 * BLOCK_SIZE
        assert out[1, 0].item() == 91 * BLOCK_SIZE
        assert lens.tolist() == [1, 1]

    def test_normal_prefill_multi_req(self, compute):
        """Case 13: normal prefill multi-request gather."""
        device = torch.device("cuda")
        topk = torch.tensor(
            [[0, -1, -1, -1], [0, -1, -1, -1], [64, -1, -1, -1]],
            dtype=torch.int32,
            device=device,
        )
        token_to_req = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
        block_table = torch.tensor(
            [[10, 11], [20, 21]], dtype=torch.int32, device=device
        )
        is_valid = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == 10 * BLOCK_SIZE
        assert out[1, 0].item() == 20 * BLOCK_SIZE
        assert out[2, 0].item() == 21 * BLOCK_SIZE
        assert lens.tolist() == [1, 1, 1]

    def test_invalid_before_valid_compacts_left(self, compute):
        """Invalid-before-valid entries pack left; survivor preserved."""
        device = torch.device("cuda")
        # block col0=10, col1=11; local 8 => block_indices=0? wait block_size=64
        # Use block_size=64: local 8 is still col0. Need invalid column then valid.
        # local 128 => col2 OOB for cols=2; local 64 => col1 valid.
        topk = torch.tensor([[128, 64, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[10, 11]], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        # Survivor (col1 -> 11*64) must pack left.
        assert out[0, 0].item() == 11 * BLOCK_SIZE
        assert out[0, 1].item() == -1
        assert lens[0].item() == 1

    def test_logical_cols_not_stride_boundary(self, compute):
        """shape[1] is the column bound, not row stride."""
        device = torch.device("cuda")
        backing = torch.tensor([[10, 11, 12]], dtype=torch.int32, device=device)
        block_table = backing[:, :2]
        assert block_table.shape == (1, 2)
        assert block_table.stride(0) == 3
        # local 8 with block_size=4 would be used in #55692; here BLOCK_SIZE=64
        # local 128 => block_indices=2, which is < stride(3) but == cols boundary.
        topk = torch.tensor([[128, -1, -1, -1]], dtype=torch.int32, device=device)
        token_to_req = torch.tensor([0], dtype=torch.int32, device=device)
        is_valid = torch.tensor([True], dtype=torch.bool, device=device)
        out, lens = _run(compute, topk, token_to_req, block_table, is_valid)
        assert out[0, 0].item() == -1
        assert lens[0].item() == 0

