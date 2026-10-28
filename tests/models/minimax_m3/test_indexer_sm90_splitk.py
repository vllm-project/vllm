# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU correctness for the MiniMax M3 indexer SM90 split-K score path.

Invariant, checked bit-exactly on-device: the split-K score path equals the
split_k=1 path inside every row's causal range (splits write disjoint block
columns).

Standalone runs (no compiled vllm needed) load the kernels from the source
tree via stub modules; with a vllm install present the package import wins.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

BLOCK, HEAD_DIM, TOPK, INIT, LOCAL = 128, 128, 16, 0, 1


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _kernels():
    try:
        from vllm.models.minimax_m3.common.ops import index_topk

        return (index_topk,)
    except Exception:
        env_repo = os.environ.get("M3_REPO_ROOT")
        repo = Path(env_repo) if env_repo else Path(__file__).resolve().parents[3]
        base = repo / "vllm/models/minimax_m3/common"
        vllm = types.ModuleType("vllm")
        platforms = types.ModuleType("vllm.platforms")

        class _P:
            @staticmethod
            def is_cuda():
                return torch.cuda.is_available()

            @staticmethod
            def is_device_capability(cap):
                return False

            @staticmethod
            def is_device_capability_family(family):
                return (family // 10) == 9

            @staticmethod
            def is_arch_support_pdl():
                return False

        platforms.current_platform = _P
        import triton
        import triton.language as tl

        tu = types.ModuleType("vllm.triton_utils")
        tu.triton, tu.tl = triton, tl
        utils = types.ModuleType("vllm.utils")
        math_utils = types.ModuleType("vllm.utils.math_utils")
        math_utils.round_up = lambda x, m: (x + m - 1) // m * m
        vllm.platforms, vllm.triton_utils, vllm.utils = platforms, tu, utils
        sys.modules.update(
            {"vllm": vllm, "vllm.platforms": platforms, "vllm.triton_utils": tu,
             "vllm.utils": utils, "vllm.utils.math_utils": math_utils}
        )
        return (_load("m3_index_topk", base / "ops/index_topk.py"),)


(kernels,) = _kernels()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_platform_family_gate_semantics() -> None:
    """Regression guard: the SM90 split-K gate must use family(90).

    is_device_capability_family compares (cap.to_int() // 10) with
    (argument // 10), so family(9) means family 0.x and silently disabled
    split-K on every SM9.0 deployment while all stubbed tests stayed green.
    """
    from vllm.platforms import current_platform

    assert not current_platform.is_device_capability_family(9)
    if current_platform.is_cuda() and current_platform.get_device_capability().major == 9:
        assert current_platform.is_device_capability_family(90)



def _batch(device, seed):
    g = torch.Generator().manual_seed(seed)
    query_lens = [1024, 512, 512]
    contexts = [32768, 65536, 4096]
    seq_lens = [c + q for c, q in zip(contexts, query_lens)]
    rows, max_seq = sum(query_lens), max(
        c + q for c, q in zip(contexts, query_lens)
    )
    pages = (max_seq + BLOCK - 1) // BLOCK
    bt = torch.empty(len(query_lens), pages, dtype=torch.int32)
    pool = torch.arange(pages * len(query_lens))
    pool = pool[torch.randperm(pages * len(query_lens), generator=g)]
    for i in range(len(query_lens)):
        bt[i] = pool[i * pages : (i + 1) * pages]
    ik = torch.randn(pages * len(query_lens), BLOCK, HEAD_DIM,
                     generator=g).to(torch.bfloat16).to(device)
    iq = torch.randn(rows, 1, HEAD_DIM, generator=g).to(torch.bfloat16).to(device)
    cu = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(query_lens), 0)
    return (
        query_lens, contexts, seq_lens, rows, max_seq,
        bt.to(device), ik, iq, cu.to(device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(contexts, dtype=torch.int32, device=device),
    )


def _score(iq, ik, bt, cu, seq_t, ctx_t, rows, max_seq, max_q, split_k):
    max_block = (max_seq + BLOCK - 1) // BLOCK
    stride = (max_block + 15) // 16 * 16
    score = torch.full((1, rows, stride), float("nan"),
                       dtype=torch.float32, device=iq.device)
    n_tiles = (max_q + 63) // 64
    kernels._index_block_score_kernel[(n_tiles, cu.numel() - 1, split_k)](
        iq, ik, score, bt, cu, seq_t, ctx_t,
        1, HEAD_DIM,
        *iq.stride(), *ik.stride(), *score.stride(), bt.stride(0),
        BLOCK_SIZE_Q=64, BLOCK_SIZE_K=BLOCK, USE_SPLIT_K=split_k > 1,
    )
    torch.cuda.synchronize()
    return score


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("split_k", [4, 16, 33, 64, 128])
def test_sm90_split_k_bit_identical(split_k: int) -> None:
    """Split-K writes disjoint block columns: scores match split_k=1 exactly
    inside each row's causal range and leave no NaN there."""
    (query_lens, contexts, _, rows, max_seq, bt, ik, iq, cu,
     seq_t, ctx_t) = _batch(torch.device("cuda"), 7)
    ref = _score(iq, ik, bt, cu, seq_t, ctx_t, rows, max_seq,
                 max(query_lens), 1)
    got = _score(iq, ik, bt, cu, seq_t, ctx_t, rows, max_seq,
                 max(query_lens), split_k)
    for r, (c, q) in enumerate(zip(contexts, query_lens)):
        start, end = int(cu[r]), int(cu[r + 1])
        row = torch.arange(end - start, device=iq.device)
        vis = (c + row + BLOCK) // BLOCK
        m = torch.arange(ref.shape[2], device=iq.device)[None, :] < vis[:, None]
        assert torch.equal(ref[0, start:end][m], got[0, start:end][m])
        assert not torch.isnan(got[0, start:end][m]).any()
