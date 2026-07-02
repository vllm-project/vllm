# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Ring Attention correctness.

Verifies that :func:`ring_flash_attn_varlen_func` produces the same
output as standard single-GPU ``flash_attn_varlen_func`` across:
  - Bidirectional and causal attention
  - GQA (num_kv_heads < num_q_heads)
  - Multiple dtypes (bf16, fp16)
  - Multi-request packed batches with different lengths
  - CP sizes 2 and 4

Two run modes:
  1. pytest (uses ray for multi-GPU, compatible with vLLM CI):
       pytest tests/distributed/test_ring_attn.py -v
  2. torchrun (standalone, no ray/pytest):
       torchrun --nproc_per_node=2 tests/distributed/test_ring_attn.py
       torchrun --nproc_per_node=4 tests/distributed/test_ring_attn.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

ATOL = 5e-3
RTOL = 5e-3

# Test configurations: (seq_lens, Hq, Hkv, D, causal, dtype, label)
VARLEN_CONFIGS = [
    # MHA bidirectional
    ([512], 16, 16, 64, False, torch.bfloat16, "single_bidir_MHA"),
    ([256, 256], 16, 16, 64, False, torch.bfloat16, "multi_eq_bidir"),
    ([128, 384, 512], 16, 16, 64, False, torch.bfloat16, "multi_diff_bidir"),
    # MHA causal
    ([512], 16, 16, 64, True, torch.bfloat16, "single_causal_MHA"),
    ([256, 256], 16, 16, 64, True, torch.bfloat16, "multi_eq_causal"),
    ([128, 384, 512], 16, 16, 64, True, torch.bfloat16, "multi_diff_causal"),
    # GQA
    ([512], 32, 8, 128, False, torch.bfloat16, "single_bidir_GQA"),
    ([512], 32, 8, 128, True, torch.bfloat16, "single_causal_GQA"),
    ([128, 384], 32, 8, 128, False, torch.bfloat16, "multi_bidir_GQA"),
    ([128, 384], 32, 8, 128, True, torch.bfloat16, "multi_causal_GQA"),
    # fp16
    ([256, 256], 16, 16, 64, False, torch.float16, "multi_bidir_fp16"),
    ([512], 32, 8, 128, False, torch.float16, "single_GQA_fp16"),
    # Longer sequences (closer to AR prefill)
    ([2048], 32, 8, 128, True, torch.bfloat16, "long_causal_GQA"),
    ([512, 1024], 32, 8, 128, True, torch.bfloat16, "multi_long_causal_GQA"),
]


# =====================================================================
# Helpers
# =====================================================================


def _make_varlen_inputs(
    seq_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    total = sum(seq_lens)
    q = torch.randn(total, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device=device)

    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )

    return q, k, v, cu, max(seq_lens)


def _shard_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: list[int],
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Shard packed tensors: each rank gets a contiguous chunk per request."""
    q_chunks, k_chunks, v_chunks = [], [], []
    local_lens = []

    offset = 0
    for sl in seq_lens:
        chunk_size = sl // world_size
        start = offset + rank * chunk_size
        q_chunks.append(q[start : start + chunk_size])
        k_chunks.append(k[start : start + chunk_size])
        v_chunks.append(v[start : start + chunk_size])
        local_lens.append(chunk_size)
        offset += sl

    q_local = torch.cat(q_chunks, dim=0).contiguous()
    k_local = torch.cat(k_chunks, dim=0).contiguous()
    v_local = torch.cat(v_chunks, dim=0).contiguous()

    cu_local = torch.zeros(len(local_lens) + 1, dtype=torch.int32, device=q.device)
    cu_local[1:] = torch.cumsum(
        torch.tensor(local_lens, dtype=torch.int32, device=q.device), dim=0
    )

    return q_local, k_local, v_local, cu_local, max(local_lens)


def _run_one_config(
    seq_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Run one test config and assert correctness."""
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func as reference_varlen,
    )
    from vllm.v1.attention.ops.ring_attn import ring_flash_attn_varlen_func

    if any(sl % world_size != 0 for sl in seq_lens):
        return  # skip non-divisible configs

    torch.manual_seed(42)
    q, k, v, cu, max_sl = _make_varlen_inputs(
        seq_lens, num_q_heads, num_kv_heads, head_dim, dtype, device
    )

    ref, _, _ = reference_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_sl,
        max_seqlen_k=max_sl,
        causal=causal,
        return_attn_probs=True,
    )

    q_l, k_l, v_l, cu_l, max_l = _shard_varlen(q, k, v, seq_lens, rank, world_size)

    ring = ring_flash_attn_varlen_func(
        q_l,
        k_l,
        v_l,
        cu_seqlens_q=cu_l,
        cu_seqlens_k=cu_l,
        max_seqlen_q=max_l,
        max_seqlen_k=max_l,
        cp_group=dist.group.WORLD,
        causal=causal,
    )

    ref_chunks = []
    offset = 0
    for sl in seq_lens:
        chunk_size = sl // world_size
        start = offset + rank * chunk_size
        ref_chunks.append(ref[start : start + chunk_size])
        offset += sl
    ref_local = torch.cat(ref_chunks, dim=0)

    torch.testing.assert_close(
        ring.float(),
        ref_local.float(),
        atol=ATOL,
        rtol=RTOL,
        msg=f"Ring Attention mismatch on rank {rank}",
    )


# =====================================================================
# Standalone torchrun mode
# =====================================================================


def _run_standalone():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n=== Ring Attention varlen tests (CP={world_size}) ===")

    all_pass = True
    for seq_lens, Hq, Hkv, D, causal, dtype, label in VARLEN_CONFIGS:
        if any(sl % world_size != 0 for sl in seq_lens):
            if rank == 0:
                print(f"  [SKIP] {label}")
            continue
        try:
            _run_one_config(
                seq_lens, Hq, Hkv, D, causal, dtype, rank, world_size, device
            )
            if rank == 0:
                print(f"  [PASS] {label}")
        except AssertionError as e:
            all_pass = False
            if rank == 0:
                print(f"  [FAIL] {label}: {e}")

    dist.destroy_process_group()

    if rank == 0:
        print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")
    sys.exit(0 if all_pass else 1)


# =====================================================================
# pytest + ray mode (for vLLM CI)
# =====================================================================

if __name__ == "__main__":
    _run_standalone()
else:
    import pytest
    import ray

    from tests.utils import (
        init_test_distributed_environment,
        multi_process_parallel,
    )

    @ray.remote(num_gpus=1, max_calls=1)
    def _ring_attn_worker(
        monkeypatch: pytest.MonkeyPatch,
        tp_size: int,
        pp_size: int,
        rank: int,
        distributed_init_port: str,
    ) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        seq_lens = [int(x) for x in os.environ["TEST_SEQ_LENS"].split(",")]
        Hq = int(os.environ["TEST_HQ"])
        Hkv = int(os.environ["TEST_HKV"])
        D = int(os.environ["TEST_D"])
        causal = os.environ["TEST_CAUSAL"] == "1"
        dtype = getattr(torch, os.environ["TEST_DTYPE"])

        _run_one_config(seq_lens, Hq, Hkv, D, causal, dtype, rank, tp_size, device)

    CP_SIZES = [2, 4]

    @pytest.mark.distributed
    @pytest.mark.parametrize("cp_size", CP_SIZES)
    @pytest.mark.parametrize("seq_lens,Hq,Hkv,D,causal,dtype,label", VARLEN_CONFIGS)
    def test_ring_attn_varlen(
        monkeypatch: pytest.MonkeyPatch,
        cp_size: int,
        seq_lens: list[int],
        Hq: int,
        Hkv: int,
        D: int,
        causal: bool,
        dtype: torch.dtype,
        label: str,
    ):
        if any(sl % cp_size != 0 for sl in seq_lens):
            pytest.skip(f"seq_lens not divisible by cp_size={cp_size}")

        monkeypatch.setenv("TEST_SEQ_LENS", ",".join(str(s) for s in seq_lens))
        monkeypatch.setenv("TEST_HQ", str(Hq))
        monkeypatch.setenv("TEST_HKV", str(Hkv))
        monkeypatch.setenv("TEST_D", str(D))
        monkeypatch.setenv("TEST_CAUSAL", "1" if causal else "0")
        monkeypatch.setenv("TEST_DTYPE", str(dtype).split(".")[-1])

        multi_process_parallel(
            monkeypatch,
            tp_size=cp_size,
            pp_size=1,
            test_target=_ring_attn_worker,
        )
