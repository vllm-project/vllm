# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: mamba "align" pre-copy cost vs the copy-free per-step extra.

For the Qwen3.5-35B-A3B GDN config (TP1 state sizes) it compares, per decode
step, the two approaches the copy-free change trades between:

* OLD pre-copy (``batch_memcpy``): migrates a request's full mamba state (conv +
  ssm of every GDN layer) from the previous block to the new running block. This
  fires ONCE per block-boundary crossing, i.e. roughly every
  ``block_size / accept_len`` steps. ``block_size`` is the prefix-cache value
  from the serve log line "Setting attention block size to N tokens".
* NEW copy-free extra (``_gather_resolve_align_src_kernel`` + the per-step H2D
  copy of the src columns done in ``preprocess_mamba``): resolves the src block
  ids in ``gdn_attn.build``. This fires EVERY step.

So the fair comparison is amortized-old (``copy / steps_per_copy``) vs new
(every step). Run with ``pytest -s`` or ``python <thisfile>``; it only prints a
table (no hard perf assert -- the numbers are the point).
"""

from __future__ import annotations

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.backends.gdn_attn import _gather_resolve_align_src_kernel
from vllm.v1.worker.mamba_utils import batch_memcpy

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="GDN align micro-bench needs CUDA/Triton",
    )
except ModuleNotFoundError:  # allow running directly as `python <thisfile>`
    pytest = None

# --- Qwen3.5-35B-A3B-FP8 GDN config (from the model config.json) ------------
# TP shards the value/conv heads, so per-rank state (and thus the pre-copy
# bytes) shrink by TP while the resolve kernel is per-request and TP-invariant.
# Default TP2 matches the deployment config; set BENCH_TP=1 for single-GPU.
TP = int(os.environ.get("BENCH_TP", "2"))
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4  # linear_conv_kernel_dim
NUM_SPEC = 2  # num_speculative_tokens (MTP) -> widens the conv state
NUM_GDN_LAYERS = 30  # 30 linear_attention layers (+10 full_attention)
SSM_DTYPE = torch.float32  # mamba_ssm_dtype
CONV_DTYPE = torch.bfloat16

# Prefix-cache serve log: "Setting attention block size to N tokens".
BLOCK_SIZE_TOKENS = int(os.environ.get("BENCH_BLOCK_SIZE", "1088"))
ACCEPT_LEN = 2.05  # measured acceptance length (num_spec=2)

CONV_DIM = HEAD_K_DIM * NUM_K_HEADS * 2 + HEAD_V_DIM * NUM_V_HEADS  # 8192
SSM_SHAPE = (NUM_V_HEADS // TP, HEAD_V_DIM, HEAD_K_DIM)
CONV_SHAPE = (CONV_DIM // TP, CONV_KERNEL - 1 + NUM_SPEC)

NUM_BLOCKS = 2048  # state-pool blocks (pool >> L2, so copies miss cache)
BLOCK = 256  # resolve-kernel tile
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds / call


def _make_copy_inputs(B, device):
    """batch_memcpy ptr/size arrays for B reqs each migrating conv+ssm of every
    GDN layer (2 * NUM_GDN_LAYERS * B entries) through shared conv/ssm pools."""
    ssm_pool = torch.empty((NUM_BLOCKS, *SSM_SHAPE), dtype=SSM_DTYPE, device=device)
    conv_pool = torch.empty((NUM_BLOCKS, *CONV_SHAPE), dtype=CONV_DTYPE, device=device)
    ssm_bytes = ssm_pool[0].numel() * ssm_pool.element_size()
    conv_bytes = conv_pool[0].numel() * conv_pool.element_size()

    src_ptrs, dst_ptrs, sizes = [], [], []
    n = 0
    for _layer in range(NUM_GDN_LAYERS):
        for _r in range(B):
            s = n % NUM_BLOCKS
            d = (n + NUM_BLOCKS // 2) % NUM_BLOCKS  # distinct src/dst block
            n += 1
            src_ptrs += [ssm_pool[s].data_ptr(), conv_pool[s].data_ptr()]
            dst_ptrs += [ssm_pool[d].data_ptr(), conv_pool[d].data_ptr()]
            sizes += [ssm_bytes, conv_bytes]

    src = torch.tensor(src_ptrs, dtype=torch.int64, device=device)
    dst = torch.tensor(dst_ptrs, dtype=torch.int64, device=device)
    sz = torch.tensor(sizes, dtype=torch.int64, device=device)
    total_mib = sum(sizes) / 2**20
    return (ssm_pool, conv_pool), (src, dst, sz), total_mib


def _make_resolve_inputs(B, device):
    g = torch.Generator().manual_seed(0)
    idx = torch.arange(B, dtype=torch.int32, device=device)  # identity (V1)
    ssm_col = torch.randint(0, 64, (B,), dtype=torch.int32, generator=g).to(device)
    conv_col = torch.randint(0, 64, (B,), dtype=torch.int32, generator=g).to(device)
    conv_off = torch.randint(0, 4, (B,), dtype=torch.int32, generator=g).to(device)
    bt = torch.randint(1, 100000, (B, 128), dtype=torch.int32, generator=g).to(device)
    out = tuple(torch.empty(B, dtype=torch.int32, device=device) for _ in range(3))
    # host columns for the per-step H2D copy that preprocess_mamba does
    host = tuple(torch.empty(B, dtype=torch.int32, pin_memory=True) for _ in range(3))
    dev = tuple(torch.empty(B, dtype=torch.int32, device=device) for _ in range(3))
    return idx, ssm_col, conv_col, conv_off, bt, out, host, dev


def _resolve_call(inp):
    idx, ssm_col, conv_col, conv_off, bt, out, _host, _dev = inp
    B = idx.shape[0]
    _gather_resolve_align_src_kernel[(triton.cdiv(B, BLOCK),)](
        idx,
        ssm_col,
        conv_col,
        conv_off,
        bt,
        bt.stride(0),
        bt.stride(1),
        bt.size(1) - 1,
        out[0],
        out[1],
        out[2],
        B,
        B,
        BLOCK=BLOCK,
    )


def _h2d_call(inp):
    *_, host, dev = inp
    for h, d in zip(host, dev):
        d.copy_(h, non_blocking=True)


def _run_bench():
    device = torch.device("cuda")
    steps_per_copy = BLOCK_SIZE_TOKENS / ACCEPT_LEN
    print(
        f"\nQwen3.5-35B GDN TP{TP}: ssm/block={SSM_SHAPE} {SSM_DTYPE}, "
        f"conv/block={CONV_SHAPE} {CONV_DTYPE}, {NUM_GDN_LAYERS} GDN layers"
    )
    print(
        f"block_size={BLOCK_SIZE_TOKENS} tok, accept_len={ACCEPT_LEN} "
        f"-> ~{steps_per_copy:.0f} steps/copy\n"
    )
    print(
        f"{'B':>4} {'MiB/copy':>9} {'copy_full':>10} {'copy/step':>10} "
        f"{'resolve':>8} {'h2d':>7} {'extra/step':>11} {'copyfree Δ/step':>16}"
    )
    print(
        f"{'':>4} {'':>9} {'(us)':>10} {'(us,amort)':>10} {'(us)':>8} "
        f"{'(us)':>7} {'(us)':>11} {'(us)':>16}"
    )
    for B in BATCHES:
        pools, (src, dst, sz), mib = _make_copy_inputs(B, device)
        copy_us = _bench(lambda s=src, d=dst, z=sz: batch_memcpy(s, d, z), iters=50)
        r_inputs = _make_resolve_inputs(B, device)
        resolve_us = _bench(lambda r=r_inputs: _resolve_call(r), iters=300)
        h2d_us = _bench(lambda r=r_inputs: _h2d_call(r), iters=300)
        del pools, src, dst, sz, r_inputs
        torch.accelerator.empty_cache()

        amort = copy_us / steps_per_copy
        extra = resolve_us + h2d_us
        delta = extra - amort  # >0: copy-free costs more per step; <0: it saves
        print(
            f"{B:>4} {mib:>9.0f} {copy_us:>10.1f} {amort:>10.2f} "
            f"{resolve_us:>8.2f} {h2d_us:>7.2f} {extra:>11.2f} {delta:>+16.2f}"
        )


def test_align_copy_vs_resolve_bench(capsys):
    with capsys.disabled():
        _run_bench()


if __name__ == "__main__":
    _run_bench()
