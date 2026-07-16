# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
import torch

from vllm.lora.lora_overlap_loader import LoRAOverlapLoader

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# 32 layers × rank 64 × hidden 4096 × 4 slots — mirrors the sglang PR #15512 setup
N_LAYERS = 32
RANK = 64
HIDDEN = 4096
N_SLOTS = 4


@pytest.fixture(scope="module")
def weight_buffers():
    torch.manual_seed(42)
    cpu_a = [
        [torch.randn(RANK, HIDDEN, dtype=torch.float16).pin_memory() for _ in range(N_SLOTS)]
        for _ in range(N_LAYERS)
    ]
    cpu_b = [
        [torch.randn(HIDDEN, RANK, dtype=torch.float16).pin_memory() for _ in range(N_SLOTS)]
        for _ in range(N_LAYERS)
    ]
    return cpu_a, cpu_b


def _make_gpu_buffers(device):
    gpu_a = [
        torch.zeros(N_SLOTS, 1, RANK, HIDDEN, device=device, dtype=torch.float16)
        for _ in range(N_LAYERS)
    ]
    gpu_b = [
        torch.zeros(N_SLOTS, 1, HIDDEN, RANK, device=device, dtype=torch.float16)
        for _ in range(N_LAYERS)
    ]
    return gpu_a, gpu_b


def test_equivalence(weight_buffers):
    """Overlap ON/OFF must produce byte-identical GPU weights."""
    cpu_a, cpu_b = weight_buffers
    device = torch.device("cuda")
    loader = LoRAOverlapLoader(device)

    gpu_a_seq, gpu_b_seq = _make_gpu_buffers(device)
    for i in range(N_LAYERS):
        for s in range(N_SLOTS):
            gpu_a_seq[i][s, 0].copy_(cpu_a[i][s])
            gpu_b_seq[i][s, 0].copy_(cpu_b[i][s])
    torch.cuda.synchronize()

    gpu_a_ovl, gpu_b_ovl = _make_gpu_buffers(device)
    with loader.load_context():
        for i in range(N_LAYERS):
            for s in range(N_SLOTS):
                gpu_a_ovl[i][s, 0].copy_(cpu_a[i][s], non_blocking=True)
                gpu_b_ovl[i][s, 0].copy_(cpu_b[i][s], non_blocking=True)
    loader.synchronize()
    torch.cuda.synchronize()

    for i in range(N_LAYERS):
        assert torch.equal(gpu_a_seq[i], gpu_a_ovl[i]), f"lora_a mismatch at layer {i}"
        assert torch.equal(gpu_b_seq[i], gpu_b_ovl[i]), f"lora_b mismatch at layer {i}"


def test_timing_overlap_not_slower(weight_buffers):
    """Overlap mode must not be slower than sequential (5-trial mean, 2% margin)."""
    cpu_a, cpu_b = weight_buffers
    device = torch.device("cuda")
    loader = LoRAOverlapLoader(device)

    gpu_a, gpu_b = _make_gpu_buffers(device)
    mat_a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    mat_b = torch.randn(4096, 4096, dtype=torch.float16, device=device)

    def model_compute(n_iters=50):
        x = mat_a
        for _ in range(n_iters):
            x = torch.mm(x, mat_b)
        return x

    def do_copies():
        for i in range(N_LAYERS):
            for s in range(N_SLOTS):
                gpu_a[i][s, 0].copy_(cpu_a[i][s], non_blocking=True)
                gpu_b[i][s, 0].copy_(cpu_b[i][s], non_blocking=True)

    # warmup
    model_compute(n_iters=5)
    do_copies()
    torch.cuda.synchronize()
    with loader.load_context():
        do_copies()
    loader.synchronize()
    torch.cuda.synchronize()

    n_trials = 5

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        model_compute()
        do_copies()
        torch.cuda.synchronize()
    seq_ms = (time.perf_counter() - t0) / n_trials * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        model_compute()
        with loader.load_context():
            do_copies()
        loader.synchronize()
        torch.cuda.synchronize()
    ovl_ms = (time.perf_counter() - t0) / n_trials * 1000

    assert ovl_ms <= seq_ms * 1.02, (
        f"Overlap ({ovl_ms:.1f} ms) slower than sequential ({seq_ms:.1f} ms)"
    )