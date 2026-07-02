# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.lora.lora_overlap_loader import LoRAOverlapLoader

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_synchronize_noop_when_no_pending():
    loader = LoRAOverlapLoader(torch.device("cuda"))
    assert not loader.has_pending
    loader.synchronize()
    assert not loader.has_pending


def test_correctness_h2d_copy():
    loader = LoRAOverlapLoader(torch.device("cuda"))

    cpu = torch.randn(4096, 4096, dtype=torch.float32).pin_memory()
    gpu = torch.empty(4096, 4096, dtype=torch.float32, device="cuda")

    with loader.load_context():
        gpu.copy_(cpu, non_blocking=True)

    assert loader.has_pending
    loader.synchronize()
    torch.cuda.synchronize()

    assert (gpu.cpu() - cpu).abs().max().item() == 0.0
    assert not loader.has_pending


def test_overlap_produces_matching_results():
    loader = LoRAOverlapLoader(torch.device("cuda"))

    nelem = 64 * 1024 * 1024
    cpu = torch.randn(nelem, dtype=torch.float32).pin_memory()
    gpu = torch.empty(nelem, dtype=torch.float32, device="cuda")
    work = torch.randn(2048, 2048, dtype=torch.float32, device="cuda")

    # warmup
    for _ in range(10):
        work = torch.mm(work, work)
        work = work / (work.abs().max() + 1e-6)
    gpu.copy_(cpu, non_blocking=False)
    with loader.load_context():
        gpu.copy_(cpu, non_blocking=True)
    loader.synchronize()
    torch.cuda.synchronize()

    # overlap: H2D on side stream while compute runs on default stream
    out_seq = work.clone()
    for _ in range(400):
        out_seq = torch.mm(out_seq, out_seq)
        out_seq = out_seq / (out_seq.abs().max() + 1e-6)
    gpu.copy_(cpu, non_blocking=False)
    torch.cuda.synchronize()

    out_ovl = work.clone()
    for _ in range(400):
        out_ovl = torch.mm(out_ovl, out_ovl)
        out_ovl = out_ovl / (out_ovl.abs().max() + 1e-6)
    with loader.load_context():
        gpu.copy_(cpu, non_blocking=True)
    loader.synchronize()
    torch.cuda.synchronize()

    assert torch.allclose(out_seq, out_ovl, atol=1e-4)