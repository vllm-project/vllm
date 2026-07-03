# -*- coding: utf-8 -*-
"""
Mamba CoW (Copy-on-Write) PoC — RTX 5070
Python DAG(CPU) + PyTorch .copy_()(GPU搬运工) = 并发安全
"""

import torch

def mamba_step(state_block, input_token):
    """模拟 Mamba in-place 覆盖写: h_t = 0.9*h_{t-1} + 0.1*x_t"""
    state_block.mul_(0.9).add_(input_token * 0.1)

def run():
    assert torch.cuda.is_available(), "No CUDA"
    dev = "cuda"; dim = 16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    # State pool (模拟 vLLM BlockManager)
    pool = torch.zeros((4, dim), device=dev)
    pool[0] = torch.arange(dim, device=dev, dtype=torch.float32)

    # === Scenario A: No CoW (灾难) ===
    print("\n--- A: No CoW ---")
    req_a, req_b = 0, 0
    inp_a = torch.ones(dim, device=dev) * 10.0
    inp_b = torch.ones(dim, device=dev) * 50.0
    mamba_step(pool[req_a], inp_a)
    mamba_step(pool[req_b], inp_b)  # B覆盖了A的结果!
    print(f"  A state: {pool[req_a][:3].tolist()}")
    print(f"  B state: {pool[req_b][:3].tolist()}")
    both_same = torch.allclose(pool[req_a], pool[req_b])
    print(f"  A==B? {both_same} [FAIL: B polluted A]")

    # === Scenario B: CoW with DAG + GPU搬运工 ===
    print("\n--- B: CoW (DAG + .copy_) ---")
    pool[0] = torch.arange(dim, device=dev, dtype=torch.float32)  # reset
    dag = {0: 2}  # Block 0 shared by A and B, ref_count=2
    req_a, req_b = 0, 0

    # DAG detects fork: ref_count > 1, trigger CoW
    if dag[req_b] > 1:
        print(f"  [DAG] Block 0 ref_count={dag[req_b]} > 1, CoW triggered")
        new_block = 1
        pool[new_block].copy_(pool[req_b])  # GPU mover
        print(f"  [GPU] Block 0 copied to Block 1 ({pool[0].data_ptr()} -> {pool[1].data_ptr()})")
        req_b = new_block
        dag[0] -= 1
        dag[new_block] = 1

    mamba_step(pool[req_a], inp_a)
    mamba_step(pool[req_b], inp_b)
    print(f"  A state: {pool[0][:3].tolist()}")
    print(f"  B state: {pool[1][:3].tolist()}")
    different = not torch.allclose(pool[0], pool[1])
    assert different, "CoW FAILED"
    print(f"  A!=B? {different} [PASS: CoW works on {torch.cuda.get_device_name(0)}]")
    print(f"\n  DAG(ref_count) + PyTorch .copy_() = Mamba concurrency without CUDA kernel changes.")

if __name__ == "__main__":
    run()
