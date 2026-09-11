"""End-to-end vLLM throughput bench: Python BlockPool vs Rust.

Monkey-patches `vllm.v1.core.block_pool.BlockPool` with `RustBlockPool` when
VLLM_USE_RUST_BLOCK_POOL=1, *before* any other vLLM import. Safe under vLLM
0.18 too (the BlockPool class definition is byte-for-byte identical between
0.18 and our 0.19 fork, confirmed via diff).

Run:
    NANOVLLM_EAGER=1 VLLM_USE_RUST_BLOCK_POOL=1 python bench_e2e.py
    NANOVLLM_EAGER=1 VLLM_USE_RUST_BLOCK_POOL=0 python bench_e2e.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 1. Ensure rust_block_pool.py (from our branch) is importable as
#    vllm.v1.core.rust_block_pool, even though the conda env's vllm doesn't
#    ship it. Copy-on-read via a file-based loader.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _install_rust_block_pool_shim():
    """Import vllm.v1.core.block_pool, then swap its BlockPool symbol for
    RustBlockPool. Must happen *before* anything else imports vllm."""
    import importlib.util
    import vllm  # triggers cuda platform imports — safe
    import vllm.v1.core.block_pool as bp

    # Load our rust_block_pool.py from the branch
    shim_path = os.path.join(_REPO_ROOT, "vllm", "v1", "core", "rust_block_pool.py")
    if not os.path.exists(shim_path):
        raise FileNotFoundError(f"missing {shim_path}")
    spec = importlib.util.spec_from_file_location(
        "vllm.v1.core.rust_block_pool", shim_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vllm.v1.core.rust_block_pool"] = mod
    spec.loader.exec_module(mod)

    bp.BlockPool = mod.RustBlockPool
    print(
        f"[bench_e2e] patched vllm.v1.core.block_pool.BlockPool -> "
        f"{bp.BlockPool.__module__}.{bp.BlockPool.__name__}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--max-in", type=int, default=512)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    use_rust = os.environ.get("VLLM_USE_RUST_BLOCK_POOL") == "1"
    if use_rust:
        _install_rust_block_pool_shim()
        # Pass the shim to child engine processes (vLLM spawns EngineCore).
        hook_dir = os.path.join(_REPO_ROOT, "vllm_rs", "_sitehook")
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = hook_dir + (os.pathsep + existing if existing else "")

    # Import AFTER the patch so the scheduler picks up the new BlockPool.
    from vllm import LLM, SamplingParams

    import random
    random.seed(0)
    # Use TokensPrompt objects so we can pass raw token-ids without needing
    # a tokenizer round-trip.
    from vllm.inputs import TokensPrompt
    prompts = [
        TokensPrompt(prompt_token_ids=[random.randint(0, 10000)
                                       for _ in range(random.randint(64, args.max_in))])
        for _ in range(args.num_seqs)
    ]
    sp = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=random.randint(64, args.max_out),
        )
        for _ in range(args.num_seqs)
    ]

    print(
        f"[bench_e2e] model={args.model}  num_seqs={args.num_seqs}  "
        f"max_in={args.max_in}  max_out={args.max_out}  rust={use_rust}",
        flush=True,
    )

    llm = LLM(model=args.model, max_model_len=1024, enforce_eager=True, gpu_memory_utilization=0.6)
    # Warmup
    llm.generate([TokensPrompt(prompt_token_ids=[1, 2, 3])],
                 SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8))

    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    t = time.time() - t0
    total_out = sum(s.max_tokens for s in sp)
    print(
        f"[bench_e2e] RESULT  rust={use_rust}  "
        f"total_out_tok={total_out}  time={t:.3f}s  tput={total_out / t:.1f} tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
