# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hal XPU e2e smoke for hierarchical (Colibri-style) expert staging.

Default: Mixtral-8x7B Instruct AWQ (~25 GiB) — fits in Arc VRAM + host RAM
without swap. Override with HIER_MODEL.
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("ZE_AFFINITY_MASK", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from vllm.model_executor.offloader.hierarchical.manager import get_tier_manager

MODEL = os.environ.get(
    "HIER_MODEL",
    "/tank/nas/models/Mixtral-8x7B-Instruct-v0.1-AWQ",
)
# Mixtral E=8. Default slots=4 stages experts; keep prompts short so a
# single forward does not need more unique experts than slots.
TIER_NUM_SLOTS = int(os.environ.get("HIER_TIER_NUM_SLOTS", "4"))
TIER_RAM_GB = float(os.environ.get("HIER_TIER_RAM_GB", "8"))
GPU_MEM_UTIL = float(os.environ.get("HIER_GPU_MEM_UTIL", "0.85"))


def main() -> None:
    print(
        f"E2E_START loading {MODEL} with hierarchical staging "
        f"(slots={TIER_NUM_SLOTS}, ram_gb={TIER_RAM_GB}, "
        f"gpu_mem_util={GPU_MEM_UTIL})...",
        flush=True,
    )
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        offload_backend="hierarchical",
        tier_num_slots=TIER_NUM_SLOTS,
        tier_ram_gb=TIER_RAM_GB,
        tier_policy="quality",
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
        quantization="awq",
    )
    print(f"E2E_LOADED in {time.time() - t0:.1f}s", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    t1 = time.time()
    outs = llm.generate(["Hi"], sp)
    elapsed = time.time() - t1
    text = outs[0].outputs[0].text
    n = len(outs[0].outputs[0].token_ids)
    mgr = get_tier_manager()
    stats = mgr.stats.snapshot() if mgr is not None else {}
    print(
        "E2E_RESULT",
        json.dumps(
            {
                "model": MODEL,
                "slots": TIER_NUM_SLOTS,
                "gpu_mem_util": GPU_MEM_UTIL,
                "text": text,
                "tokens": n,
                "gen_s": elapsed,
                "tok_s": n / max(elapsed, 1e-6),
                "tier_stats": stats,
            }
        ),
        flush=True,
    )
    print("E2E_OK", flush=True)


if __name__ == "__main__":
    main()
