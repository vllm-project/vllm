# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that GPU memory is fully released after RixlConnector shutdown on ROCm.

Regression test for ROCm/ucx#33: UCX rocm_ipc transport permanently pinned
GPU memory via hsa_amd_ipc_memory_create during ucp_mem_map, causing
GPU memory to be unrecoverable after engine shutdown.
"""

import gc

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm platform required",
)


def _mb(b: int) -> float:
    return b / (1024 * 1024)


def _gpu_snapshot(tag: str, prev_alloc: float = 0.0) -> dict:
    """Print and return current GPU memory stats."""
    torch.accelerator.synchronize()
    alloc = torch.accelerator.memory_allocated()
    reserved = torch.accelerator.memory_reserved()
    # mem_get_info is not available on torch.accelerator
    try:
        drv_free, drv_total = torch.cuda.mem_get_info()
        drv_used = drv_total - drv_free
        drv_pct = drv_used / drv_total * 100
    except Exception:
        drv_used = drv_total = drv_pct = 0
    alloc_mb = _mb(alloc)
    drv_used_mb = _mb(drv_used)
    delta = alloc_mb - prev_alloc
    print(
        f"  {tag:<40s} | {alloc_mb:>9.1f} alloc | "
        f"{_mb(reserved):>9.1f} rsrvd | "
        f"{drv_used_mb:>9.1f} driver ({drv_pct:.1f}%) | "
        f"delta {delta:>+9.1f}"
    )
    return {
        "tag": tag,
        "alloc_mb": alloc_mb,
        "drv_used_mb": drv_used_mb,
        "drv_pct": drv_pct,
    }


def _full_gpu_cleanup():
    """gc.collect + torch empty_cache, multiple rounds."""
    gc.unfreeze()
    for _ in range(3):
        if gc.collect() == 0:
            break
    torch.accelerator.empty_cache()


@pytest.mark.parametrize("model_name, sw_size", [("google/gemma-3-1b-it", 512)])
def test_gpu_memory_rixl_hma(model_name, sw_size):
    """Track GPU memory through NixlConnector create/infer/shutdown cycle."""
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.5,
        "kv_transfer_config": KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
        ),
        "max_model_len": 2048,
        "disable_hybrid_kv_cache_manager": False,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": False,
        "block_size": 16,
    }

    print("\n" + "=" * 90)
    print("GPU MEMORY -- RIXL NixlConnector HMA (ROCm)")
    print("=" * 90)
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    snap0 = _gpu_snapshot("0. baseline", 0.0)

    # create + infer
    llm = LLM(**llm_kwargs)
    snap1 = _gpu_snapshot("1. after LLM()", snap0["alloc_mb"])

    llm.generate(
        ["hi" * 1401],
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                }
            },
        ),
    )
    snap2 = _gpu_snapshot("2. after generate()", snap1["alloc_mb"])

    # shutdown + cleanup
    print("\n--- shutdown ---")
    llm.llm_engine.engine_core.shutdown()
    _gpu_snapshot("3. after shutdown()", snap2["alloc_mb"])

    del llm
    _full_gpu_cleanup()
    cleanup_dist_env_and_memory()
    _full_gpu_cleanup()
    torch._dynamo.reset()
    gc.collect()
    torch.accelerator.empty_cache()
    snap_final = _gpu_snapshot("4. final", snap2["alloc_mb"])

    # summary
    print("\n" + "=" * 90)
    baseline = snap0["alloc_mb"]
    final = snap_final["alloc_mb"]
    peak = snap2["alloc_mb"]
    total_alloc = peak - baseline

    print(
        f"  PyTorch:  baseline={baseline:.0f}  peak={peak:.0f}  "
        f"final={final:.0f}  "
        f"leaked={final - baseline:.0f} MB"
        + (
            f" ({(final - baseline) / total_alloc * 100:.1f}%)"
            if total_alloc > 0
            else ""
        )
    )

    drv_base = snap0["drv_used_mb"]
    drv_final = snap_final["drv_used_mb"]
    drv_leaked = drv_final - drv_base
    print(
        f"  Driver:   baseline={drv_base:.0f} ({snap0['drv_pct']:.1f}%)  "
        f"peak={snap2['drv_used_mb']:.0f} ({snap2['drv_pct']:.1f}%)  "
        f"final={drv_final:.0f} ({snap_final['drv_pct']:.1f}%)  "
        f"leaked={drv_leaked:.0f} MB"
    )
    print("=" * 90)

    # Peak driver memory used above baseline
    drv_peak = snap2["drv_used_mb"] - drv_base
    leak_pct = (drv_leaked / drv_peak * 100) if drv_peak > 0 else 0
    max_leak_pct = 10
    assert leak_pct <= max_leak_pct, (
        f"{drv_leaked:.0f} MB ({leak_pct:.1f}%) of driver-level GPU memory "
        f"not freed after NixlConnector shutdown "
        f"(peak allocation: {drv_peak:.0f} MB, threshold: {max_leak_pct}%)"
    )


@pytest.mark.parametrize("model_name", ["google/gemma-3-1b-it"])
def test_gpu_memory_no_rixl_baseline(model_name):
    """Same workload without NixlConnector.  Comparing driver-level memory
    between this and test_gpu_memory_rixl_hma isolates UCX/RIXL impact."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    print("\n" + "=" * 90)
    print("CONTROL -- same model, no RIXL connector")
    print("=" * 90)
    gc.collect()
    torch.accelerator.empty_cache()
    snap0 = _gpu_snapshot("baseline", 0.0)

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        max_num_batched_tokens=1024,
        enable_prefix_caching=False,
        block_size=16,
    )
    _gpu_snapshot("after LLM()", snap0["alloc_mb"])

    llm.generate(["hi " * 500], SamplingParams(max_tokens=1))
    snap_peak = _gpu_snapshot("after generate()", snap0["alloc_mb"])

    llm.llm_engine.engine_core.shutdown()
    del llm
    _full_gpu_cleanup()
    cleanup_dist_env_and_memory()
    _full_gpu_cleanup()
    torch._dynamo.reset()
    gc.collect()
    torch.accelerator.empty_cache()
    snap_final = _gpu_snapshot("final", snap0["alloc_mb"])

    drv_base = snap0["drv_used_mb"]
    drv_leaked = snap_final["drv_used_mb"] - drv_base
    drv_peak = snap_peak["drv_used_mb"] - drv_base
    print(f"\n  Driver leaked (no rixl): {drv_leaked:.0f} MB")
    print("=" * 90)

    leak_pct = (drv_leaked / drv_peak * 100) if drv_peak > 0 else 0
    max_leak_pct = 10
    assert leak_pct <= max_leak_pct, (
        f"{drv_leaked:.0f} MB ({leak_pct:.1f}%) of driver-level GPU memory "
        f"not freed after baseline shutdown "
        f"(peak allocation: {drv_peak:.0f} MB, threshold: {max_leak_pct}%)"
    )
