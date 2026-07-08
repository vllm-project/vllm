# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the VLLM_STOCK_CAPTURE_KV_PREP prototype: folding the
slot-mapping (KV-cache management) kernel into the stock FULL decode cudagraph.

The prototype must be byte-identical to the eager baseline while the KV cache is
exercised in many ways -- padded decode (R < captured graph size), long sequences
spanning many KV blocks, shared-prefix block reuse, a batch-size sweep, block churn
across many requests, and chunked prefill -- in BOTH FULL cudagraph modes. To
isolate the flag's effect from run-to-run kernel-autotune nondeterminism, flag-off
and flag-on run in the SAME process (shared JIT/autotune) with flashinfer autotune
disabled, and we compare greedy token ids. We also assert (via the compilation
counter) that the capture hook actually fired -- a silent gate fallback would make
off==on trivially and hide a broken prototype.
"""

import gc

import pytest
import torch

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationMode
from vllm.platforms import current_platform

MODEL = "facebook/opt-125m"

# Prompts/conditions run inside a single LLM instance (the flag is fixed per
# instance). Each exercises a different KV-cache regime. capture_sizes below make
# batch 3 and 7 pad to 4 and 8, so the R < captured-graph-size path is hit.
_SHARED_PREFIX = (
    "In the beginning the kingdom was small and its people were few, "
    "but over many long years it grew and prospered across the wide land. "
)
CONDITIONS = {
    # 1 req, long -> spans many 16-token KV blocks and block boundaries.
    "batch1_multiblock": (["The history of the ancient world begins when"], 220),
    # batch 3 -> pads to captured size 4 (R < N padded decode).
    "pad_batch3": (["Hello there,", "The weather today is", "My favorite food is"], 80),
    # shared long prefix -> prefix-cache block reuse across requests.
    "prefix_reuse": (
        [_SHARED_PREFIX + s for s in ("Then", "After", "Later", "Soon")],
        48,
    ),
    # batch 7 -> pads to captured size 8.
    "batch7": ([f"Item number {i} is about" for i in range(7)], 40),
    # many short reqs -> block allocation churn / reuse.
    "many_req": ([f"Question {i}:" for i in range(24)], 24),
}


def _run(capture_kv_prep: bool, cudagraph_mode: str, partition: bool, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_STOCK_CAPTURE_KV_PREP", "1" if capture_kv_prep else "0")
    # vLLM caches env vars after the first LLM init in a process; reset so this
    # run's flag value is re-read rather than the previous run's cached one.
    envs.disable_envs_cache()
    before = compilation_counter.stock_kv_prep_captures
    llm = LLM(
        model=MODEL,
        enforce_eager=False,
        max_model_len=1024,
        # Fixed KV cache -> skips dynamic memory profiling (whose assert is flaky
        # on a shared box where free GPU memory fluctuates during profiling). 2 GiB
        # is ample for opt-125m across all conditions below.
        kv_cache_memory_bytes=2 * 1024**3,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,  # small -> long prompts chunk-prefill
        block_size=16,
        kernel_config={"enable_flashinfer_autotune": False},  # determinism
        compilation_config={
            "mode": CompilationMode.STOCK_TORCH_COMPILE,
            "cudagraph_mode": cudagraph_mode,
            "use_inductor_graph_partition": partition,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        },
    )
    outs = {}
    for name, (prompts, max_tokens) in CONDITIONS.items():
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        res = llm.generate(prompts, sp, use_tqdm=False)
        outs[name] = [list(o.outputs[0].token_ids) for o in res]
    captures = compilation_counter.stock_kv_prep_captures - before
    del llm
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()
    return outs, captures


@pytest.mark.forked
@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.support_static_graph_mode(),
    reason="requires CUDA + full cudagraph support",
)
@pytest.mark.parametrize(
    "cudagraph_mode,partition",
    [
        # Whole-graph FULL decode capture (no graph partition).
        ("FULL_DECODE_ONLY", False),
        # The v1 default: FULL decode + piecewise prefill via Inductor partition.
        ("FULL_AND_PIECEWISE", True),
    ],
)
def test_stock_capture_kv_prep_byte_identical(cudagraph_mode, partition, monkeypatch):
    """flag-off == flag-on token ids across varied KV conditions, both FULL modes,
    with the capture hook proven to have fired."""
    off, off_caps = _run(False, cudagraph_mode, partition, monkeypatch)
    on, on_caps = _run(True, cudagraph_mode, partition, monkeypatch)

    # The prototype must actually engage under the flag (guard against a silent
    # gate fallback that would make off==on vacuously).
    assert off_caps == 0, f"flag off but hook fired {off_caps}x"
    assert on_caps > 0, (
        f"flag on but capture hook never fired ({cudagraph_mode}, partition={partition}"
        "); the prototype fell back to eager -- the test would not prove anything."
    )

    for name in CONDITIONS:
        assert on[name] == off[name], (
            f"KV-prep-capture changed output for condition '{name}' "
            f"({cudagraph_mode}): off vs on token ids differ"
        )
