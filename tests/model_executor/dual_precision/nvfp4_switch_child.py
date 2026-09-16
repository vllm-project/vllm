# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Child process for ``test_nvfp4_switch_gpu.py``: generate at one base precision.

``argv[1]`` is ``bf16`` (no scheduler policy, the shadow stays attached but
unbound) or ``w4`` (``VLLM_DUAL_PRECISION_POLICY=uniform_w4``, so every attached
layer decodes off the NVFP4 shadow). ``argv[2]`` is where the JSON report goes.

One precision per process because the policy is read once when the scheduler is
built, and because two 9B engines do not fit on one card at this utilization.
"""

from __future__ import annotations

import json
import os
import sys

PROMPTS = [
    "The capital of France is",
    "Question: What is 17 plus 25? Answer:",
]


def collect(model) -> dict:
    """Runs inside the worker: which base each binding actually points at."""
    from vllm.model_executor.dual_precision import (
        get_active_precision,
        get_dual_precision_state,
    )

    state = get_dual_precision_state(model)
    shadow = [b for b in state.bindings if b.shadow_active]
    return {
        "active_precision": get_active_precision(model),
        "bindings_on_shadow": sum(1 for b in shadow if b.active is b.int4_or_fallback),
        "shadow_bindings_total": len(shadow),
    }


def main() -> int:
    mode, report_path = sys.argv[1], sys.argv[2]
    bf16_model, nvfp4_model = os.environ["BF16_MODEL"], os.environ["NVFP4_MODEL"]

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_DUAL_PRECISION_ROLLOUT"] = "1"
    os.environ["VLLM_DUAL_PRECISION_INT4_MODEL"] = nvfp4_model
    os.environ["VLLM_DUAL_PRECISION_BF16_LAYERS"] = "none"
    os.environ["VLLM_DUAL_PRECISION_INT4_MODULES"] = "all"
    if mode == "w4":
        os.environ["VLLM_DUAL_PRECISION_POLICY"] = "uniform_w4"
    else:
        os.environ.pop("VLLM_DUAL_PRECISION_POLICY", None)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=bf16_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=0,
    )
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=24))
    (report,) = llm.apply_model(collect)
    report["texts"] = [o.outputs[0].text for o in outputs]
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
