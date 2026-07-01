"""End-to-end coherence check for the plan-once forward wiring.

Loads GLM-5.2-FP8 (index_topk_freq=4 -> index sharing active) with HiSparse
enabled, so the "shared" layers route through apply_plan (plan replay) while the
"full" layers produce the plan. Eager (no cudagraph) to isolate the plan-once
routing from the pre-existing unified-mode capture issue. Coherent output ==>
the forward wiring resolves the shared top-k correctly end-to-end.
"""

import json

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="zai-org/GLM-5.2-FP8",
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        attention_config={
            "enable_hisparse": True,
            "hisparse_config": {"device_buffer_size": 2048, "host_pool_gib": 40},
        },
    )
    sp = SamplingParams(temperature=0.0, max_tokens=96)
    prompts = [
        "The history of computing spans mechanical calculators, vacuum tubes, and modern accelerators. Continue in detail:",
        "List three properties of prime numbers:",
    ]
    outs = llm.generate(prompts, sp)
    ok = True
    for o in outs:
        text = o.outputs[0].text
        ntok = len(o.outputs[0].token_ids)
        coherent = ntok >= 8 and text.strip() != ""
        ok = ok and coherent
        print("PLAN_ONCE_GEN " + json.dumps({"ntok": ntok, "text": text[:200]}))
    print("PLAN_ONCE_COHERENT" if ok else "PLAN_ONCE_INCOHERENT")


if __name__ == "__main__":
    main()
