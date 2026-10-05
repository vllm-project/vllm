# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Live-mode runner: real GLM-5.2 in vLLM TP8 where the MonoKernel REPLACES MoE layers
# 3..77 on pure-decode steps (arm C), vs vLLM baselines (arms A/B). Greedy sanity +
# GSM8K.
#
# Run from the repo root in a ROCm vLLM env with this branch, under a watchdog, e.g.
#   timeout 5400 python tools/mono_check/run_live.py --model <GLM-5.2 checkpoint> \
#       --arm C --sanity --gsm8k 250 --out <out_dir>
# Arms (identical LLM settings; only env / install differ):
#   A: vLLM as shipped (VLLM_ROCM_USE_AITER=1)
#   B: A + VLLM_ROCM_USE_AITER_FP4BMM=0 VLLM_ROCM_USE_AITER_FP8BMM=0 (bf16 W_UK/W_UV)
#   C: A + live MonoKernel on MoE layers 3..77 (pure-decode steps only)
# GSM8K format: lm-eval "gsm8k" style 5-shot plain completion ("Question: ...\nAnswer:
# ..." with the first 5 train exemplars, i.e. lm-eval's first_n sampler), greedy, max
# 256 new tokens, stop at "Question:" / "</s>" / "<|im_end|>"; strict-match = "#### N",
# flexible = last number.
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
ap.add_argument("--model", required=True, help="local GLM-5.2 checkpoint directory")
ap.add_argument("--out", required=True)
ap.add_argument(
    "--spec-k",
    type=int,
    default=0,
    help="speculative decoding: vLLM native MTP with K draft tokens "
    "(arm C runs verify steps on the MonoKernel via "
    "LiveConfig.spec_decode AUTO)",
)
ap.add_argument(
    "--sanity",
    action="store_true",
    help="greedy sanity at batch 1 and 4 (arm C: also mono off/on agreement)",
)
ap.add_argument("--sanity-tokens", type=int, default=64)
ap.add_argument(
    "--gsm8k",
    type=int,
    default=0,
    help="number of GSM8K test questions (0 = skip; 1319 = full)",
)
ap.add_argument("--gsm8k-start", type=int, default=0)
ap.add_argument("--max-num-seqs", type=int, default=8)
ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--num-blocks", type=int, default=8192)
ap.add_argument("--gpu-mem", type=float, default=0.70)
ap.add_argument(
    "--graphs",
    action="store_true",
    help="HIP graphs (FULL decode capture) instead of enforce_eager",
)
ap.add_argument("--capture-sizes", default="1,2,4,5,6,8")
ap.add_argument("--indexer-mode", default="attn", choices=["attn", "indexer_only"])
ap.add_argument("--sizes", default="1,2,4,5,6,8")
ap.add_argument(
    "--indexer-selftest",
    type=int,
    default=0,
    help="indexer_only: compare vs full attention for N mono steps",
)
ap.add_argument("--poll-limit", type=int, default=20_000_000)
ap.add_argument("--layers", default="3-77")
ap.add_argument(
    "--ref-sanity",
    default="",
    help="sanity.json of an earlier run to compare token ids against",
)
# LiveConfig switches: unset = the LiveConfig default (live.py); --x / --no-x override
# it
BOOL = argparse.BooleanOptionalAction
ap.add_argument(
    "--fused-indexer",
    action=BOOL,
    default=None,
    help="LiveConfig.fused_indexer (in-kernel paged indexer; FP8 attention)",
)
ap.add_argument(
    "--poll-early-out", action=BOOL, default=None, help="LiveConfig.poll_early_out"
)
ap.add_argument(
    "--attention-weight",
    default=None,
    choices=["bf16", "fp8_block128"],
    help="LiveConfig.attention_weight",
)
ap.add_argument(
    "--weights-selfcheck",
    default="",
    help="comma list of layers: mono_weights_selfcheck RPC (needs FSE=1)",
)
ap.add_argument(
    "--live-json",
    default="{}",
    help="extra LiveConfig fields as JSON (override the flags above)",
)
a = ap.parse_args()

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
# Shared expert on the main stream in every arm (nothing else resident while the
# persistent kernel runs; numerically identical for the baselines).
os.environ.setdefault("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "0")
if a.arm == "B":
    os.environ["VLLM_ROCM_USE_AITER_FP4BMM"] = "0"
    os.environ["VLLM_ROCM_USE_AITER_FP8BMM"] = "0"

from gsm8k import flexible, gsm8k_data, norm_num, same, stderr, strict  # noqa: E402

PROMPTS = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Explain in one paragraph why the sky is blue.",
    "List the first ten prime numbers:",
]


def log(*x):
    print(*x, flush=True)


def main():
    os.makedirs(a.out, exist_ok=True)
    from vllm import LLM, SamplingParams

    kw = dict(
        model=a.model,
        tensor_parallel_size=8,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_mem,
        max_num_seqs=a.max_num_seqs,
        trust_remote_code=True,
        kv_cache_dtype="auto",
        block_size=16,
        num_gpu_blocks_override=a.num_blocks,
        worker_extension_cls="tools.mono_check.harness.live_worker_ext.MonoLiveWorkerExtension",
    )
    if a.spec_k > 0:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": a.spec_k}
        kw["disable_log_stats"] = (
            # the spec-decode counters (acceptance) need stat logging (LLM default: off)
            False
        )
    if a.graphs:
        kw["compilation_config"] = dict(
            mode=0,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[int(s) for s in a.capture_sizes.split(",")],
        )
    else:
        kw["enforce_eager"] = True
    meta = dict(
        arm=a.arm,
        argv=sys.argv,
        env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
        llm=kw,
    )
    t0 = time.time()
    llm = None
    try:
        if a.arm == "C" and a.graphs:
            # Kernel ops must exist before vLLM captures graphs -> install from inside
            # the worker right after model load (see live_worker_ext / env hook).
            os.environ["MONO_LIVE_PREINSTALL"] = json.dumps(_live_cfg())
        llm = LLM(**kw)
        log(f"LLM init {time.time() - t0:.0f}s")
        meta["init_s"] = time.time() - t0
        meta["mem_after_init"] = llm.collective_rpc("mono_mem")
        if a.arm == "C" and not a.graphs:
            t1 = time.time()
            info = llm.collective_rpc("mono_live_install", args=(_live_cfg(),))
            log(f"mono install {time.time() - t1:.0f}s", info[0])
            meta["install"] = info
        if a.arm == "C":
            meta["mem_after_install"] = llm.collective_rpc("mono_mem")
            log(
                "mem per rank:",
                [round(m["device_used_gib"], 1) for m in meta["mem_after_install"]],
            )
        if a.weights_selfcheck:
            t1 = time.time()
            wc = llm.collective_rpc(
                "mono_weights_selfcheck",
                args=([int(x) for x in a.weights_selfcheck.split(",")],),
            )
            meta["weights_selfcheck"] = wc
            log(
                f"weights selfcheck {time.time() - t1:.0f}s:",
                json.dumps(wc, default=str)[:4000],
            )
        if a.sanity:
            meta["sanity"] = sanity(llm, SamplingParams)
        if a.spec_k > 0:
            meta["spec_decode"] = spec_metrics(llm)
            log("spec decode:", meta["spec_decode"])
        if a.gsm8k:
            meta["gsm8k"] = gsm8k(llm, SamplingParams)
    finally:
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=60)
            except TypeError:
                llm.llm_engine.engine_core.shutdown()
            log("SHUTDOWN")


def _parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            x, y = part.split("-")
            out += list(range(int(x), int(y) + 1))
        else:
            out.append(int(part))
    return out


def _live_cfg():
    d = dict(
        ckpt=a.model,
        layers=_parse_layers(a.layers),
        sizes=[int(s) for s in a.sizes.split(",")],
        poll_limit=a.poll_limit,
        indexer_mode=a.indexer_mode,
        step_sync=not a.graphs,
        max_model_len=a.max_model_len,
        extra=dict(indexer_selftest_steps=a.indexer_selftest),
    )
    for k in ("fused_indexer", "poll_early_out", "attention_weight"):
        if getattr(a, k) is not None:
            d[k] = getattr(a, k)
    d.update(json.loads(a.live_json))
    return d


def spec_metrics(llm):
    """VLLM's speculative-decoding counters (draft / accepted tokens, per-position
    acceptance) -> acceptance rate."""
    out = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                v = getattr(m, "value", None)
                if v is None and hasattr(m, "values"):
                    v = list(m.values)
                out[m.name] = v
    except Exception as e:  # noqa: BLE001 -- metrics API differences across vLLM versions
        out["error"] = repr(e)
    d = out.get("vllm:spec_decode_num_draft_tokens")
    acc = out.get("vllm:spec_decode_num_accepted_tokens")
    if isinstance(d, (int, float)) and isinstance(acc, (int, float)) and d:
        out["acceptance_rate"] = acc / d
    return out


def _stats(llm):
    if a.arm != "C":
        return None
    return llm.collective_rpc("mono_live_stats")[0]


def _reset(llm):
    if a.arm == "C":
        llm.collective_rpc("mono_live_reset_stats")


def _gen(llm, SP, prompts, n):
    sp = SP(temperature=0.0, max_tokens=n, ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs], [
        o.outputs[0].text for o in outs
    ]


def _first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return min(len(x), len(y))


def sanity(llm, SP):
    res = {}
    for bs in (1, 4):
        prompts = PROMPTS[:bs]
        runs = {}
        if a.arm == "C" and a.graphs:
            # Replayed graphs bake the dispatch decision in: the on/off toggle would not
            # take effect, so only mono runs here; eager baselines come from
            # --ref-sanity.
            order = [("mono1", None), ("mono2", None)]
        elif a.arm == "C":
            order = [
                ("base1", False),
                ("base2", False),
                ("mono1", True),
                ("mono2", True),
            ]
        else:
            order = [("base1", None), ("base2", None)]
        for name, on in order:
            if on is not None:
                llm.collective_rpc("mono_live_enable", args=(on,))
            _reset(llm)
            t = time.time()
            ids, texts = _gen(llm, SP, prompts, a.sanity_tokens)
            runs[name] = dict(
                ids=ids, texts=texts, s=time.time() - t, stats=_stats(llm)
            )
            log(
                f"[sanity bs={bs} {name}] {time.time() - t:.1f}s cov="
                f"{(runs[name]['stats'] or {}).get('coverage')}"
            )
            for p, tx in zip(prompts, texts):
                log(f"   {p[:30]!r} -> {tx[:160]!r}")
        if a.arm == "C":
            llm.collective_rpc("mono_live_enable", args=(True,))
        if a.ref_sanity:
            ref = json.load(open(a.ref_sanity))[str(bs)]["runs"]
            for k, v in ref.items():
                runs["ref_" + k] = dict(ids=v["ids"], texts=v["texts"])
        agree = {}
        names = list(runs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                agree[f"{names[i]}~{names[j]}"] = [
                    _first_diff(x, y)
                    for x, y in zip(runs[names[i]]["ids"], runs[names[j]]["ids"])
                ]
        log(f"[sanity bs={bs}] first differing token index per prompt: {agree}")
        res[bs] = dict(runs=runs, first_diff=agree)
    with open(os.path.join(a.out, "sanity.json"), "w") as f:
        json.dump(res, f, indent=1)
    return {bs: r["first_diff"] for bs, r in res.items()}


def gsm8k(llm, SP):
    d = gsm8k_data()
    shots = d["train"][:5]
    prefix = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots
    )
    qs = d["test"][a.gsm8k_start : a.gsm8k_start + a.gsm8k]
    prompts = [prefix + f"Question: {q['question']}\nAnswer:" for q in qs]
    sp = SP(temperature=0.0, max_tokens=256, stop=["Question:", "</s>", "<|im_end|>"])
    _reset(llm)
    t = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t
    st = _stats(llm)
    ks = kf = 0
    gen_tokens = 0
    with open(os.path.join(a.out, "gsm8k_records.jsonl"), "w") as f:
        for i, (q, o) in enumerate(zip(qs, outs)):
            text = o.outputs[0].text
            gen_tokens += len(o.outputs[0].token_ids)
            g = norm_num(q["answer"].split("####")[-1])
            ps, pf = strict(text), flexible(text)
            okS, okF = same(ps, g), same(pf, g)
            ks += okS
            kf += okF
            f.write(
                json.dumps(
                    dict(
                        i=a.gsm8k_start + i,
                        gold=g,
                        strict=ps,
                        flex=pf,
                        ok_strict=okS,
                        ok_flex=okF,
                        ntok=len(o.outputs[0].token_ids),
                        text=text,
                    )
                )
                + "\n"
            )
    n = len(qs)
    dec_out = sum(max(0, len(o.outputs[0].token_ids) - 1) for o in outs)
    if st:
        st["decode_tokens_from_outputs"] = dec_out
        st["coverage_dev"] = st["dev_mono_tokens"] / max(1, dec_out)
        st["poll_errors"] = llm.collective_rpc("mono_live_poll_errors")
    r = dict(
        n=n,
        strict=ks / n,
        strict_se=stderr(ks, n),
        flex=kf / n,
        flex_se=stderr(kf, n),
        seconds=dt,
        gen_tokens=gen_tokens,
        mono=st,
    )
    log(
        f"=== GSM8K arm {a.arm} n={n}: strict {ks}/{n} = {ks / n:.4f} +- "
        f"{stderr(ks, n):.4f}; "
        f"flexible {kf}/{n} = {kf / n:.4f} +- {stderr(kf, n):.4f}; {dt:.0f}s, "
        f"{gen_tokens} gen tokens"
    )
    if st:
        log(
            f"    mono coverage {st['coverage']:.4f} (mono_tokens {st['mono_tokens']} "
            "/ decode_tokens "
            f"{st['decode_tokens']}; dev_mono_tokens {st['dev_mono_tokens']} / output "
            f"decode tokens {dec_out} = {st['coverage_dev']:.4f}); steps mono "
            f"{st['steps_mono']} "
            f"fallback_decode {st['steps_fallback_decode']} reasons "
            f"{st['fallback_reasons']} by_S {st['mono_steps_by_S']} "
            f"expired {st['expired'][:3]} enabled {st['enabled']} poll_errors(rank0) "
            f"{st['poll_errors'][0]}"
        )
    with open(os.path.join(a.out, "gsm8k_summary.json"), "w") as f:
        json.dump(r, f, indent=1)
    return r


if __name__ == "__main__":
    main()
    sys.stdout.flush()
