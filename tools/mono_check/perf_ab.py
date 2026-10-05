# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Graph-vs-graph decode latency A/B. Offline LLM TP8, prefix caching off, both arms HIP
# graphs (mode 0, FULL_DECODE_ONLY, capture 1,2,4,5,6,8).
#   arm A: vLLM as shipped; arm C: live MonoKernel (default indexer_only).
# Per point (batch x ctx): 1 warm-up + R repeats of B distinct prompts, 256 output
# tokens, ignore_eos, greedy. Metrics: per-request TPOT = (last_token_ts -
# first_token_ts)/(n-1) (engine-core timestamps), per-step GPU time of pure-decode steps
# (CUDA events, rank 0, perf_ext), decode tok/s = B / step time (B / TPOT under MTP), C
# coverage = kernel rows / decode tokens (MTP: / decode-or-verify rows), tok/req-step =
# emitted tokens per request per pure step (1 + accepted under MTP).
#   timeout 5400 python tools/mono_check/perf_ab.py --model <GLM-5.2 checkpoint> \
#       --arm A --out <out_dir> [--profile long:1] [--batches 1,2,4,8]
import argparse
import json
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["A", "C"])
ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True, help="local GLM-5.2 checkpoint directory")
ap.add_argument("--batches", default="1,2,4,8")
ap.add_argument("--ctx", default="short,long")
ap.add_argument("--short-len", type=int, default=128)
ap.add_argument("--long-len", type=int, default=3000)
ap.add_argument("--out-len", type=int, default=256)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument(
    "--indexer-mode", default="indexer_only", choices=["attn", "indexer_only"]
)
ap.add_argument("--poll-limit", type=int, default=20_000_000)
ap.add_argument(
    "--profile",
    default="",
    help="comma list of ctx:batch points to torch-profile, e.g. long:1,long:8",
)
ap.add_argument("--profile-tokens", type=int, default=12)
ap.add_argument(
    "--eager",
    action="store_true",
    help="enforce_eager (no HIP graphs); arm C then runs with step_sync=True "
    "(live-mode eager default)",
)
ap.add_argument(
    "--spec-k",
    type=int,
    default=0,
    help="vLLM native MTP with K draft tokens; capture "
    "sizes / kernel widths become the multiples of 1 + K among 1,2,4,5,6,8,10",
)
ap.add_argument(
    "--live-json",
    default="{}",
    help="extra LiveConfig fields as JSON, e.g. '{\"fused_indexer\": false}'",
)
a = ap.parse_args()

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "0")
os.environ["MONO_PERF_STEPS"] = "1"
# "warn": log and count every expired step and continue (MONO_LIVE_FAILSTOP=0 in the
# environment still selects the unchecked path)
os.environ.setdefault("MONO_LIVE_FAILSTOP", "warn")
# per-rank step timestamps + health per rep
ALLRANKS = os.environ.get("MONO_PERF_ALLRANKS", "0") == "1"
SIZES = [1, 2, 4, 5, 6, 8]


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def _f(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    global SIZES
    os.makedirs(a.out, exist_ok=True)
    if (
        a.spec_k > 0
        # verify steps carry B * (1 + k) rows: graph sizes and widths must be multiples
        # of 1 + k
    ):
        # S12 spills
        SIZES = [w for w in (1, 2, 4, 5, 6, 8, 10) if w % (1 + a.spec_k) == 0]
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=a.model,
        tensor_parallel_size=8,
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        max_num_seqs=8,
        trust_remote_code=True,
        kv_cache_dtype="auto",
        block_size=16,
        num_gpu_blocks_override=8192,
        enable_prefix_caching=False,
        disable_log_stats=False,
        worker_extension_cls="tools.mono_check.harness.perf_ext.MonoPerfWorkerExtension",
        compilation_config=dict(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=SIZES
        ),
    )
    if a.spec_k > 0:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": a.spec_k}
    if a.eager:
        kw.pop("compilation_config")
        kw["enforce_eager"] = True
    if a.profile:
        kw["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=os.path.join(a.out, "traces"),
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=False,
            torch_profiler_use_gzip=True,
        )
    if a.arm == "C":
        cfg = dict(
            ckpt=a.model,
            layers=list(range(3, 78)),
            sizes=SIZES,
            poll_limit=a.poll_limit,
            indexer_mode=a.indexer_mode,
            step_sync=a.eager,
            max_model_len=4096,
        )
        cfg.update(json.loads(a.live_json))
        os.environ["MONO_LIVE_PREINSTALL"] = json.dumps(cfg)
    meta = dict(
        argv=sys.argv,
        env={k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "MONO_"))},
        llm=kw,
    )
    t0 = time.time()
    llm = LLM(**kw)
    meta["init_s"] = time.time() - t0
    # host/namespace pids of every worker, to match dmesg GPU-fault pids to this run
    if ALLRANKS:
        try:
            meta["worker_nspids"] = llm.collective_rpc(
                lambda self: [
                    ln.split()[1:]
                    for ln in open("/proc/self/status")
                    if ln.startswith("NSpid")
                ][0]
            )
        except Exception as e:  # noqa: BLE001
            meta["worker_nspids"] = repr(e)
    log(f"LLM init {meta['init_s']:.0f}s")
    tok = llm.get_tokenizer()
    # distinct natural-text prompts: slices of the GSM8K train set token stream
    from gsm8k import gsm8k_data

    text = "".join(
        r["question"] + "\n" + r["answer"] + "\n\n" for r in gsm8k_data()["train"]
    )
    ids = tok.encode(text, add_special_tokens=False)
    off = [0]

    def prompts(n, L):
        out = []
        for _ in range(n):
            out.append(TokensPrompt(prompt_token_ids=ids[off[0] : off[0] + L]))
            off[0] += L
        return out

    sp = SamplingParams(temperature=0.0, max_tokens=a.out_len, ignore_eos=True)
    results = []
    lens = dict(short=a.short_len, long=a.long_len)
    for ctx in a.ctx.split(","):
        for B in [int(x) for x in a.batches.split(",")]:
            pt = dict(arm=a.arm, ctx=ctx, prompt_len=lens[ctx], batch=B, reps=[])
            for rep in range(a.repeats + 1):  # rep 0 = warm-up
                hs0 = llm.collective_rpc("mono_health") if a.arm == "C" else None
                h0 = hs0[0] if hs0 else None
                llm.collective_rpc("mono_perf_start")
                t = time.time()
                outs = llm.generate(prompts(B, lens[ctx]), sp, use_tqdm=False)
                wall = time.time() - t
                all_steps = llm.collective_rpc("mono_perf_stop")
                steps = all_steps[0]["steps"]
                hs1 = llm.collective_rpc("mono_health") if a.arm == "C" else None
                h1 = hs1[0] if hs1 else None
                tpots = []
                for o in outs:
                    m = o.metrics
                    n = len(o.outputs[0].token_ids)
                    assert n == a.out_len, n
                    tpots.append((m.last_token_ts - m.first_token_ts) / (n - 1) * 1000)
                dec = [
                    s["ms"]
                    for s in steps
                    if s["pure"] and s["prev_pure"] and s["n_req"] == B
                ]
                r = dict(
                    rep=rep,
                    wall_s=wall,
                    tpot_req_ms=tpots,
                    tpot_mean=statistics.mean(tpots),
                    step_ms_p50=pct(dec, 0.5),
                    step_ms_p90=pct(dec, 0.9),
                    step_ms_mean=statistics.mean(dec) if dec else None,
                    n_decode_steps=len(dec),
                    n_steps=len(steps),
                    text0=o.outputs[0].text[:80] if rep == 1 else None,
                )
                # decode / MTP-verify rows the scheduler ran (rank 0,
                # perf_ext; the first record is the prefill step) and the emitted tokens
                # per request per pure step (1 without MTP; 1 + accepted with it,
                # slightly low: the last step's tokens past max_tokens are dropped)
                # decode/verify rows, mixed steps included
                dec_rows = sum(s.get("dec_rows", 0) for s in steps)
                req_steps = sum(s["n_req"] for s in steps if s["pure"])
                r["decode_rows"] = dec_rows
                r["tok_per_req_step"] = (
                    B * (a.out_len - 1) / req_steps if req_steps else None
                )
                if a.arm == "C":
                    # rows the kernel ran (drafts included)
                    toks = h1["dev_mono_tokens"] - h0["dev_mono_tokens"]
                    r["mono_tokens"] = toks
                    # without MTP unchanged (kernel rows / decode tokens); with MTP that
                    # ratio counted (1 + k) verify rows per (1 + accepted) emitted
                    # tokens (> 1): use kernel rows / decode-or-verify rows instead
                    if a.spec_k:
                        r["coverage"] = toks / dec_rows if dec_rows else None
                    else:
                        r["coverage"] = toks / (B * (a.out_len - 1))
                    r["poll_errors"] = h1["poll_error_total"]
                    # every rank: sticky (S:stage) keys, warn-mode
                    # incidents with their kernel steps
                    r["poll_by_rank"] = [
                        dict(
                            keys=sorted(h.get("poll_error_counts", {})),
                            incidents=(h.get("poll_incidents") or 0)
                            - (g.get("poll_incidents") or 0),
                            log=(h.get("poll_incident_log") or [])[
                                len(g.get("poll_incident_log") or []) :
                            ],
                            inband=h.get("inband_trips", 0),
                            steps=h.get("kernel_step_counters"),
                        )
                        for g, h in zip(hs0, hs1)
                    ]
                if ALLRANKS:
                    # per step: GPU ms on each rank, and the host entry skew across
                    # ranks (max - min, ms)
                    n = min(len(x["steps"]) for x in all_steps)
                    r["rank_step_ms"] = [
                        [round(x["steps"][j]["ms"], 3) for x in all_steps]
                        for j in range(n)
                    ]
                    r["host_skew_ms"] = [
                        round(
                            (
                                max(x["steps"][j]["t_in"] for x in all_steps)
                                - min(x["steps"][j]["t_in"] for x in all_steps)
                            )
                            / 1e6,
                            3,
                        )
                        for j in range(n)
                    ]
                    r["late_rank"] = [
                        max(
                            range(len(all_steps)),
                            key=lambda q: all_steps[q]["steps"][j]["t_in"],
                        )
                        for j in range(n)
                    ]
                    r["token_ids"] = [list(o.outputs[0].token_ids) for o in outs]
                pt["reps"].append(r)
                log(
                    f"{a.arm} {ctx} B={B} rep{rep}: tpot {r['tpot_mean']:.2f} ms, step "
                    f"p50 {_f(r['step_ms_p50'])} "
                    f"p90 {_f(r['step_ms_p90'])} ms ({len(dec)} decode steps), "
                    "tok/req-step "
                    f"{_f(r['tok_per_req_step'], 3)}, cov {r.get('coverage')}"
                )
            reps = pt["reps"][1:]
            all_tp = [x for r in reps for x in r["tpot_req_ms"]]
            all_st_p50 = [
                r["step_ms_p50"] for r in reps if r["step_ms_p50"] is not None
            ]
            all_st_p90 = [
                r["step_ms_p90"] for r in reps if r["step_ms_p90"] is not None
            ]
            tprs = [r["tok_per_req_step"] for r in reps if r["tok_per_req_step"]]
            covs = [r["coverage"] for r in reps if r.get("coverage") is not None]
            step_p50 = statistics.median(all_st_p50) if all_st_p50 else None
            pt.update(
                tpot_mean=statistics.mean(all_tp),
                tpot_p50=pct(all_tp, 0.5),
                tpot_p90=pct(all_tp, 0.9),
                step_p50=step_p50,
                step_p90=statistics.median(all_st_p90) if all_st_p90 else None,
                # under MTP a step emits 1 + accepted tokens per request: derive
                # throughput from TPOT
                tok_s=B
                * 1000
                / (
                    statistics.mean(all_tp)
                    if a.spec_k or step_p50 is None
                    else step_p50
                ),
                tok_s_tpot=B * 1000 / statistics.mean(all_tp),
                tok_per_req_step=statistics.mean(tprs) if tprs else None,
                coverage=statistics.mean(covs) if a.arm == "C" and covs else None,
            )
            results.append(pt)
            with open(os.path.join(a.out, "perf.json"), "w") as f:
                json.dump(dict(meta=meta, results=results), f, indent=1, default=str)
    if a.profile:
        for spec in a.profile.split(","):
            ctx, B = spec.split(":")
            B = int(B)
            llm.generate(
                prompts(B, lens[ctx]),
                SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
                use_tqdm=False,
            )
            # profile a generate whose prefill happens before start: submit, let it run,
            # profile the middle
            llm.start_profile()
            llm.generate(
                prompts(B, lens[ctx]),
                SamplingParams(
                    temperature=0.0, max_tokens=a.profile_tokens, ignore_eos=True
                ),
                use_tqdm=False,
            )
            llm.stop_profile()
            d = os.path.join(a.out, "traces")
            log(
                f"profiled {ctx}:{B}; trace files now: "
                f"{sorted(os.listdir(d)) if os.path.isdir(d) else None}"
            )
    log("SUMMARY")
    for pt in results:
        log(
            f"{pt['arm']} {pt['ctx']:5s} B={pt['batch']}: TPOT mean "
            f"{pt['tpot_mean']:.2f} p50 {pt['tpot_p50']:.2f} "
            f"p90 {pt['tpot_p90']:.2f} ms | step p50 {_f(pt['step_p50'])} p90 "
            f"{_f(pt['step_p90'])} ms | "
            f"{pt['tok_s']:.0f} tok/s | tok/req-step {_f(pt['tok_per_req_step'], 3)} | "
            f"cov {pt['coverage']}"
        )
    try:
        llm.llm_engine.engine_core.shutdown(timeout=60)
    except TypeError:
        llm.llm_engine.engine_core.shutdown()
    log("SHUTDOWN")


if __name__ == "__main__":
    main()
