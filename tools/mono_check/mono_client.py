# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Async OpenAI-API client for the live-mono server (tools/mono_check/serve_mono.sh).
#   python mono_client.py health --url http://127.0.0.1:8199
#   python mono_client.py load --url ... --out DIR [--phases 1:90,4:150,8:180,16:240]
#   python mono_client.py gsm8k --url ... --out DIR [--n 1319] [--concurrency 16]
#   python mono_client.py rescore --records <out_dir>/gsm8k_records.jsonl
#       (scoring-code self-check)
# GSM8K prompt format / scoring are copied verbatim from tools/mono_check/run_live.py
# (5-shot, first 5 train exemplars, "Question: q\nAnswer: a\n\n", greedy, max 256, stop
# Question:/</s>/<|im_end|>, strict "#### N").
import argparse
import asyncio
import contextlib
import json
import os
import random
import time

import aiohttp

MODEL = "glm52"


# GSM8K data / prompt / scoring: tools/mono_check/gsm8k.py (shared with run_live.py)
from gsm8k import (  # noqa: E402
    STOP,
    flexible,
    gold,
    gsm8k_data,
    gsm_prefix,
    same,
    stderr,
    strict,
)


# ----------------------------------------------------------------- HTTP helpers
def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


async def rpc(sess, url, method, timeout=120):
    async with sess.post(
        f"{url}/collective_rpc",
        json=dict(method=method, timeout=timeout),
        timeout=aiohttp.ClientTimeout(total=timeout + 30),
    ) as r:
        r.raise_for_status()
        return (await r.json())["results"]


async def metrics(sess, url):
    async with sess.get(f"{url}/metrics") as r:
        txt = await r.text()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for key in (
            "vllm:generation_tokens_total",
            "vllm:prompt_tokens_total",
            "vllm:time_to_first_token_seconds_count",
            "vllm:request_success_total",
            "vllm:num_preemptions_total",
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_queries_total",
        ):
            if line.startswith(key):
                with contextlib.suppress(ValueError):
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


async def health_snapshot(sess, url):
    async with sess.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=30)) as r:
        ok = r.status == 200
    h = await rpc(sess, url, "mono_health")
    m = await metrics(sess, url)
    hashes = {x["dispatch"]["hash"] for x in h}
    steps = {json.dumps(x.get("kernel_step_counters"), sort_keys=True) for x in h}
    dev_steps = {x.get("dev_mono_steps") for x in h}
    summ = dict(
        health_ok=ok,
        ranks=len(h),
        dispatch_hash_uniform=len(hashes) == 1,
        dispatch_calls=h[0]["dispatch"]["calls"],
        dispatch_by_mode=h[0]["dispatch"]["by_mode"],
        kernel_steps_uniform=len(steps) == 1,
        kernel_step_counters=h[0].get("kernel_step_counters"),
        dev_mono_steps_uniform=len(dev_steps) == 1,
        dev_mono_tokens=h[0].get("dev_mono_tokens"),
        dev_mono_steps=h[0].get("dev_mono_steps"),
        poll_error_total=sum(x.get("poll_error_total", 0) for x in h),
        poll_error_counts={
            x["rank"]: x.get("poll_error_counts")
            for x in h
            if x.get("poll_error_total")
        },
        enabled=[x.get("enabled") for x in h],
        sched=h[0]["sched"],
        host=h[0].get("host"),
        metrics=m,
    )
    return summ, h


# ----------------------------------------------------------------- requests
class Stats:
    def __init__(self):
        self.recs = []

    def add(self, **r):
        self.recs.append(r)


async def completion(
    sess,
    url,
    prompt,
    max_tokens,
    stream,
    abort_after=None,
    temperature=0.0,
    stop=None,
    timeout=600,
):
    """Returns dict(text, ntok, finish, aborted, error, latency, ttft)."""
    body = dict(
        model=MODEL,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0 if temperature == 0 else 0.95,
        stream=stream,
    )
    if stop:
        body["stop"] = stop
    if stream:
        body["stream_options"] = dict(include_usage=True)
    t0 = time.time()
    ttft = None
    res = dict(text="", ntok=None, finish=None, aborted=False, error=None)
    try:
        async with sess.post(
            f"{url}/v1/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as r:
            if r.status != 200:
                res["error"] = f"http {r.status}: {(await r.text())[:200]}"
                return res
            if not stream:
                j = await r.json()
                c = j["choices"][0]
                res.update(
                    text=c["text"],
                    finish=c.get("finish_reason"),
                    ntok=j["usage"]["completion_tokens"],
                    ptok=j["usage"]["prompt_tokens"],
                )
            else:
                nchunks = 0
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    j = json.loads(data)
                    if j.get("choices"):
                        c = j["choices"][0]
                        if c.get("text"):
                            if ttft is None:
                                ttft = time.time() - t0
                            res["text"] += c["text"]
                            nchunks += 1
                        if c.get("finish_reason"):
                            res["finish"] = c["finish_reason"]
                    if j.get("usage"):
                        res["ntok"] = j["usage"]["completion_tokens"]
                        res["ptok"] = j["usage"]["prompt_tokens"]
                    if abort_after is not None and nchunks >= abort_after:
                        res["aborted"] = True
                        # client disconnect mid-stream -> server aborts the request
                        r.close()
                        break
    except asyncio.TimeoutError:
        res["error"] = "timeout"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    res["latency"] = time.time() - t0
    res["ttft"] = ttft
    return res


TINY = [
    ("The capital of France is", ["Paris"]),
    ("The largest planet in our solar system is", ["Jupiter"]),
    ("Water is made of hydrogen and", ["oxygen"]),
    ("The author of 'Romeo and Juliet' is William", ["Shakespeare"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("Q: What is 12 times 12?\nA:", ["144"]),
    ("The first president of the United States was George", ["Washington"]),
    ("Translate to French: 'thank you' ->", ["merci", "Merci"]),
]


def make_jobs(d, rng, tok_len):
    """Mixed request generator. Kinds: tiny (expected substring), gsm5 (5-shot, ~0.9k
    tok, scored), gsmlong (N-shot ~3k tok, sparse top-2048 path active, scored), free
    (sampled, unscored)."""
    pre5 = gsm_prefix(d, 5)
    prelong = gsm_prefix(d, tok_len["long_shots"])
    test = d["test"]

    def job():
        u = rng.random()
        stream = rng.random() < 0.5
        abort = rng.randint(2, 20) if (stream and rng.random() < 0.2) else None
        if u < 0.25:
            p, exp = rng.choice(TINY)
            return dict(
                kind="tiny",
                prompt=p,
                expect=exp,
                max_tokens=rng.choice([8, 16, 32, 96]),
                stream=stream,
                abort_after=abort,
            )
        if u < 0.55:
            q = rng.choice(test)
            return dict(
                kind="gsm5",
                prompt=pre5 + f"Question: {q['question']}\nAnswer:",
                gold=gold(q),
                max_tokens=256,
                stop=STOP,
                stream=stream,
                abort_after=abort,
            )
        if u < 0.80:
            q = rng.choice(test)
            return dict(
                kind="gsmlong",
                prompt=prelong + f"Question: {q['question']}\nAnswer:",
                gold=gold(q),
                max_tokens=256,
                stop=STOP,
                stream=stream,
                abort_after=abort,
            )
        p = rng.choice(
            [
                "Write a short story about a lighthouse keeper.",
                "Explain how a transistor works.",
                "def quicksort(arr):",
                "List ten animals that live in the ocean:",
            ]
        )
        return dict(
            kind="free",
            prompt=p,
            max_tokens=rng.choice([64, 200, 400]),
            stream=stream,
            abort_after=abort,
            temperature=0.7,
        )

    return job


def score(job, res):
    if res.get("error") or res.get("aborted"):
        return None
    t = res["text"]
    if job["kind"] == "tiny":
        return any(e in t for e in job["expect"])
    if job["kind"] in ("gsm5", "gsmlong"):
        return same(strict(t), job["gold"])
    return len(t.strip()) > 0


async def tokenize_len(sess, url, prompt):
    async with sess.post(f"{url}/tokenize", json=dict(model=MODEL, prompt=prompt)) as r:
        return (await r.json())["count"]


async def run_load(a):
    d = gsm8k_data()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    conn = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=conn) as sess:
        # pick the long-prompt shot count so prompts are ~3k tokens (> top-k 2048)
        # (longest test question ~ +250 tokens; max_model_len 4096 must hold prompt +
        # 256 output)
        shots = 12
        n = await tokenize_len(
            sess, a.url, gsm_prefix(d, shots) + "Question: x\nAnswer:"
        )
        while shots < 40:
            n2 = await tokenize_len(
                sess, a.url, gsm_prefix(d, shots + 1) + "Question: x\nAnswer:"
            )
            if n2 > a.long_target:
                break
            shots, n = shots + 1, n2
        n5 = await tokenize_len(sess, a.url, gsm_prefix(d, 5) + "Question: x\nAnswer:")
        log(f"long prompt: {shots} shots = {n} tokens; 5-shot = {n5} tokens")
        job = make_jobs(d, rng, dict(long_shots=shots))
        base, _ = await health_snapshot(sess, a.url)
        log(
            "initial health:",
            json.dumps(
                {
                    k: base[k]
                    for k in (
                        "health_ok",
                        "dispatch_hash_uniform",
                        "kernel_steps_uniform",
                        "poll_error_total",
                        "sched",
                    )
                }
            ),
        )
        out = dict(long_prompt_tokens=n, short5_tokens=n5, phases=[], initial=base)
        recf = open(os.path.join(a.out, "load_records.jsonl"), "w")
        for spec in a.phases.split(","):
            conc, dur = (int(x) for x in spec.split(":"))
            h0, _ = await health_snapshot(sess, a.url)
            t_end = time.time() + dur
            recs = []
            last_done = [time.time()]

            async def worker(wid):
                while time.time() < t_end:
                    j = job()
                    r = await completion(
                        sess,
                        a.url,
                        j["prompt"],
                        j["max_tokens"],
                        j["stream"],
                        j.get("abort_after"),
                        j.get("temperature", 0.0),
                        j.get("stop"),
                        timeout=a.req_timeout,
                    )
                    ok = score(j, r)
                    last_done[0] = time.time()
                    rec = dict(
                        phase=conc,
                        kind=j["kind"],
                        stream=j["stream"],
                        abort_after=j.get("abort_after"),
                        ok=ok,
                        **{
                            k: r.get(k)
                            for k in (
                                "ntok",
                                "ptok",
                                "finish",
                                "aborted",
                                "error",
                                "latency",
                                "ttft",
                            )
                        },
                        text=r["text"][:300],
                    )
                    recs.append(rec)
                    recf.write(json.dumps(rec) + "\n")

            async def watchdog():
                while time.time() < t_end + a.req_timeout:
                    await asyncio.sleep(10)
                    if time.time() - last_done[0] > a.hang_s:
                        log(
                            f"WATCHDOG: no request finished in {a.hang_s}s (phase "
                            f"c={conc})"
                        )
                        return True
                return False

            t0 = time.time()
            wd = asyncio.create_task(watchdog())
            await asyncio.gather(*(worker(i) for i in range(conc)))
            wd.cancel()
            el = time.time() - t0
            h1, raw = await health_snapshot(sess, a.url)
            m0, m1 = h0["metrics"], h1["metrics"]
            gen = m1.get("vllm:generation_tokens_total", 0) - m0.get(
                "vllm:generation_tokens_total", 0
            )
            firsts = m1.get("vllm:time_to_first_token_seconds_count", 0) - m0.get(
                "vllm:time_to_first_token_seconds_count", 0
            )
            dec = gen - firsts
            mono = (h1["dev_mono_tokens"] or 0) - (h0["dev_mono_tokens"] or 0)
            by_kind = {}
            for r in recs:
                k = by_kind.setdefault(
                    r["kind"], dict(n=0, scored=0, ok=0, aborted=0, errors=0)
                )
                k["n"] += 1
                k["aborted"] += bool(r["aborted"])
                k["errors"] += bool(r["error"])
                if r["ok"] is not None:
                    k["scored"] += 1
                    k["ok"] += bool(r["ok"])
            ph = dict(
                concurrency=conc,
                seconds=el,
                requests=len(recs),
                errors=sum(bool(r["error"]) for r in recs),
                aborted=sum(bool(r["aborted"]) for r in recs),
                streamed=sum(bool(r["stream"]) for r in recs),
                by_kind=by_kind,
                gen_tokens=gen,
                decode_tokens=dec,
                mono_tokens=mono,
                coverage=mono / dec if dec else None,
                health_after=h1["health_ok"],
                dispatch_hash_uniform=h1["dispatch_hash_uniform"],
                kernel_steps_uniform=h1["kernel_steps_uniform"],
                kernel_step_counters=h1["kernel_step_counters"],
                dev_mono_steps_uniform=h1["dev_mono_steps_uniform"],
                poll_error_total=h1["poll_error_total"],
                poll_error_counts=h1["poll_error_counts"],
                enabled=h1["enabled"],
                dispatch_by_mode=h1["dispatch_by_mode"],
                lat_p50=sorted(r["latency"] for r in recs)[len(recs) // 2]
                if recs
                else None,
                lat_max=max((r["latency"] for r in recs), default=None),
                hang=wd.done() and not wd.cancelled() and wd.result(),
            )
            out["phases"].append(ph)
            ph_log = {k: v for k, v in ph.items() if k != "dispatch_by_mode"}
            log(f"PHASE c={conc}: {json.dumps(ph_log)}")
            recf.flush()
            if ph["errors"] and a.stop_on_error:
                break
        fin, raw = await health_snapshot(sess, a.url)
        out["final"] = fin
        out["final_raw"] = raw
        # quick post-test sanity: deterministic greedy request still coherent
        r = await completion(sess, a.url, "The capital of France is", 32, False)
        out["post_sanity"] = r
        log(
            "post-test sanity:",
            repr(r["text"][:120]),
            "health",
            fin["health_ok"],
            "poll errors",
            fin["poll_error_total"],
        )
        recf.close()
    with open(os.path.join(a.out, "load_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)


async def run_gsm8k(a):
    d = gsm8k_data()
    pre = gsm_prefix(d, 5)
    qs = d["test"][a.start : a.start + a.n]
    os.makedirs(a.out, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    results = [None] * len(qs)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=256)) as sess:
        h0, _ = await health_snapshot(sess, a.url)

        async def one(i, q):
            async with sem:
                for attempt in range(2):
                    r = await completion(
                        sess,
                        a.url,
                        pre + f"Question: {q['question']}\nAnswer:",
                        256,
                        a.stream,
                        stop=STOP,
                        timeout=a.req_timeout,
                    )
                    if not r["error"]:
                        break
                    log(f"q{i} error {r['error']} (attempt {attempt})")
                results[i] = r

        t0 = time.time()
        await asyncio.gather(*(one(i, q) for i, q in enumerate(qs)))
        dt = time.time() - t0
        h1, raw = await health_snapshot(sess, a.url)
    ks = kf = 0
    errs = 0
    with open(os.path.join(a.out, "gsm8k_records.jsonl"), "w") as f:
        for i, (q, r) in enumerate(zip(qs, results)):
            text = r["text"]
            errs += bool(r["error"])
            g = gold(q)
            ps, pf = strict(text), flexible(text)
            okS, okF = same(ps, g), same(pf, g)
            ks += okS
            kf += okF
            f.write(
                json.dumps(
                    dict(
                        i=a.start + i,
                        gold=g,
                        strict=ps,
                        flex=pf,
                        ok_strict=okS,
                        ok_flex=okF,
                        ntok=r["ntok"],
                        error=r["error"],
                        text=text,
                    )
                )
                + "\n"
            )
    n = len(qs)
    m0, m1 = h0["metrics"], h1["metrics"]
    gen = m1.get("vllm:generation_tokens_total", 0) - m0.get(
        "vllm:generation_tokens_total", 0
    )
    firsts = m1.get("vllm:time_to_first_token_seconds_count", 0) - m0.get(
        "vllm:time_to_first_token_seconds_count", 0
    )
    dec = gen - firsts
    mono = (h1["dev_mono_tokens"] or 0) - (h0["dev_mono_tokens"] or 0)
    res = dict(
        n=n,
        strict=ks / n,
        strict_se=stderr(ks, n),
        flex=kf / n,
        flex_se=stderr(kf, n),
        seconds=dt,
        errors=errs,
        decode_tokens=dec,
        mono_tokens=mono,
        coverage=mono / dec if dec else None,
        health=h1,
        client_decode_tokens=sum(max(0, (r["ntok"] or 0) - 1) for r in results),
    )
    log(
        f"=== GSM8K server n={n}: strict {ks}/{n} = {ks / n:.4f} +- "
        f"{stderr(ks, n):.4f}; flex {kf}/{n} = {kf / n:.4f}; "
        f"{dt:.0f}s; errors {errs}; coverage {res['coverage']} ({mono}/{dec}); poll "
        f"errors {h1['poll_error_total']}; "
        f"dispatch uniform {h1['dispatch_hash_uniform']} kernel steps uniform "
        f"{h1['kernel_steps_uniform']}"
    )
    with open(os.path.join(a.out, "gsm8k_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


async def run_health(a):
    async with aiohttp.ClientSession() as sess:
        s, raw = await health_snapshot(sess, a.url)
    print(json.dumps(s if not a.raw else raw, indent=1, default=str))


def rescore(a):
    n = bad = 0
    for line in open(a.records):
        r = json.loads(line)
        q = r["text"]
        okS = same(strict(q), r["gold"])
        n += 1
        bad += okS != r["ok_strict"]
    print(f"rescored {n} records, {bad} mismatches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["health", "load", "gsm8k", "rescore"])
    ap.add_argument("--url", default="http://127.0.0.1:8199")
    ap.add_argument("--out", default="")
    ap.add_argument("--phases", default="1:90,4:150,8:180,16:240")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--req-timeout", type=float, default=900)
    ap.add_argument("--hang-s", type=float, default=300)
    ap.add_argument("--long-target", type=int, default=3100)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--records", default="")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    if a.cmd == "rescore":
        return rescore(a)
    asyncio.run(dict(health=run_health, load=run_load, gsm8k=run_gsm8k)[a.cmd](a))


if __name__ == "__main__":
    main()
