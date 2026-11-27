#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprob drift: does a candidate build change a model's outputs by more than
reruns of the reference build differ from each other?

`record` scores one run against a running server and saves it. `compare` decides
from the runs of two builds whether they differ. README.md says how.

Usage, against a new server with empty caches for each run (see README.md):
    python -m tests.evals.logprob_drift.logprob_drift_eval record --out ref/1.npz
    python -m tests.evals.logprob_drift.logprob_drift_eval compare \\
        --reference ref/*.npz --candidate new/*.npz
"""

import argparse
import asyncio
import functools
import hashlib
import io
import itertools
import json
import math
import os
import secrets
import sys
import zipfile
from pathlib import Path

import aiohttp
import numpy as np
import requests

TOP_K = 20
CONTEXT = 1024  # tokens per probe chunk; the second half is scored, as in llama.cpp
CHUNKS = 32
NUM_QUESTIONS = 250
NUM_SHOTS = 5
TRAFFIC_TOKENS = 256
DECODE_TOKENS = 128
STOP = ["Question", "Assistant:", "<|separator|>"]  # as in tests/evals/gsm8k
WIKI_URL = (
    "https://huggingface.co/datasets/ggml-org/ci/resolve/"
    "927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip"
)
WIKI_MEMBER = "wikitext-2-raw/wiki.test.raw"
WIKI_SHA = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
# Reference and candidate runs after which the builds are compared.
STAGES = ((2, 1), (4, 2), (5, 4))
CHANCE = 1e-2  # how often identical builds may read as changed
TAIL_FLOOR = 1e-9
FORMAT = 3


# Recording


async def record(base_url: str, seed: int, setup: dict) -> dict:
    """One run against the server at base_url.

    The seed shuffles the order requests are sent in, so batches differ per run.
    setup holds anything that must match between runs to compare them, such as
    the server's arguments.
    """
    model = get(f"{base_url}/v1/models")["data"][0]["id"]
    rng = np.random.default_rng(seed)
    prompts = gsm8k_prompts()
    # The server's own tokens. About 4 characters a token, so this text holds
    # twice the tokens needed.
    text = wiki_text()[: 8 * CHUNKS * CONTEXT]
    body = {"model": model, "prompt": text, "add_special_tokens": False}
    response = requests.post(f"{base_url}/tokenize", json=body, timeout=120)
    response.raise_for_status()
    ids = response.json()["tokens"]
    chunks = np.array(ids[: CHUNKS * CONTEXT], np.int32).reshape(CHUNKS, CONTEXT)

    timeout = aiohttp.ClientTimeout(total=3600)
    # Send everything at once, each request on its own connection: the server
    # closes idle ones, and a request sent on one as it closes is lost.
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    generate = {
        "logprobs": TOP_K,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def complete(body: dict) -> dict:
            body = {"model": model, "temperature": 0.0, **body}
            async with session.post(f"{base_url}/v1/completions", json=body) as r:
                if not r.ok:
                    raise RuntimeError(f"{r.status} from the server: {await r.text()}")
                return (await r.json())["choices"][0]

        # First, one at a time on an idle server with an empty prefix cache, so
        # that each decodes alone, the same way in every run.
        decode = [
            await complete(
                {"prompt": c[: CONTEXT // 2].tolist(), "max_tokens": DECODE_TOKENS}
                | generate
            )
            for c in chunks
        ]
        traffic = await shuffled(
            [
                {
                    "prompt": p,
                    "max_tokens": TRAFFIC_TOKENS,
                    "stop": STOP,
                }
                | generate
                for p in prompts
            ],
            complete,
            rng,
        )
        probes = await shuffled(
            [
                {"prompt": c.tolist(), "max_tokens": 1, "prompt_logprobs": TOP_K}
                for c in chunks
            ],
            complete,
            rng,
        )

    run = (
        generated_arrays(traffic, "traffic", TRAFFIC_TOKENS)
        | generated_arrays(decode, "decode", DECODE_TOKENS)
        | probe_arrays(probes, chunks)
    )
    prompt_ids = json.dumps([c["prompt_token_ids"] for c in traffic]).encode()
    run["meta"] = {
        "format": FORMAT,
        "vllm": get(f"{base_url}/version")["version"],
        "seed": seed,
        "model": model,
        "setup": setup,
        "gpu": gpu_name(),
        "data": {
            "prompts": hashlib.sha256(prompt_ids).hexdigest(),
            "text": WIKI_SHA,
            "top_k": TOP_K,
        },
    }
    return run


def get(url: str) -> dict:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


async def shuffled(bodies: list[dict], complete, rng: np.random.Generator) -> list:
    """The responses to bodies, sent in a random order, in bodies' order."""
    order = rng.permutation(len(bodies))
    responses = await asyncio.gather(*(complete(bodies[i]) for i in order))
    return [responses[i] for i in np.argsort(order)]


def generated_arrays(choices: list[dict], part: str, length: int) -> dict:
    n = len(choices)
    out = {
        "tokens": np.full((n, length), -1, np.int32),
        "ids": np.full((n, length, TOP_K), -1, np.int32),
        "lp": np.full((n, length, TOP_K), -np.inf, np.float32),
        "lengths": np.zeros(n, np.int32),
        "text": np.array([choice["text"] for choice in choices]),
    }
    for i, choice in enumerate(choices):
        tokens, tops = choice["token_ids"], choice["logprobs"]["top_logprobs"]
        if len(tops) != len(tokens):
            raise ValueError(
                f"the server returned {len(tops)} logprobs for {len(tokens)} "
                "generated tokens; this build can't be scored"
            )
        out["lengths"][i] = len(tokens)
        out["tokens"][i, : len(tokens)] = tokens
        for t, top in enumerate(tops):
            row = {int(k.removeprefix("token_id:")): v for k, v in top.items()}
            put(out["ids"][i, t], out["lp"][i, t], row)
    return {f"{part}_{k}": v for k, v in out.items()}


def probe_arrays(choices: list[dict], chunks: np.ndarray) -> dict[str, np.ndarray]:
    first = CONTEXT // 2
    scored = CONTEXT - first
    out = {
        "probe_tokens": chunks[:, first:],
        "probe_ids": np.full((CHUNKS, scored, TOP_K), -1, np.int32),
        "probe_lp": np.full((CHUNKS, scored, TOP_K), -np.inf, np.float32),
        # Each scored token's own logprob, which vLLM returns whatever its rank.
        "probe_own": np.zeros((CHUNKS, scored), np.float32),
    }
    for i, choice in enumerate(choices):
        for t in range(first, CONTEXT):
            row = {
                int(k): v["logprob"] for k, v in choice["prompt_logprobs"][t].items()
            }
            put(out["probe_ids"][i, t - first], out["probe_lp"][i, t - first], row)
            out["probe_own"][i, t - first] = row[int(chunks[i, t])]
    return out


def put(ids_out: np.ndarray, lp_out: np.ndarray, logprobs: dict[int, float]) -> None:
    """Write one position's top-k ids and logprobs, best first."""
    ranked = sorted(logprobs.items(), key=lambda kv: -kv[1])[:TOP_K]
    for k, (token, lp) in enumerate(ranked):
        ids_out[k], lp_out[k] = token, lp


def gpu_name() -> str:
    """The GPU type. On NVIDIA GPUs this reads NVML, so this process holds no
    GPU memory while the server runs."""
    try:
        from vllm.platforms import current_platform

        return current_platform.get_device_name()
    except Exception:
        return "unknown"


def gsm8k_prompts() -> list[str]:
    """Few-shot GSM8K prompts as tests/evals/gsm8k builds them, so they share a
    prefix for the prefix cache to reuse."""
    from tests.evals.gsm8k.gsm8k_eval import load_gsm8k_data

    train, test = load_gsm8k_data()
    shots = "".join(
        f"Question: {q['question']}\nAnswer: {q['answer']}\n\n"
        for q in train[:NUM_SHOTS]
    )
    return [f"{shots}Question: {q['question']}\nAnswer:" for q in test[:NUM_QUESTIONS]]


def wiki_text() -> str:
    """llama.cpp's copy of the WikiText-2 test set, downloaded once."""
    path = Path.home() / ".cache" / "vllm" / "evals" / "wiki.test.raw"
    if not path.exists():
        response = requests.get(WIKI_URL, timeout=120)
        response.raise_for_status()
        data = zipfile.ZipFile(io.BytesIO(response.content)).read(WIKI_MEMBER)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != WIKI_SHA:
        raise ValueError(f"{path} is not the expected WikiText-2: delete it")
    return data.decode()


def fresh_caches(root: str) -> dict[str, str]:
    """Environment variables that give a server empty compile and autotune
    caches under root, so that no run reuses another's kernel choices."""
    return {
        "VLLM_CACHE_ROOT": root,
        "TRITON_CACHE_DIR": f"{root}/triton",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": f"{root}/flashinfer",
    }


def save(run: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in run.items() if k != "meta"}
    np.savez_compressed(path, **arrays, meta=json.dumps(run["meta"]))


def load(path: Path) -> dict:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        return arrays | {"meta": json.loads(str(z["meta"]))}


# Comparing


def compare(refs: list[dict], cands: list[dict]) -> dict:
    """Whether the candidate runs differ from the reference runs beyond noise."""
    check(refs, cands)
    runs = refs + cands
    kl = {
        "probes": probe_kls,
        "decode": functools.partial(generated_kls, part="decode"),
        "traffic": functools.partial(generated_kls, part="traffic"),
    }
    # [i][j] is KL(run i || run j) by item, a chunk or a prompt.
    items = {p: np.array([[kl[p](a, b) for b in runs] for a in runs]) for p in kl}
    matrices = {part: kls.mean(axis=-1) for part, kls in items.items()}
    result = decide(matrices, len(refs))
    part = max(result["ratios"], key=result["ratios"].get)
    return {
        "stats": stats(refs[0], cands[0]),
        **result,
        "where": where(items[part], part, len(refs), refs[0], cands[0]),
        # Enough to redo the verdict without the runs.
        "matrices": {part: m.tolist() for part, m in matrices.items()},
    }


def check(refs: list[dict], cands: list[dict]) -> None:
    """Raise unless the runs can be compared."""
    # With as many candidate runs as reference runs, a change from bit-identical
    # reruns scores infinite for the reversed split too, and none is alone on top.
    most = STAGES[-1][0]
    if not 1 <= len(cands) < len(refs) <= most:
        raise ValueError(
            f"{len(refs)} reference and {len(cands)} candidate runs: record 2 to "
            f"{most} runs of the reference and fewer of the candidate"
        )
    for side, runs in (("reference", refs), ("candidate", cands)):
        if len(versions := {run["meta"]["vllm"] for run in runs}) > 1:
            raise ValueError(
                f"the {side} runs come from vLLM {', '.join(sorted(versions))}: "
                "pass the runs of one build per side"
            )
    runs = refs + cands
    if len({run["meta"]["seed"] for run in runs}) < len(runs):
        raise ValueError("two runs share a seed: record each run with its own")
    first = runs[0]["meta"]
    differ = {
        k
        for run in runs
        for k in first.keys() - {"vllm", "seed"}
        if run["meta"].get(k) != first[k]
    }
    if differ:
        raise ValueError(
            f"runs differ in {', '.join(sorted(differ))}: record every run with the "
            "same model, server settings, GPU type and data"
        )
    if any((run["probe_tokens"] != runs[0]["probe_tokens"]).any() for run in runs):
        raise ValueError("runs scored different probe tokens")


def decide(matrices: dict[str, np.ndarray], n: int) -> dict:
    """The verdict from each part's KL matrix between runs, the reference's n
    runs first: [i][j] is KL(run i || run j).

    A split's score is the largest of its ratios, each about 1 when nothing
    changed. The builds separate when the real split scores above every other
    one, which identical builds do 1 time in the number of splits. So the
    verdict is "same" when they don't separate, "changed" when they do with
    enough splits for that chance to be at most CHANCE, and "more" when they do
    with fewer.
    """
    total = len(next(iter(matrices.values())))
    real = tuple(range(n))
    others = (s for s in itertools.combinations(range(total), n) if s != real)
    ratios = np.array(
        [
            [score(m, part, ref) for part, m in matrices.items()]
            for ref in (real, *others)
        ]
    )
    scores = ratios.max(axis=1)
    needed = float(scores[1:].max())  # the real split must beat every other
    if scores[0] <= needed:
        verdict = "same"
    elif 1 / len(scores) <= CHANCE:
        verdict = "changed"
    else:
        verdict = "more"
    return {
        "ratios": dict(zip(matrices, ratios[0].tolist())),
        "score": float(scores[0]),
        "needed": needed,
        "p": int((scores >= scores[0]).sum()) / len(scores),
        "splits": len(scores),
        "runs": (n, total - n),
        "verdict": verdict,
    }


def score(matrix: np.ndarray, part: str, ref: tuple[int, ...]) -> float:
    """Mean KL across the groups over the mean KL between reruns.

    Probe and decode reruns are often bit-identical, so their noise is pooled
    from both groups. Traffic noise comes from the reference group alone:
    random KV-cache corruption makes a candidate's runs disagree with each
    other, and pooling would count that as noise.
    """
    cand = tuple(i for i in range(len(matrix)) if i not in ref)
    groups = (ref, cand) if part != "traffic" else (ref,)
    between = mean(matrix[np.ix_(ref, cand)])  # KL(reference || candidate)
    noise = mean(
        np.concatenate(
            [matrix[np.ix_(g, g)][~np.eye(len(g), dtype=bool)] for g in groups]
        )
    )
    if noise == 0:  # bit-identical reruns: any difference is beyond noise
        return 0.0 if between == 0 else math.inf
    return between / noise


def mean(values: np.ndarray) -> float:
    """The mean, independent of the values' order, so that splits with the same
    KLs tie exactly."""
    return math.fsum(values.ravel().tolist()) / values.size


def probe_kls(a: dict, b: dict) -> np.ndarray:
    """Each chunk's mean KL over its positions."""
    return bucketed_kl(
        a["probe_ids"], a["probe_lp"], b["probe_ids"], b["probe_lp"]
    ).mean(axis=-1)


def generated_kls(a: dict, b: dict, part: str) -> np.ndarray:
    """Each prompt's mean KL over the positions both runs condition alike: up to
    and including the first differing token, or the shorter output's end. A
    prompt with no such position counts as 0."""
    n = np.minimum(a[f"{part}_lengths"], b[f"{part}_lengths"])
    differ = a[f"{part}_tokens"] != b[f"{part}_tokens"]
    before = np.cumsum(differ, axis=1) - differ
    compared = (before == 0) & (np.arange(differ.shape[1]) < n[:, None])
    kl = np.zeros(compared.shape)
    kl[compared] = bucketed_kl(
        a[f"{part}_ids"][compared],
        a[f"{part}_lp"][compared],
        b[f"{part}_ids"][compared],
        b[f"{part}_lp"][compared],
    )
    return kl.sum(axis=1) / np.maximum(compared.sum(axis=1), 1)


def bucketed_kl(
    ids_a: np.ndarray, lp_a: np.ndarray, ids_b: np.ndarray, lp_b: np.ndarray
) -> np.ndarray:
    """KL(a || b) per top-k list on the last axis, a lower bound on the true KL.

    Ids both lists hold count one by one, and the rest of the vocab in two
    buckets: the ids only a lists, and the ids a doesn't list. Merging ids
    never raises KL. b's mass on the first bucket is unknown, but each of its
    ids has at most b's smallest listed probability; the bound takes the mass
    in that range that gives the least KL.
    """
    pa = np.exp(lp_a.astype(np.float64))
    pb = np.exp(lp_b.astype(np.float64))
    listed = pa != 0  # padding ids (-1) have -inf logprobs
    match = (ids_a[..., :, None] == ids_b[..., None, :]) & listed[..., None]
    shared = match.any(axis=-1)
    pb_shared = (match * pb[..., None, :]).sum(axis=-1)
    only_a = listed & ~shared
    a_only = np.where(only_a, pa, 0.0).sum(axis=-1)
    a_shared = np.where(shared, pa, 0.0).sum(axis=-1)
    # Floored alike, so that identical lists give exactly 0.
    a_rest = np.maximum(1 - a_shared - a_only, TAIL_FLOOR)
    b_off = np.maximum(1 - pb_shared.sum(axis=-1), TAIL_FLOOR)  # off shared ids
    # b's mass on the ids only a lists: the KL falls until it reaches its share
    # of b_off in proportion to a's, and b can't give it more than this.
    b_only_max = np.minimum(only_a.sum(axis=-1) * pb.min(axis=-1), 1 - pb.sum(axis=-1))
    b_only = np.minimum(
        b_off * a_only / np.maximum(a_only + a_rest, TAIL_FLOOR),
        np.maximum(b_only_max, TAIL_FLOOR),
    )
    b_rest = np.maximum(b_off - b_only, TAIL_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shared, pa * (np.log(pa) - np.log(pb_shared)), 0.0)
        only = np.where(a_only > 0, a_only * np.log(a_only / b_only), 0.0)
        rest = a_rest * np.log(a_rest / b_rest)
    return terms.sum(axis=-1) + only + rest


def stats(ref: dict, cand: dict) -> dict:
    """How far the first candidate run is from the first reference run."""
    ids_a, ids_b = ref["probe_ids"], cand["probe_ids"]
    kl = bucketed_kl(ids_a, ref["probe_lp"], ids_b, cand["probe_lp"])
    # Outputs are padded with -1 past their end, so outputs of different
    # lengths differ here too.
    differ = ref["traffic_tokens"] != cand["traffic_tokens"]
    # The text's mean negative log-likelihood: higher is a worse model.
    nll = [-float(run["probe_own"].mean()) for run in (ref, cand)]
    return {
        "kl": float(kl.mean()),
        "nll_change": nll[1] / nll[0] - 1,
        "same_top": float((ids_a[..., 0] == ids_b[..., 0]).mean()),
        "outputs_differ": float(differ.any(axis=1).mean()),
    }


def where(kls: np.ndarray, part: str, n: int, ref: dict, cand: dict) -> list[str]:
    """Where the builds differ most in a part, for a reader about to debug it:
    the item that differs most across builds beyond its reruns, the reference's
    n runs first, and how the first run of each build reads it."""
    across = kls[:n, n:].mean(axis=(0, 1))
    within = kls[:n, :n][~np.eye(n, dtype=bool)].max(axis=0)
    item = int(np.argmax(across - within))
    unit = "chunks" if part == "probes" else "prompts"
    head = (
        f"Where: {part}, {int((across > within).sum())} of {len(across)} {unit} differ "
        f"more across builds than between reruns; most, {unit[:-1]} {item}"
    )
    if part == "probes":
        a, b = ((run["probe_ids"][item], run["probe_lp"][item]) for run in (ref, cand))
        t = int(np.argmax(bucketed_kl(*a, *b)))
        tops = [
            ", ".join(f"{i} ({math.exp(x):.3f})" for i, x in zip(ids[t, :3], lp[t, :3]))
            for ids, lp in (a, b)
        ]
        return [
            f"{head}, at position {t} (token ids, probabilities)",
            f"  reference: {tops[0]}",
            f"  candidate: {tops[1]}",
        ]
    a, b = (str(run[f"{part}_text"][item]) for run in (ref, cand))
    same = len(os.path.commonprefix([a, b]))
    start = max(0, same - 40)
    return [
        f"{head}, whose first runs' outputs agree for {same} characters",
        f"  reference: {'…' * (start > 0)}{a[start : same + 60]!r}",
        f"  candidate: {'…' * (start > 0)}{b[start : same + 60]!r}",
    ]


def render(r: dict) -> str:
    def times(x: float) -> str:
        return f"{x:,.2f}×" if math.isfinite(x) else "∞×"

    stat, (n, m) = r["stats"], r["runs"]
    got, needed = times(r["score"]), times(r["needed"])
    parts = ", ".join(f"{part} {times(x)}" for part, x in r["ratios"].items())
    odds = f"identical builds read this way 1 in {r['splits']:,} times"
    if r["verdict"] == "same":
        verdict = (
            f"Verdict: same within noise, {got} is not above the {needed} needed "
            f"(p = {r['p']:.4f} over {r['splits']:,} splits)"
        )
    elif r["verdict"] == "changed":
        verdict = f"Verdict: CHANGED, {got} is above the {needed} needed; {odds}"
    else:
        refs, cands = next((s for s in STAGES if s[1] > m), STAGES[-1])
        verdict = (
            f"Undecided: the builds differ, {got} against the {needed} needed, "
            f"but {odds}; the next stage takes {max(n, refs)} reference and {cands} "
            "candidate runs"
        )
    first = (
        f"First runs: probe KL {stat['kl']:.6f}, "
        f"NLL {100 * stat['nll_change']:+.3f} %, "
        f"same top token {100 * stat['same_top']:.2f} %, "
        f"{100 * stat['outputs_differ']:.1f} % of outputs differ"
    )
    noise = f"Across builds over rerun noise, {n} reference and {m} candidate runs:"
    where = r["where"] if r["verdict"] != "same" else []
    return "\n".join([first, f"{noise} {parts}", verdict, *where])


def jsonable(x):
    """The value with infinite floats as strings, which JSON lacks."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    return str(x) if isinstance(x, float) and not math.isfinite(x) else x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="score one run against a running server")
    rec.add_argument("--host", default="http://127.0.0.1")
    rec.add_argument("--port", type=int, default=8000)
    rec.add_argument("--out", type=Path, required=True, help=".npz path")
    rec.add_argument("--seed", type=int, help="request order; random by default")
    rec.add_argument(
        "--setup",
        default="{}",
        help="json that must match between compared runs, e.g. the server's arguments",
    )
    cmp = sub.add_parser("compare", help="compare recorded runs")
    cmp.add_argument("--reference", type=Path, nargs="+", required=True)
    cmp.add_argument("--candidate", type=Path, nargs="+", required=True)
    cmp.add_argument("--save-results", type=Path, help="Save results to JSON file")
    args = parser.parse_args()
    if args.command == "record":
        seed = secrets.randbelow(2**31) if args.seed is None else args.seed
        url = f"{args.host}:{args.port}"
        save(asyncio.run(record(url, seed, json.loads(args.setup))), args.out)
        return
    refs = [load(p) for p in args.reference]
    cands = [load(p) for p in args.candidate]
    try:
        result = compare(refs, cands)
    except ValueError as e:  # 1 is the exit code of a change
        print(e, file=sys.stderr)
        raise SystemExit(2) from None
    print(render(result))
    if args.save_results:
        args.save_results.write_text(json.dumps(jsonable(result), indent=2) + "\n")
    raise SystemExit({"same": 0, "changed": 1, "more": 3}[result["verdict"]])


if __name__ == "__main__":
    main()
