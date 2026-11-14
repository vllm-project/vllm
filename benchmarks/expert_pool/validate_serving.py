# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-token multi-needle retrieval and concurrent HTTP decode measurements."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    marker = "ARCHIVE_CONTENT_SENTINEL"
    keys = ["amber", "birch", "cedar", "dune", "elm"]
    message = (
        "Read the archive. Later records do not override critical records.\n"
        + marker
        + "\nReturn ONLY a JSON object mapping amber, birch, cedar, "
        "dune, and elm to the exact codes in their critical records. "
        "No explanation or Markdown."
    )
    template = tok.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    marker_ids = tok.encode(marker, add_special_tokens=False)
    starts = [
        i
        for i in range(len(template) - len(marker_ids) + 1)
        if template[i : i + len(marker_ids)] == marker_ids
    ]
    assert len(starts) == 1, "Sentinel must occupy one exact token sequence"
    prefix = template[: starts[0]]
    suffix = template[starts[0] + len(marker_ids) :]
    rng = random.Random(56177)
    words = [
        "shipment",
        "archive",
        "ordinary",
        "calendar",
        "warehouse",
        "invoice",
        "package",
        "address",
        "pencil",
        "report",
        "desk",
        "paper",
        "amberwood",
        "cotton",
        "truck",
        "road",
        "inventory",
        "receipt",
        "cargo",
        "office",
    ]
    corpus = []
    if args.corpus:
        paths = sorted(
            p for p in Path(args.corpus).rglob("*.py") if "third_party" not in p.parts
        )
        rng.shuffle(paths)
        sections = []
        size = 0
        for path in paths:
            data = path.read_bytes()
            sections.append(
                f"\n=== Archived source: {path.relative_to(args.corpus)} ===\n"
                + data.decode()
            )
            corpus.append(
                {
                    "path": str(path.relative_to(args.corpus)),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            size += len(data)
            if size >= 8_000_000:
                break
        text = "\n".join(sections)
    else:
        text = "\n".join(
            f"Record {i}: "
            + " ".join(rng.choices(words, k=10))
            + f", quantity {rng.randrange(100, 999)}."
            for i in range(100000)
        )
    filler = tok.encode(text, add_special_tokens=False)
    tokenizer_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    }
    manifest = {
        "tokenizer": args.tokenizer,
        "cases": [],
        "corpus": corpus,
        "tokenizer_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(args.tokenizer).iterdir()
            if p.is_file() and p.name in tokenizer_files
        },
    }
    for length in args.lengths:
        for repeat in range(args.repeats):
            seed = 56177 + length + repeat
            prng = random.Random(seed)
            expected = {
                key: "".join(prng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=12))
                for key in keys
            }
            facts = [
                tok.encode(
                    "\nCRITICAL RECORD: the exact code for "
                    f"{key} is {expected[key]}.\n",
                    add_special_tokens=False,
                )
                for key in keys
            ]
            remaining = length - len(prefix) - len(suffix) - sum(map(len, facts))
            assert 0 < remaining < len(filler)
            offset = repeat * 777
            haystack = filler[offset : offset + remaining]
            assert len(haystack) == remaining
            archive = []
            cursor = 0
            locations = []
            for fraction, fact in zip((0.001, 0.25, 0.5, 0.75, 0.99), facts):
                at = int(remaining * fraction)
                archive += haystack[cursor:at]
                locations.append(len(prefix) + len(archive))
                archive += fact
                cursor = at
            archive += haystack[cursor:]
            prompt = prefix + archive + suffix
            assert len(prompt) == length
            case = {
                "length": length,
                "repeat": repeat,
                "seed": seed,
                "expected": expected,
                "needle_token_positions": locations,
                "prompt_sha256": digest(prompt),
                "prompt_token_ids": prompt,
            }
            name = f"retrieval-{length}-{repeat}.json"
            (out / name).write_text(json.dumps(case))
            manifest["cases"].append(
                {k: v for k, v in case.items() if k != "prompt_token_ids"}
                | {"file": name}
            )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def request(base, model, prompt, output_tokens, exact_output=False):
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if exact_output:
        body.update(min_tokens=output_tokens, ignore_eos=True)
    result = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "prompt_tokens_expected": len(prompt),
        "prompt_sha256": digest(prompt),
        "output_tokens_requested": output_tokens,
        "exact_output": exact_output,
        "content": "",
        "usage": None,
        "finish_reason": None,
        "done": False,
        "first_token_s": None,
        "last_token_s": None,
        "passed": False,
    }
    start = time.perf_counter()
    result["started_monotonic"] = start
    try:
        http = urllib.request.Request(
            base.rstrip("/") + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http, timeout=3600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    result["done"] = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("text"):
                        elapsed = time.perf_counter() - start
                        if result["first_token_s"] is None:
                            result["first_token_s"] = elapsed
                        result["last_token_s"] = elapsed
                        result["content"] += choice["text"]
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
        assert result["done"] and result["content"], "Incomplete SSE"
        usage = result["usage"]
        assert usage and usage["prompt_tokens"] == len(prompt), (
            "Wrong input token count"
        )
        assert result["finish_reason"] in ("stop", "length"), "Unexpected finish"
        if exact_output:
            assert usage["completion_tokens"] == output_tokens
        duration = result["last_token_s"] - result["first_token_s"]
        result["decode_tok_s"] = (
            (usage["completion_tokens"] - 1) / duration if duration > 0 else None
        )
        result["passed"] = True
    except Exception as exc:
        result["error"] = repr(exc)
        if isinstance(exc, urllib.error.HTTPError):
            result["error_body"] = exc.read().decode(errors="replace")
    result["elapsed_s"] = time.perf_counter() - start
    result["finished_monotonic"] = time.perf_counter()
    result["output_sha256"] = hashlib.sha256(result["content"].encode()).hexdigest()
    return result


def retrieve(args, path):
    case = json.loads(path.read_text())
    result = request(args.base_url, args.model, case["prompt_token_ids"], 256)
    result.update(
        case_file=path.name,
        expected=case["expected"],
        needle_token_positions=case["needle_token_positions"],
    )
    try:
        parsed = json.loads(result["content"].strip())
        result["retrieved"] = parsed
        result["correct_needles"] = sum(
            parsed.get(k) == v for k, v in case["expected"].items()
        )
        assert result["correct_needles"] == 5 and parsed == case["expected"], (
            "Incorrect retrieval"
        )
        assert result["finish_reason"] == "stop", "Retrieval output truncated"
    except Exception as exc:
        result.update(passed=False, validation_error=repr(exc))
    return result


def run(args):
    destination = Path(args.output)
    destination.mkdir(exist_ok=False)
    if args.kind == "plan":
        phases = []
        for kind in ("decode", "retrieval"):
            child = argparse.Namespace(**vars(args))
            child.kind = kind
            child.output = str(destination / kind)
            try:
                run(child)
                phases.append({"kind": kind, "passed": True})
            except SystemExit as exc:
                phases.append({"kind": kind, "passed": False, "exit_code": exc.code})
        (destination / "summary.json").write_text(
            json.dumps(
                {"phases": phases, "passed": all(p["passed"] for p in phases)}, indent=2
            )
            + "\n"
        )
        if not all(p["passed"] for p in phases):
            raise SystemExit(1)
        return
    results = []
    if args.kind in ("retrieval", "retrieval-concurrent"):
        manifest = json.loads((Path(args.prompts) / "manifest.json").read_text())
        selected = [
            row
            for row in manifest["cases"]
            if (not args.lengths or row["length"] in args.lengths)
            and row["repeat"] < args.repeats
        ]
        workers = max(args.concurrency) if args.kind == "retrieval-concurrent" else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(retrieve, args, Path(args.prompts) / row["file"]): row
                for row in selected
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                result = future.result()
                results.append(result)
                (destination / row["file"]).write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("content", "expected", "retrieved")
                        }
                    ),
                    flush=True,
                )
    else:
        pair = json.loads(Path(args.pair).read_text())
        prompts = [t["prompt_token_ids"] for t in pair["tasks"]]
        warmup = request(
            args.base_url, args.model, prompts[0], args.warmup_tokens, True
        )
        (destination / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
        assert warmup["passed"], "Warmup failed"
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = [
                        executor.submit(
                            request,
                            args.base_url,
                            args.model,
                            prompts[(i + repeat) % len(prompts)],
                            args.output_tokens,
                            True,
                        )
                        for i in range(concurrency)
                    ]
                    batch = [future.result() for future in futures]
                first = min(r["started_monotonic"] for r in batch)
                last = max(r["finished_monotonic"] for r in batch)
                item = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "requests": batch,
                    "aggregate_e2e_tok_s": sum(
                        (r.get("usage") or {}).get("completion_tokens", 0)
                        for r in batch
                    )
                    / (last - first),
                    "passed": all(r["passed"] for r in batch),
                }
                if item["passed"]:
                    first_text = min(
                        r["started_monotonic"] + r["first_token_s"] for r in batch
                    )
                    last_text = max(
                        r["started_monotonic"] + r["last_token_s"] for r in batch
                    )
                    item["aggregate_decode_window_tok_s"] = sum(
                        r["usage"]["completion_tokens"] - 1 for r in batch
                    ) / (last_text - first_text)
                results.append(item)
                (destination / f"c{concurrency}-{repeat}.json").write_text(
                    json.dumps(item, indent=2) + "\n"
                )
                print(
                    json.dumps({k: v for k, v in item.items() if k != "requests"}),
                    flush=True,
                )
    summary = {
        "kind": args.kind,
        "passed": bool(results) and all(r["passed"] for r in results),
        "results": results,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[12500, 81600, 128000, 262000, 524000, 1000000],
    )
    prep.add_argument("--repeats", type=int, default=3)
    prep.add_argument("--corpus")
    runner = sub.add_parser("run")
    runner.add_argument(
        "--kind",
        choices=["retrieval", "retrieval-concurrent", "decode", "plan"],
        required=True,
    )
    runner.add_argument("--base-url", default="http://127.0.0.1:18098")
    runner.add_argument("--model", default="flashnext")
    runner.add_argument("--output", required=True)
    runner.add_argument("--prompts")
    runner.add_argument("--lengths", nargs="+", type=int)
    runner.add_argument("--pair")
    runner.add_argument("--concurrency", nargs="+", type=int, default=[1, 2])
    runner.add_argument("--repeats", type=int, default=3)
    runner.add_argument("--output-tokens", type=int, default=256)
    runner.add_argument("--warmup-tokens", type=int, default=128)
    a = p.parse_args()
    prepare(a) if a.command == "prepare" else run(a)


if __name__ == "__main__":
    main()
