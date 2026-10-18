# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark speculative decoding with a reference-replay oracle.

The oracle replays saved target-model outputs as draft tokens. Comparing it
with ngram and EAGLE3 provides a performance reference for the tested setup.
Replay has nonzero CPU overhead, and proposals may be rejected, so the result
is not a theoretical upper bound.

Typical use (one configuration per process; see benchmarks/README.md):

    # 1. baseline, also writes the reference outputs the oracle replays
    python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B \
        --mode baseline --prompts prompts.json --max-num-seqs 32 \
        --no-async-scheduling --write-ref ref.json --out base.json
    # 2. reference-replay oracle at k=4
    python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B \
        --mode oracle --k 4 --prompts prompts.json --max-num-seqs 32 \
        --ref ref.json --out oracle_k4.json
    # 3. a real drafter for comparison
    python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B \
        --mode eagle3 --k 3 --draft-model RedHatAI/Qwen3-8B-speculator.eagle3 \
        --prompts prompts.json --max-num-seqs 32 --no-async-scheduling \
        --ref ref.json --out e3.json

Use ``--no-async-scheduling`` on the baseline and EAGLE3 to match the scheduling
mode used by CPU-side drafters (``custom_class``, ``ngram``).

``--prompts`` is a JSON list of user-turn strings; the chat template is
applied with thinking disabled for Qwen3 models.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

# The oracle is loaded by vLLM's custom_class proposer via this module path,
# so the benchmarks directory must be importable (it is when the script is
# run from the repository root or by path).
ORACLE_CLASS = "benchmark_spec_decode_oracle.OracleProposer"


class OracleProposer:
    """Drafter that replays a reference output for each request.

    The reference file (``ORACLE_REF_FILE``) is a JSON list of
    ``{"prompt_token_ids": [...], "output_token_ids": [...]}`` written by a
    ``--mode baseline`` run. Requests are identified by the bytes of their
    first ``L`` prompt tokens, where ``L`` is the shortest prompt length.

    While generation matches the reference, up to ``k`` reference tokens
    are proposed. If the output diverges, the proposer attempts to realign
    using the last ``ORACLE_RESYNC_NGRAM`` (3) generated tokens in the reference.
    ``ORACLE_STRICT=1`` disables realignment, so divergent requests receive
    no drafts. Cumulative proposer statistics are written to
    ``ORACLE_STATS_DIR/oracle_stats_<pid>.json`` every 100 calls. These snapshots
    include warm-up, may omit the final calls, and can be absent for short runs.
    """

    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.k = int(spec.num_speculative_tokens)
        self.max_model_len = int(vllm_config.model_config.max_model_len)
        self.strict = os.environ.get("ORACLE_STRICT", "0") == "1"
        self.m = int(os.environ.get("ORACLE_RESYNC_NGRAM", "3"))
        with open(os.environ["ORACLE_REF_FILE"]) as f:
            refs = json.load(f)
        self.key_len = min(len(r["prompt_token_ids"]) for r in refs)
        self.table: dict[bytes, tuple[int, np.ndarray]] = {}
        for r in refs:
            prompt = np.asarray(r["prompt_token_ids"], dtype=np.int32)
            key = prompt[: self.key_len].tobytes()
            if key in self.table:
                raise ValueError(
                    f"two prompts share their first {self.key_len} tokens; "
                    "the oracle cannot tell them apart"
                )
            out = np.asarray(r["output_token_ids"], dtype=np.int32)
            self.table[key] = (len(prompt), out)
        self.counters = {
            "calls": 0,
            "rows": 0,
            "lookup_miss": 0,
            "on_ref": 0,
            "diverged": 0,
            "resynced": 0,
            "no_resync": 0,
            "past_end": 0,
            "drafted_tokens": 0,
        }
        self.time_s = 0.0
        stats_dir = os.environ.get("ORACLE_STATS_DIR", "/tmp")
        self.stats_path = os.path.join(stats_dir, f"oracle_stats_{os.getpid()}.json")

    def _flush_stats(self) -> None:
        stats: dict[str, object] = dict(self.counters)
        stats.update(
            pid=os.getpid(),
            time_s=round(self.time_s, 4),
            ms_per_call=round(1000 * self.time_s / max(1, self.counters["calls"]), 4),
            key_len=self.key_len,
            strict=self.strict,
            resync_ngram=self.m,
        )
        try:
            with open(self.stats_path, "w") as f:
                json.dump(stats, f)
        except OSError:
            pass

    def _resync(self, gen: np.ndarray, ref: np.ndarray) -> int | None:
        """Return the index in ``ref`` after the last match of ``gen[-m:]``."""
        m = self.m
        if len(gen) < m or len(ref) <= m:
            return None
        gram = gen[-m:]
        n = len(ref) - m + 1
        hits = np.ones(n, dtype=bool)
        for j in range(m):
            hits &= ref[j : j + n] == gram[j]
        idx = np.flatnonzero(hits)
        if idx.size == 0:
            return None
        end = int(idx[-1]) + m
        return end if end < len(ref) else None

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings=None,
    ) -> list[list[int]]:
        """Propose up to ``k`` reference tokens for eligible requests."""
        t0 = time.perf_counter()
        c = self.counters
        c["calls"] += 1
        k, key_len = self.k, self.key_len
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            c["rows"] += 1
            n = int(num_tokens_no_spec[i])
            if n >= self.max_model_len or n < key_len:
                drafts.append([])
                continue
            row = token_ids_cpu[i]
            key = row[:key_len].astype(np.int32, copy=False).tobytes()
            hit = self.table.get(key)
            if hit is None:
                c["lookup_miss"] += 1
                drafts.append([])
                continue
            prompt_len, ref = hit
            pos = n - prompt_len
            if pos < 0 or pos >= len(ref):
                c["past_end"] += 1
                drafts.append([])
                continue
            gen = row[prompt_len:n]
            if np.array_equal(gen, ref[:pos]):
                c["on_ref"] += 1
                start = pos
            else:
                c["diverged"] += 1
                resynced = None if self.strict else self._resync(gen, ref)
                if resynced is None:
                    c["no_resync"] += 1
                    drafts.append([])
                    continue
                c["resynced"] += 1
                start = resynced
            draft = ref[start : start + k].tolist()
            c["drafted_tokens"] += len(draft)
            drafts.append(draft)
        self.time_s += time.perf_counter() - t0
        if c["calls"] % 100 == 0:
            self._flush_stats()
        return drafts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument(
        "--mode", required=True, choices=["baseline", "oracle", "ngram", "eagle3"]
    )
    p.add_argument("--k", type=int, default=0, help="num_speculative_tokens")
    p.add_argument("--draft-model", default=None, help="EAGLE3 draft head")
    p.add_argument("--prompts", required=True, help="JSON list of prompt strings")
    p.add_argument("--num-prompts", type=int, default=0, help="0 = all")
    p.add_argument("--ref", default=None, help="reference outputs from a baseline run")
    p.add_argument("--write-ref", default=None, help="baseline: write reference here")
    p.add_argument("--strict", action="store_true", help="oracle: stop on divergence")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument(
        "--serial", action="store_true", help="send prompts one at a time (bs=1)"
    )
    p.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="force async_scheduling=False; use on the baseline the oracle is "
        "compared against, since vLLM disables it for CPU-side drafters",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--out", required=True, help="result JSON path")
    p.add_argument("--tag", default="", help="free-form label stored in the result")
    return p.parse_args()


def build_prompts(model: str, prompts_path: str, num_prompts: int) -> list[list[int]]:
    from transformers import AutoTokenizer

    with open(prompts_path) as f:
        texts = json.load(f)
    if num_prompts:
        texts = texts[:num_prompts]
    tok = AutoTokenizer.from_pretrained(model)
    kwargs = {"enable_thinking": False} if "qwen3" in model.lower() else {}
    prompt_ids = []
    for text in texts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            **kwargs,
        )
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        prompt_ids.append([int(t) for t in ids])
    return prompt_ids


def speculative_config(args: argparse.Namespace) -> dict | None:
    if args.mode == "baseline":
        return None
    if args.mode == "oracle":
        if not args.ref:
            raise SystemExit("--ref is required for --mode oracle")
        os.environ["ORACLE_REF_FILE"] = args.ref
        os.environ["ORACLE_STRICT"] = "1" if args.strict else "0"
        stats_dir = f"/tmp/oracle_stats_{os.getpid()}"
        os.makedirs(stats_dir, exist_ok=True)
        os.environ["ORACLE_STATS_DIR"] = stats_dir
        return {
            "method": "custom_class",
            "model": ORACLE_CLASS,
            "num_speculative_tokens": args.k,
        }
    if args.mode == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": args.k,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
        }
    if not args.draft_model:
        raise SystemExit("--draft-model is required for --mode eagle3")
    return {
        "method": "eagle3",
        "model": args.draft_model,
        "num_speculative_tokens": args.k,
    }


def spec_decode_metrics(llm) -> dict:
    """Collect the spec-decode counters exposed by ``LLM.get_metrics()``."""
    out: dict = {}
    for m in llm.get_metrics():
        name = getattr(m, "name", "")
        if "spec_decode" not in name:
            continue
        if hasattr(m, "value"):
            out[name] = out.get(name, 0) + m.value
        elif hasattr(m, "values"):
            out[name] = list(m.values)
    return out


def subtract(after: dict, before: dict) -> dict:
    delta: dict = {}
    for name, v in after.items():
        b = before.get(name)
        if isinstance(v, list):
            if isinstance(b, list) and len(b) == len(v):
                delta[name] = [x - y for x, y in zip(v, b)]
            else:
                delta[name] = v
        elif isinstance(v, (int, float)):
            delta[name] = v - (b if isinstance(b, (int, float)) else 0)
        else:
            delta[name] = v
    return delta


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    prompt_ids = build_prompts(args.model, args.prompts, args.num_prompts)
    spec = speculative_config(args)

    t_init = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=spec,
        disable_log_stats=False,
        seed=0,
        # None lets vLLM choose the scheduling mode for this configuration.
        async_scheduling=False if args.no_async_scheduling else None,
    )
    init_s = time.time() - t_init

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]

    # Warm-up is not timed and its spec-decode counters are subtracted.
    llm.generate(inputs[:2], sampling, use_tqdm=False)
    before = spec_decode_metrics(llm)

    per_request_latency: list[float] = []
    t0 = time.time()
    if args.serial:
        outputs = []
        for inp in inputs:
            t = time.time()
            outputs.append(llm.generate([inp], sampling, use_tqdm=False)[0])
            per_request_latency.append(time.time() - t)
    else:
        outputs = llm.generate(inputs, sampling, use_tqdm=False)
    wall = time.time() - t0

    out_ids = [list(o.outputs[0].token_ids) for o in outputs]
    n_out = sum(len(x) for x in out_ids)
    metrics = subtract(spec_decode_metrics(llm), before)

    result = {
        "tag": args.tag,
        "mode": args.mode,
        "k": args.k,
        "model": args.model,
        "draft_model": args.draft_model,
        "strict_oracle": bool(args.strict) if args.mode == "oracle" else None,
        "async_scheduling_forced_off": args.no_async_scheduling,
        "num_prompts": len(inputs),
        "serial": args.serial,
        "max_num_seqs": args.max_num_seqs,
        "max_tokens": args.max_tokens,
        "init_s": round(init_s, 1),
        "wall_s": round(wall, 3),
        "output_tokens": n_out,
        "tokens_per_s": round(n_out / wall, 2),
        "mean_output_len": round(n_out / max(1, len(inputs)), 1),
        "spec_metrics": metrics,
    }
    if per_request_latency:
        lat = sorted(per_request_latency)
        result["req_latency_mean_s"] = round(sum(lat) / len(lat), 3)
        result["req_latency_p50_s"] = round(lat[len(lat) // 2], 3)
        # Generation wall time per output token, including prefill.
        result["tpot_ms"] = round(1000 * wall / max(1, n_out), 2)

    drafts = metrics.get("vllm:spec_decode_num_drafts", 0)
    draft_tokens = metrics.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens", 0)
    if drafts:
        # Mean accepted draft tokens per draft event, plus the bonus token.
        result["mean_acceptance_len"] = round(1 + accepted / drafts, 3)
        result["draft_acceptance_rate"] = round(accepted / max(1, draft_tokens), 4)

    if args.mode == "oracle":
        stats = []
        for path in glob.glob(os.path.join(os.environ["ORACLE_STATS_DIR"], "*.json")):
            with open(path) as f:
                stats.append(json.load(f))
        result["oracle_proposer_stats"] = stats

    if args.ref:
        with open(args.ref) as f:
            refs = json.load(f)
        ref_map = {tuple(r["prompt_token_ids"]): r["output_token_ids"] for r in refs}
        same = sum(1 for p, o in zip(prompt_ids, out_ids) if ref_map.get(tuple(p)) == o)
        result["outputs_identical_to_ref"] = same
        result["outputs_identical_frac"] = round(same / len(out_ids), 3)

    if args.write_ref:
        with open(args.write_ref, "w") as f:
            json.dump(
                [
                    {"prompt_token_ids": p, "output_token_ids": o}
                    for p, o in zip(prompt_ids, out_ids)
                ],
                f,
            )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
