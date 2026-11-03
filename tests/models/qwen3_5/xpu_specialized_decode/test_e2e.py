# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End to end on two Xe2 GPUs: greedy outputs of the mono decode
(``--monokernel``) against the stock decode of the same eager model, with
unaligned prompt lengths (different per request), MTP speculative decoding
(verify kernels, the drafter on the kernels, candidate rejection sampling),
steps that run vLLM's layers (more than 8 rows, a verify width without a
kernel), candidate sampling, the KV cache bound twice (memory profiling, then
the real cache) and fail-stop (a stalled rank stops the engine).

Needs two Xe2 GPUs and a Qwen3.6-35B-A3B checkpoint in
VLLM_XPU_MONO_TEST_MODEL; skipped otherwise. Every run is its own process.

The prompts are vLLM's docs cut at the given lengths. Greedy outputs may
differ at near-ties (the kernels accumulate in another order): a divergence is
accepted only where the stock decode's two best tokens are within NEAR_TIE
nats.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import regex as re

MODEL = os.environ.get("VLLM_XPU_MONO_TEST_MODEL", "")
NEAR_TIE = 0.5
PROMPT_LENGTHS = [1, 63, 64, 65, 1000, 4097, 9001]


def _two_xe2_gpus() -> bool:
    try:
        import torch
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    ops = torch.ops._xpu_C
    return (
        torch.xpu.is_available()
        and torch.xpu.device_count() >= 2
        and hasattr(ops, "mono_decode_supported")
        and ops.mono_decode_supported(0)
        and ops.mono_decode_supported(1)
    )


pytestmark = pytest.mark.skipif(
    not (MODEL and os.path.isdir(MODEL) and _two_xe2_gpus()),
    reason="needs two Xe2 GPUs and VLLM_XPU_MONO_TEST_MODEL",
)


def _run(mono: bool, case: dict, env: dict | None = None, check: bool = True):
    out = tempfile.mktemp(suffix=".json")
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="1", **(env or {}))
    p = subprocess.run(
        [sys.executable, __file__, json.dumps(dict(case, mono=mono)), out],
        env=env,
        timeout=1800,
        capture_output=True,
        text=True,
    )
    log = p.stdout + p.stderr
    if not check:
        return p.returncode, log
    assert p.returncode == 0, log[-4000:]
    with open(out) as f:
        res = json.load(f)
    m = re.search(r"Qwen3.5 XPU mono decode: (\d+) decoder layers", log)
    res["mono_layers"] = int(m.group(1)) if m else 0
    res["log"] = log
    return res


def _greedy_matches(stock: dict, mono: dict) -> None:
    for i, (s, m) in enumerate(zip(stock["rows"], mono["rows"])):
        ids_s, ids_m = s["ids"], m["ids"]
        assert len(ids_m) == len(ids_s)
        for t, (a, b) in enumerate(zip(ids_s, ids_m)):
            if a == b:
                continue
            top = s["top"][t]
            gap = top[0][1] - top[1][1] if len(top) > 1 else float("inf")
            assert gap < NEAR_TIE, (
                f"row {i} token {t}: mono {b} vs stock {a}, stock margin {gap:.3f} nats"
            )
            break  # the continuations differ after a near-tie


CASES: dict[str, dict] = {
    # 7 requests of unaligned lengths: decode steps of 8 (padded) rows
    "decode_unaligned": dict(lengths=PROMPT_LENGTHS),
    "decode_single": dict(lengths=[4097]),
    # 12 requests: decode steps wider than 8 rows run vLLM's layers
    "decode_wide": dict(lengths=[64 + 37 * i for i in range(12)]),
    # MTP: 1 x 2, 2 x 4, 1 x 8 verify rows; 4 x 4 = 16 rows (vLLM's layers);
    # k=5: 1 x 6 rows (no verify kernel at 6 tokens a request)
    "mtp_k1": dict(lengths=[65], spec=1),
    "mtp_k3_b2": dict(lengths=[63, 1000], spec=3),
    "mtp_k7": dict(lengths=[1000], spec=7),
    "mtp_k3_b4": dict(lengths=[1, 64, 65, 4097], spec=3),
    "mtp_k5": dict(lengths=[1000], spec=5),
}


@pytest.mark.parametrize("name", list(CASES))
def test_greedy_mono_vs_stock(name):
    case: dict = CASES[name]
    stock = _run(False, {**case, "logprobs": 5})
    mono = _run(True, {**case, "logprobs": 0})  # candidate sampling
    assert mono["mono_layers"] == 40, mono["log"][-2000:]
    _greedy_matches(stock, mono)


def test_caches_bound_twice():
    """Without --kv-cache-memory-bytes the memory profiling binds a temporary
    KV cache first: the executor binds every cache vLLM allocates."""
    mono = _run(True, dict(lengths=[1000]))
    assert mono["log"].count("Qwen3.5 XPU mono decode: caches bound") >= 2


def test_candidate_sampling_runs():
    """top_k / top_p sampling on the candidates (small batch, MTP)."""
    mono = _run(
        True, dict(lengths=[64, 1000], spec=1, temperature=1.0, top_k=20, top_p=0.95)
    )
    assert mono["mono_layers"] == 40
    for r in mono["rows"]:
        assert len(r["ids"]) == 32 and all(0 <= t < 248320 for t in r["ids"])


# Shifts rank 1's mailbox tags of layer 0 at its 40th step (after warm-up and
# capture): the ranks never see each other's words again, every wait gives up
# and the engine must stop. Patched into ``FailStop.after_step`` as the guards
# module loads, in every vLLM process.
_SITECUSTOMIZE = textwrap.dedent(
    """\
    import importlib.abc
    import importlib.machinery
    import sys

    _MODULE = "vllm.models.qwen3_5.xpu.specialized_decode.runtime.guards"


    def _patch(module):
        after_step = module.FailStop.after_step

        def patched(self):
            if self.rank == 1 and self.steps == 40:
                import torch

                torch.ops._xpu_C.mono_fault_inject(0, 7)
            after_step(self)

        module.FailStop.after_step = patched


    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            exec_module = spec.loader.exec_module

            def patched_exec(module):
                exec_module(module)
                _patch(module)

            spec.loader.exec_module = patched_exec
            return spec


    sys.meta_path.insert(0, _Finder())
    """
)


def test_fail_stop_stops_the_engine():
    site = tempfile.mkdtemp()
    with open(os.path.join(site, "sitecustomize.py"), "w") as f:
        f.write(_SITECUSTOMIZE)
    path = os.pathsep.join(p for p in (site, os.environ.get("PYTHONPATH")) if p)
    rc, log = _run(
        True,
        dict(lengths=[1000], max_tokens=512),
        env={"PYTHONPATH": path},
        check=False,
    )
    assert rc != 0, log[-2000:]
    assert "gave up an in-kernel wait" in log, log[-4000:]


def _text_ids(tokenizer, n: int) -> list[int]:
    """N token ids of natural text (vLLM's docs): greedy decoding of random
    token ids is out of distribution and flips at accumulation-order noise."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[4] / "docs"
    ids: list[int] = []
    for path in sorted(docs.rglob("*.md")):
        text = path.read_text(errors="ignore")
        ids += tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= n:
            return ids[:n]
    raise RuntimeError(f"{docs}: fewer than {n} tokens of text")


def _main(case: dict, out: str) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="float16",
        block_size=64,
        quantization="fp8_per_tensor",
        max_model_len=16384,
        max_num_seqs=16,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        monokernel=case["mono"],
    )
    if case.get("spec"):
        kw["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": case["spec"],
        }
    llm = LLM(**kw)
    starts = [997 * i for i in range(len(case["lengths"]))]
    ids = _text_ids(
        llm.get_tokenizer(), max(s + n for s, n in zip(starts, case["lengths"]))
    )
    prompts = [
        TokensPrompt(prompt_token_ids=ids[s : s + n])
        for s, n in zip(starts, case["lengths"])
    ]
    sp = SamplingParams(
        max_tokens=case.get("max_tokens", 32),
        ignore_eos=True,
        temperature=case.get("temperature", 0.0),
        top_k=case.get("top_k", -1),
        top_p=case.get("top_p", 1.0),
        logprobs=case.get("logprobs") or None,
    )
    rows = []
    for o in llm.generate(prompts, sp):
        c = o.outputs[0]
        rows.append(
            {
                "ids": list(c.token_ids),
                "top": [
                    sorted(((t, v.logprob) for t, v in lp.items()), key=lambda x: -x[1])
                    for lp in c.logprobs
                ]
                if c.logprobs
                else None,
            }
        )
    with open(out, "w") as f:
        json.dump({"rows": rows}, f)


if __name__ == "__main__":
    _main(json.loads(sys.argv[1]), sys.argv[2])
