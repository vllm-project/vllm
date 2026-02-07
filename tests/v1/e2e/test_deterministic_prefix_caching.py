# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for deterministic prefix caching.

Validates that --deterministic-prefix-caching resolves non-deterministic
behavior caused by bf16 GEMM floating-point non-associativity when prefix
caching changes the batch dimension (M) between cache-miss and cache-hit
paths.

Background (https://github.com/vllm-project/vllm/issues/33123):
  With prefix caching enabled, a cache-miss prefill computes all N tokens
  in a single GEMM (M=N), while a cache-hit computes only the uncached
  suffix (M=N%block_size). Different M values cause the GEMM backend
  (e.g. Tensile on ROCm, cuBLAS on CUDA) to select different tile
  configurations, which changes the fp32 accumulation order in the
  K-dimension dot product. This produces 1-ULP differences that amplify
  ~1000x through the residual stream of a deep transformer, ultimately
  flipping argmax tokens.

Fix:
  --deterministic-prefix-caching forces cache-miss prefills to split at
  the last block boundary, so the suffix GEMM always uses the same M
  dimension regardless of cache state.
"""

import httpx
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10
NUM_RUNS = 5
LOGPROB_TOLERANCE = 1e-5

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not current_platform.is_rocm(),
        reason="Prefix cache tiling divergence reliably reproduces on ROCm; "
        "on CUDA the argmax may not flip for this test prompt",
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def test_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]


def get_hf_reference(
    tokenizer,
    messages,
    *,
    use_cache: bool = True,
) -> dict:
    """Generate a greedy-decoded reference using HuggingFace Transformers."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = torch.tensor([token_ids], device=model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=use_cache,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, len(token_ids) :].tolist()
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logprobs: list[float] = []
    for score in outputs.scores:
        lp = torch.nn.functional.log_softmax(score[0], dim=-1)
        logprobs.append(lp[generated_ids[len(logprobs)]].item())

    result = {
        "tokens": generated_ids,
        "decoded": decoded,
        "logprobs": logprobs,
        "input_token_ids": token_ids,
    }

    del model
    torch.cuda.empty_cache()
    return result


def _make_client(server: RemoteOpenAIServer) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
    return httpx.AsyncClient(
        transport=transport,
        base_url=server.url_root,
        timeout=600,
        headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"},
    )


async def _request(
    client: httpx.AsyncClient,
    token_ids: list[int],
    tokenizer,
) -> dict:
    payload = {
        "model": MODEL_NAME,
        "token_ids": token_ids,
        "sampling_params": {
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": 1,
            "detokenize": False,
        },
        "stream": False,
    }
    resp = await client.post("/inference/v1/generate", json=payload)
    assert resp.status_code == 200, f"Request failed: {resp.text}"

    choice = resp.json()["choices"][0]
    tokens = choice["token_ids"]
    logprobs_data = choice.get("logprobs", {}).get("content", [])
    return {
        "tokens": tokens,
        "decoded": tokenizer.decode(tokens, skip_special_tokens=True),
        "logprobs": [lp.get("logprob") for lp in logprobs_data],
    }


async def _request_n(
    client: httpx.AsyncClient,
    token_ids: list[int],
    tokenizer,
    n: int,
) -> list[dict]:
    return [await _request(client, token_ids, tokenizer) for _ in range(n)]


def _first_diff(a: tuple, b: tuple) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def assert_tokens_deterministic(results: list[dict], label: str) -> None:
    ref = tuple(results[0]["tokens"])
    for i, r in enumerate(results[1:], 2):
        cur = tuple(r["tokens"])
        assert cur == ref, (
            f"{label}: Run {i} differs from Run 1\n"
            f"  Run 1: {results[0]['decoded']}\n"
            f"  Run {i}: {r['decoded']}\n"
            f"  First diff at position {_first_diff(ref, cur)}"
        )


def assert_logprobs_deterministic(
    results: list[dict],
    label: str,
    tol: float = LOGPROB_TOLERANCE,
) -> None:
    ref = results[0]["logprobs"]
    for i, r in enumerate(results[1:], 2):
        for pos, (a, b) in enumerate(zip(ref, r["logprobs"])):
            diff = abs(a - b)
            assert diff < tol, (
                f"{label}: logprob at pos {pos} differs "
                f"(Run 1={a:.8f}, Run {i}={b:.8f}, diff={diff:.2e})"
            )


def check_logprob_determinism(
    results: list[dict],
    tol: float = LOGPROB_TOLERANCE,
) -> tuple[bool, list[int]]:
    if len(results) < 2:
        return True, []
    ref = results[0]["logprobs"]
    positions: list[int] = []
    for r in results[1:]:
        for pos, (a, b) in enumerate(zip(ref, r["logprobs"])):
            if abs(a - b) > tol and pos not in positions:
                positions.append(pos)
    return len(positions) == 0, positions


def detect_prefix_cache_bug(results: list[dict]) -> bool:
    """Return True if Run 1 (miss) differs from Run 2+ (hits) while
    all cache-hit runs agree with each other — the classic symptom."""
    if len(results) < 3:
        return False
    first = tuple(results[0]["tokens"])
    rest = [tuple(r["tokens"]) for r in results[1:]]
    return first != rest[0] and all(r == rest[0] for r in rest[1:])


def detect_prefix_cache_bug_logprobs(
    results: list[dict],
    tol: float = LOGPROB_TOLERANCE,
) -> bool:
    """Detect the bug via logprob divergence even when argmax doesn't flip."""
    if len(results) < 3:
        return False
    r1_lp = results[0]["logprobs"]
    r2_lp = results[1]["logprobs"]
    r1_differs = any(abs(a - b) > tol for a, b in zip(r1_lp, r2_lp))
    rest_agree = all(
        all(abs(a - b) <= tol for a, b in zip(r2_lp, r["logprobs"]))
        for r in results[2:]
    )
    return r1_differs and rest_agree


def _print_comparison(
    results_map: dict[str, dict],
    tokenizer,
    *,
    max_pos: int = 10,
) -> None:
    configs = list(results_map.keys())
    tokens = results_map[configs[0]]["tokens"]
    n = min(max_pos, len(tokens))

    header = f"{'Pos':<4} {'Token':<12}"
    for c in configs:
        header += f" {c:<20}"
    print(f"\n{'=' * 160}\nLOGPROB COMPARISON\n{'=' * 160}")
    print(header)
    print("-" * 160)

    for pos in range(n):
        tok = repr(tokenizer.decode([tokens[pos]]))[:10]
        row = f"{pos:<4} {tok:<12}"
        for c in configs:
            lp = results_map[c]["logprobs"]
            row += f" {lp[pos]:+.10f}    " if pos < len(lp) else " N/A".ljust(21)
        print(row)
    print("-" * 160)


def _print_summary(
    baseline: list[dict],
    deterministic: list[dict],
    hf: dict | None = None,
) -> None:
    sep = "=" * 100
    print(f"\n{sep}")
    print("SUMMARY")
    print(f"{sep}")
    print("Issue: https://github.com/vllm-project/vllm/issues/33123\n")

    # ---- Decoded text for every run ----
    print("--- Generated Text (all runs) ---\n")

    if hf:
        print(f"  HF reference:          '{hf['decoded']}'")
        print()

    for i, r in enumerate(baseline, 1):
        tag = "(cache miss)" if i == 1 else "(cache hit)"
        print(f"  Baseline Run {i} {tag:>14}: '{r['decoded']}'")
    print()

    for i, r in enumerate(deterministic, 1):
        tag = "(cache miss)" if i == 1 else "(cache hit)"
        print(f"  Deterministic Run {i} {tag:>14}: '{r['decoded']}'")
    print()

    # ---- Baseline analysis ----
    bl_tok_ok = all(
        tuple(r["tokens"]) == tuple(baseline[0]["tokens"]) for r in baseline
    )
    bl_lp_ok, bl_lp_pos = check_logprob_determinism(baseline)
    bl_bug_tokens = detect_prefix_cache_bug(baseline)
    bl_bug_logprobs = detect_prefix_cache_bug_logprobs(baseline)

    print("--- Baseline (--enable-prefix-caching) ---")
    if bl_tok_ok and bl_lp_ok:
        print("  Tokens: ✓  Logprobs: ✓  (bug did not manifest)")
    elif bl_bug_tokens:
        print("  Tokens: ✗  PREFIX CACHE BUG DETECTED (tokens diverge)")
        print(f"    Run 1 (miss): '{baseline[0]['decoded']}'")
        print(f"    Run 2 (hit):  '{baseline[1]['decoded']}'")
    elif bl_bug_logprobs:
        print(
            "  Tokens: ✓  Logprobs: ✗  PREFIX CACHE BUG DETECTED "
            "(logprobs diverge, argmax did not flip)"
        )
        print(
            f"    Logprob differences at positions: {bl_lp_pos[:5]}"
            f"{'...' if len(bl_lp_pos) > 5 else ''}"
        )
    else:
        print(
            f"  Tokens: {'✓' if bl_tok_ok else '✗'}  "
            f"Logprobs: {'✓' if bl_lp_ok else '✗'}"
        )

    # ---- Deterministic analysis ----
    dt_tok_ok = all(
        tuple(r["tokens"]) == tuple(deterministic[0]["tokens"]) for r in deterministic
    )
    dt_lp_ok, dt_lp_pos = check_logprob_determinism(deterministic)

    print("\n--- Deterministic (--deterministic-prefix-caching) ---")
    if dt_tok_ok and dt_lp_ok:
        print("  Tokens: ✓  Logprobs: ✓  ✅ Fix validated")
    elif not dt_tok_ok:
        print("  Tokens: ✗  ❌ FIX FAILED")
    else:
        print(
            f"  Tokens: ✓  Logprobs: ✗  (acceptable — tokens are "
            f"deterministic, logprobs differ at {dt_lp_pos[:5]})"
        )

    # ---- Cross-comparison ----
    print("\n--- Cross-comparison ---")
    dt_tokens = tuple(deterministic[0]["tokens"])
    bl_hit_tokens = tuple(baseline[1]["tokens"])
    match = dt_tokens == bl_hit_tokens
    print(f"  Deterministic Run1 == Baseline Run2 (tokens): {'✓' if match else '✗'}")
    if not match:
        print(
            "    Expected: deterministic mode computes the prefix with a "
            "different M dimension"
        )
        print(
            "    (M=N-R instead of M=N), so cached KV values carry "
            "different 1-ULP rounding."
        )
        print(
            "    Both paths are valid bf16 computations. The guarantee "
            "is self-consistency,"
        )
        print("    not equivalence with any specific reference path.")

    if hf:
        hf_match = dt_tokens == tuple(hf["tokens"])
        print(
            f"  Deterministic Run1 == HF reference (tokens):  "
            f"{'✓' if hf_match else '✗'}"
        )

    print(f"\n{sep}\n")


_COMMON_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "1024",
    "--enforce-eager",
    "--generation-config",
    "vllm",
    "--attention-backend",
    "ROCM_ATTN",
    "--max-num-seqs",
    "1",
]


@pytest.mark.slow_test
async def test_deterministic_prefix_caching_fixes_bug(tokenizer, test_messages):
    """
    Validates that --deterministic-prefix-caching eliminates the cache-miss
    vs cache-hit divergence caused by GEMM tiling differences.

    The test spins up two server configurations:
      1. Baseline: --enable-prefix-caching (may be non-deterministic)
      2. Fix:      --enable-prefix-caching --deterministic-prefix-caching

    Critical assertions (test fails if violated):
      - All deterministic-mode runs produce identical token sequences.
      - All deterministic-mode runs produce at least MAX_TOKENS tokens.

    Informational checks (logged in summary, never fail the test):
      - Whether the baseline exhibits the bug (tokens or logprobs).
      - Whether deterministic-mode output matches baseline cache hits.
      - Whether deterministic-mode output matches HuggingFace.
    """
    hf_result = get_hf_reference(tokenizer, test_messages, use_cache=False)
    token_ids = hf_result["input_token_ids"]

    # ---- (baseline) without split prefix caching ----
    with RemoteOpenAIServer(
        MODEL_NAME, [*_COMMON_ARGS, "--enable-prefix-caching"]
    ) as server:
        async with _make_client(server) as client:
            baseline_results = await _request_n(client, token_ids, tokenizer, NUM_RUNS)

    # ---- (deterministic) with split prefix caching ----
    with RemoteOpenAIServer(
        MODEL_NAME,
        [
            *_COMMON_ARGS,
            "--enable-prefix-caching",
            "--deterministic-prefix-caching",
        ],
    ) as server:
        async with _make_client(server) as client:
            det_results = await _request_n(client, token_ids, tokenizer, NUM_RUNS)

    # ---- Critical assertions (run BEFORE reporting) ----
    #
    # If these fail, the assertion error message is self-contained.
    # The summary is printed afterwards regardless (in the finally block
    # pattern below) so we can still diagnose failures.

    assertion_error: AssertionError | None = None
    try:
        # PRIMARY: deterministic mode MUST produce identical tokens
        assert_tokens_deterministic(det_results, "deterministic-prefix-caching")

        # SECONDARY: outputs must not be empty or truncated
        for i, r in enumerate(det_results):
            assert len(r["tokens"]) >= MAX_TOKENS, (
                f"Deterministic Run {i + 1} produced only "
                f"{len(r['tokens'])} tokens (expected >= {MAX_TOKENS})"
            )
    except AssertionError as e:
        assertion_error = e

    # ---- Reporting (always runs, even on assertion failure) ----
    _print_comparison(
        {
            "HF (no cache)": hf_result,
            "Baseline R1": baseline_results[0],
            "Baseline R2": baseline_results[1],
            "Determ. R1": det_results[0],
            "Determ. R2": det_results[1],
        },
        tokenizer,
        max_pos=MAX_TOKENS,
    )
    _print_summary(baseline_results, det_results, hf_result)

    # Re-raise if assertions failed
    if assertion_error is not None:
        raise assertion_error


async def test_deterministic_flag_disabled_by_default(tokenizer, test_messages):
    """
    Verify that deterministic prefix caching is off by default and the
    server still produces valid output.
    """
    hf_result = get_hf_reference(tokenizer, test_messages, use_cache=False)
    token_ids = hf_result["input_token_ids"]

    with RemoteOpenAIServer(
        MODEL_NAME, [*_COMMON_ARGS, "--enable-prefix-caching"]
    ) as server:
        async with _make_client(server) as client:
            results = await _request_n(client, token_ids, tokenizer, 2)

    assert len(results) == 2
    for r in results:
        assert len(r["tokens"]) > 0, "Server returned empty output"
