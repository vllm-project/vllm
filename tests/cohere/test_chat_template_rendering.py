#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat template rendering observability test.

Starts a vLLM server (without reasoning/tool parsers), sends 1 sample per
bee_eval task using the **same code path** as ``test_bee_samples.py`` (via
:func:`send_sample`) so chat templating + tokenization behavior matches what a
real client sees, and logs at each stage:

  1. Raw messages (OpenAI messages format)
  2. Rendered prompt after applying the chat template (server-side)
  3. Token IDs after tokenization (server-side)
  4. Raw generation from the server

For (2) and (3) we call vLLM's ``POST /tokenize`` endpoint with the same
messages, which goes through the identical ``OpenAIServingRender`` pipeline as
``/v1/chat/completions``. This guarantees the logged prompt and tokens reflect
exactly what the model received during generation — no local re-tokenization
that could diverge from the server's behavior.

Runs twice: once with thinking_token_budget and once without, to verify
the chat template + reasoning interaction.

Designed to run as part of CI via run_tests.sh
(TEST_GROUP=template_tokenizer_parser_check).

Env vars:
  CT_MODEL_DIR        Model checkpoint path (only used to derive model name)
  CT_MODEL_NAME       Served model name (default: basename of CT_MODEL_DIR)
  CT_BASE_URL         Server URL (default: http://localhost:8000/v1)
  CT_DATA_DIR         Bee eval data dir (default: tests/cohere/bee_eval_data)
  CT_OUTPUT_DIR       Output directory for logs (default: /root/output)
  CT_OUTPUT_SUFFIX    Optional filename suffix appended to output JSON
                      (e.g. "no_parsers" -> chat_template_rendering_no_parsers.json).
  CT_PARSER_MODE      Informational tag stored in output. Set by run_tests.sh
                      to "no_parsers" or "with_parsers" so a single output
                      directory can hold both passes side-by-side.
  CT_THINKING_BUDGET  Thinking budget for the "with thinking" run (default: 2048)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bee_eval_checker import EvalSample, get_task
from test_bee_samples import TASK_CONFIG, send_sample

MODEL_DIR = os.environ.get("CT_MODEL_DIR", "/root/engines/c5-3a30t_fp8")
MODEL_NAME = os.environ.get("CT_MODEL_NAME", "") or Path(MODEL_DIR).name
BASE_URL = os.environ.get("CT_BASE_URL", "http://localhost:8000/v1")
DATA_DIR = os.environ.get("CT_DATA_DIR", "tests/cohere/bee_eval_data")
OUTPUT_DIR = os.environ.get("CT_OUTPUT_DIR", "/root/output")
OUTPUT_SUFFIX = os.environ.get("CT_OUTPUT_SUFFIX", "")
PARSER_MODE = os.environ.get("CT_PARSER_MODE", "")
THINKING_BUDGET = int(os.environ.get("CT_THINKING_BUDGET", "2048"))

# /tokenize and /detokenize are mounted at the server root, not under /v1.
SERVER_ROOT = BASE_URL.rstrip("/").rsplit("/v1", 1)[0]
TOKENIZE_URL = f"{SERVER_ROOT}/tokenize"
DETOKENIZE_URL = f"{SERVER_ROOT}/detokenize"


@dataclass
class SampleLog:
    task: str
    thinking_budget: int | None
    raw_messages: list[dict[str, Any]]
    rendered_text: str
    token_count: int
    token_ids_first_50: list[int]
    tokens_decoded_first_30: list[dict[str, Any]]
    ground_truth: str | list[str]
    generation: str
    reasoning: str
    elapsed_s: float
    error: str | None = None


async def server_tokenize(
    http: httpx.AsyncClient,
    model: str,
    messages: list[dict],
) -> tuple[list[int], list[str]]:
    """Call vLLM's ``/tokenize`` endpoint with chat messages.

    Returns ``(token_ids, token_strs)`` exactly as the server tokenizes them
    when it processes a ``/v1/chat/completions`` request with the same
    messages — same chat template, same ``add_special_tokens`` resolution,
    same multimodal handling.
    """
    resp = await http.post(
        TOKENIZE_URL,
        json={
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
            "return_token_strs": True,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["tokens"], data.get("token_strs") or []


async def server_detokenize(
    http: httpx.AsyncClient, model: str, token_ids: list[int]
) -> str:
    """Reconstruct the rendered prompt text via the server's /detokenize.

    Using the server keeps the test free of any local tokenization logic that
    could diverge from what the model actually saw.
    """
    if not token_ids:
        return ""
    resp = await http.post(
        DETOKENIZE_URL,
        json={"model": model, "tokens": token_ids},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json().get("prompt", "")


async def run_one_pass(
    http: httpx.AsyncClient,
    client: AsyncOpenAI,
    thinking_budget: int | None,
    label: str,
) -> list[SampleLog]:
    """Run 1 sample per task with the given thinking budget."""
    print(f"\n{'#' * 80}")
    print(f"# RUN: {label} (thinking_budget={thinking_budget})")
    print(f"{'#' * 80}")

    logs: list[SampleLog] = []

    for task_name, cfg in TASK_CONFIG.items():
        task = get_task(task_name)
        data_path = Path(DATA_DIR) / cfg.data_file
        if not data_path.exists():
            print(f"\n  SKIP {task_name}: {data_path} not found")
            continue

        samples: list[EvalSample] = task.load_samples(str(data_path), n=1)
        if not samples:
            print(f"\n  SKIP {task_name}: no samples")
            continue

        sample = samples[0]
        messages = sample.messages

        print(f"\n{'=' * 70}")
        print(f"  TASK: {task_name} | thinking_budget={thinking_budget}")
        print(f"{'=' * 70}")

        # 1. Raw messages
        print(f"\n  [1] RAW MESSAGES ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        if p.get("type") == "text":
                            parts.append(p.get("text", "")[:100])
                        else:
                            parts.append(f"[{p.get('type', '?')}]")
                preview = " ".join(parts)[:200]
            else:
                preview = str(content)[:200]
            print(f"      [{i}] {role}: {preview}")

        # 2 + 3. Server-side rendering + tokenization (same path as the
        # /v1/chat/completions request below). We get the rendered prompt
        # text via /detokenize on those same token IDs, so every artifact we
        # log comes from the server with no local re-tokenization.
        try:
            token_ids, token_strs = await server_tokenize(http, MODEL_NAME, messages)
            rendered = await server_detokenize(http, MODEL_NAME, token_ids)
        except Exception as exc:
            print(f"\n  [2/3] /tokenize or /detokenize FAILED: {exc}")
            logs.append(
                SampleLog(
                    task=task_name,
                    thinking_budget=thinking_budget,
                    raw_messages=messages,
                    rendered_text="",
                    token_count=0,
                    token_ids_first_50=[],
                    tokens_decoded_first_30=[],
                    ground_truth=sample.ground_truth,
                    generation="",
                    reasoning="",
                    elapsed_s=0.0,
                    error=f"/tokenize|/detokenize: {exc}",
                )
            )
            continue

        print(f"\n  [2] RENDERED TEXT ({len(rendered)} chars):")
        print(f"      {rendered[:500]}")
        if len(rendered) > 500:
            print(f"      ... ({len(rendered)} total)")

        decoded = [
            {"id": tid, "text": ts} for tid, ts in zip(token_ids[:30], token_strs[:30])
        ]
        print(f"\n  [3] TOKENIZATION ({len(token_ids)} tokens):")
        print(f"      IDs (first 50): {token_ids[:50]}")
        print("      Decoded (first 15):")
        for d in decoded[:15]:
            print(f"        id={d['id']:6d} -> {repr(d['text'])}")

        # 4. Send to server using the SAME path as test_bee_samples.py.
        # ``send_sample`` returns a SampleResult with generation, reasoning,
        # elapsed time, and error.
        result = await send_sample(
            client=client,
            model=MODEL_NAME,
            sample=sample,
            max_tokens=cfg.max_tokens,
            index=0,
            task=task,
            thinking_budget=thinking_budget,
        )

        print(f"\n  [4] GENERATION ({result.elapsed:.1f}s):")
        if result.error:
            print(f"      ERROR: {result.error}")
        else:
            print(f"      Content (first 300 chars): {result.generation[:300]}")
            if result.reasoning:
                print(f"      Reasoning (first 200 chars): {result.reasoning[:200]}")

        gt = sample.ground_truth
        print(f"\n      Ground truth: {gt if isinstance(gt, str) else str(gt)[:200]}")

        logs.append(
            SampleLog(
                task=task_name,
                thinking_budget=thinking_budget,
                raw_messages=messages,
                rendered_text=rendered,
                token_count=len(token_ids),
                token_ids_first_50=token_ids[:50],
                tokens_decoded_first_30=decoded,
                ground_truth=gt,
                generation=result.generation,
                reasoning=result.reasoning,
                elapsed_s=round(result.elapsed, 2),
                error=result.error,
            )
        )

    return logs


def _parser_signal(log: SampleLog) -> dict[str, Any]:
    """Diagnostic flags showing whether the reasoning parser handled the output.

    With parsers enabled, a healthy run has ``reasoning`` populated and ``content``
    free of thinking-mode special tokens. A chat template that puts
    ``<|START_THINKING|>`` in the prompt (instead of letting the model emit it)
    breaks the parser's state machine: ``reasoning`` ends up empty and
    ``<|END_THINKING|>`` / ``<|START_TEXT|>`` leak into ``content``.
    """
    leaked_tokens = [
        t
        for t in (
            "<|START_THINKING|>",
            "<|END_THINKING|>",
            "<|START_TEXT|>",
            "<|END_TEXT|>",
        )
        if t in log.generation
    ]
    return {
        "reasoning_populated": bool(log.reasoning),
        "content_has_special_tokens": bool(leaked_tokens),
        "leaked_special_tokens": leaked_tokens,
    }


async def main():
    print(f"Model name: {MODEL_NAME}")
    print(f"Base URL: {BASE_URL}")
    print(f"Tokenize URL: {TOKENIZE_URL}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Parser mode: {PARSER_MODE or '(unset)'}")
    print(f"Output suffix: {OUTPUT_SUFFIX or '(none)'}")
    print(
        "Server-side rendering + tokenization via /tokenize "
        "(same pipeline as /v1/chat/completions)."
    )

    client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")
    async with httpx.AsyncClient() as http:
        # Run without thinking budget
        logs_no_tb = await run_one_pass(
            http, client, thinking_budget=None, label="NO thinking budget"
        )

        # Run with thinking budget
        logs_with_tb = await run_one_pass(
            http,
            client,
            thinking_budget=THINKING_BUDGET,
            label=f"WITH thinking budget={THINKING_BUDGET}",
        )

    # Write structured output. The suffix lets a single output dir hold both
    # passes (no_parsers, with_parsers) without overwriting.
    all_logs = logs_no_tb + logs_with_tb
    fname = (
        f"chat_template_rendering_{OUTPUT_SUFFIX}.json"
        if OUTPUT_SUFFIX
        else "chat_template_rendering.json"
    )
    output_path = Path(OUTPUT_DIR) / fname
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for log in all_logs:
        d = {
            "task": log.task,
            "thinking_budget": log.thinking_budget,
            "parser_mode": PARSER_MODE,
            "rendered_text_length": len(log.rendered_text),
            "rendered_text_first_500": log.rendered_text[:500],
            "token_count": log.token_count,
            "token_ids_first_50": log.token_ids_first_50,
            "tokens_decoded_first_30": log.tokens_decoded_first_30,
            "ground_truth": log.ground_truth,
            "generation": log.generation,
            "reasoning": log.reasoning,
            "parser_signal": _parser_signal(log),
            "elapsed_s": log.elapsed_s,
            "error": log.error,
        }
        serializable.append(d)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nStructured output written to {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY (parser_mode={PARSER_MODE or 'unset'})")
    print(f"{'=' * 70}")
    errors = [log for log in all_logs if log.error]
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors:
            print(f"    {e.task} (tb={e.thinking_budget}): {e.error}")
    else:
        print("  All requests succeeded.")

    def _print_pass(label: str, logs: list[SampleLog]) -> None:
        print(f"\n  {label}:")
        for log in logs:
            gen_preview = log.generation[:80].replace("\n", "\\n")
            sig = _parser_signal(log)
            r = "yes" if sig["reasoning_populated"] else "no"
            leak = ",".join(sig["leaked_special_tokens"]) or "-"
            print(
                f"    {log.task:15s} tokens={log.token_count:4d} "
                f"reasoning={r:3s} leaked={leak:35s} gen={gen_preview}"
            )

    _print_pass("Without thinking budget", logs_no_tb)
    _print_pass(f"With thinking budget={THINKING_BUDGET}", logs_with_tb)

    # When parsers are enabled, a leaked <|END_THINKING|> in content strongly
    # signals a chat-template / parser mismatch. We *report* this rather than
    # fail the test so engineers can compare templates side-by-side, but it's
    # the metric the test exists to surface.
    if PARSER_MODE == "with_parsers":
        leaked = [
            log for log in all_logs if _parser_signal(log)["leaked_special_tokens"]
        ]
        no_reasoning = [
            log
            for log in all_logs
            if log.thinking_budget is not None and not log.reasoning and not log.error
        ]
        print()
        print(
            f"  Parser diagnostic: {len(leaked)}/{len(all_logs)} samples "
            f"leaked thinking tokens into content."
        )
        print(
            f"  Parser diagnostic: {len(no_reasoning)}/"
            f"{sum(1 for log in all_logs if log.thinking_budget is not None)} "
            f"samples with thinking_budget had empty reasoning."
        )

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
