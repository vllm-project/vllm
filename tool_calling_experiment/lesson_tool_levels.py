#!/usr/bin/env python3
"""Lesson: Tool Reasoning Levels -- 4 levels x 2 models x 20 samples.

Tests whether model intelligence or tool scaffolding drives correction
accuracy. Each level provides progressively richer reasoning support:
  Level 1: Raw statistics only
  Level 2: Statistics + natural language interpretation
  Level 3: Visual checklists and decision procedures
  Level 4: Explicit if/then decision rules

This is a multi-turn text-only verification experiment. The model receives
a prior prediction (from the self-consistency DB) and uses tools to
verify/revise it.

Usage:
    python tool_calling_experiment/lesson_tool_levels.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import requests as _requests  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

# ------------------------------------------------------------------
# Path setup -- keep tool_calling_experiment importable without
# shadowing the installed vllm package.
# ------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p
        for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]

from server_utils import (  # noqa: E402  # ty: ignore[unresolved-import]
    VLLMServer as _BaseVLLMServer,
)
from tools_v2 import (  # noqa: E402  # ty: ignore[unresolved-import]
    execute_tool_v2,
    get_tools_for_level,
)


class VLLMServer(_BaseVLLMServer):
    """VLLMServer with stdout redirected to a log file.

    The base class pipes stdout to subprocess.PIPE. If nobody reads
    the pipe, the 64 KB buffer fills and the server process blocks,
    then dies. This subclass redirects to a log file instead.
    """

    def start(self, timeout: int = 360) -> None:  # type: ignore[override]
        """Start vLLM server, logging to a file instead of PIPE."""
        import subprocess as _sp

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        cmd = [
            "vllm", "serve", self.model_path,
            "--trust-remote-code",
            "--max-model-len", str(self.max_model_len),
            "--enforce-eager",
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_mem),
        ]
        if self.enable_tools:
            cmd.extend([
                "--enable-auto-tool-choice",
                "--tool-call-parser", "hermes",
            ])

        log_path = os.path.join(
            _DIR, f"server_{self.port}.log"
        )
        self._log_fh = open(log_path, "w")  # noqa: SIM115
        print(f"  Server log: {log_path}")

        self.proc = _sp.Popen(
            cmd,
            env=env,
            stdout=self._log_fh,
            stderr=_sp.STDOUT,
        )

        # Poll /health until ready
        for elapsed in range(timeout):
            try:
                r = _requests.get(
                    f"{self.base_url}/health", timeout=2
                )
                if r.status_code == 200:
                    print(
                        f"  Server ready in {elapsed}s on "
                        f"port {self.port}"
                    )
                    return
            except Exception:
                pass
            time.sleep(1)
            if self.proc.poll() is not None:
                self._log_fh.close()
                with open(log_path) as f:
                    tail = f.read()[-2000:]
                raise RuntimeError(
                    f"Server died during startup:\n{tail}"
                )

        raise RuntimeError(
            f"Server failed to start within {timeout}s"
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Chat with a longer HTTP timeout (300s vs 120s)."""
        payload: dict[str, Any] = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        resp = _requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    def stop(self) -> None:
        """Stop server and close log file."""
        super().stop()
        if hasattr(self, "_log_fh") and self._log_fh:
            self._log_fh.close()


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_2B = 5
GPU_8B = 6
PORT_2B = 8305
PORT_8B = 8306

SC_DB_PATH = os.path.join(
    _PARENT,
    "self_consistency_experiment",
    "self_consistency.db",
)
RESULTS_PATH = os.path.join(_DIR, "tool_levels_results.json")

VALID_SCENES = {
    "nominal",
    "flooded",
    "incident_zone",
    "mounted_police",
    "flagger",
}

# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------
_SCENE_RE = re.compile(
    r"FINAL_SCENE:\s*(\S+)", re.IGNORECASE
)
_LONG_RE = re.compile(
    r"FINAL_LONG_ACTION:\s*(\S+)", re.IGNORECASE
)
_LAT_RE = re.compile(
    r"FINAL_LAT_ACTION:\s*(\S+)", re.IGNORECASE
)


def parse_response(text: str) -> dict[str, str | None]:
    """Parse scene/action predictions from response text."""
    result: dict[str, str | None] = {
        "predicted_scene": None,
        "predicted_long_action": None,
        "predicted_lat_action": None,
    }
    if not text:
        return result

    m = _SCENE_RE.search(text)
    if m:
        val = m.group(1).strip().strip(".,;:\"'`").lower()
        if val in VALID_SCENES:
            result["predicted_scene"] = val

    m = _LONG_RE.search(text)
    if m:
        val = m.group(1).strip().strip(".,;:\"'`").lower()
        if val in {"stop", "slowdown", "proceed", "null"}:
            result["predicted_long_action"] = val

    m = _LAT_RE.search(text)
    if m:
        val = m.group(1).strip().strip(".,;:\"'`").lower()
        if val in {"lc_left", "lc_right", "null"}:
            result["predicted_lat_action"] = val

    # Fallback: look for scene type in freeform text
    if result["predicted_scene"] is None:
        lower = text.lower()
        found = [s for s in VALID_SCENES if s in lower]
        if len(found) == 1:
            result["predicted_scene"] = found[0]
        elif len(found) > 1:
            non_nom = [s for s in found if s != "nominal"]
            if len(non_nom) == 1:
                result["predicted_scene"] = non_nom[0]

    return result


# ------------------------------------------------------------------
# Tool call extraction
# ------------------------------------------------------------------
def extract_tool_calls(
    msg: dict[str, Any],
) -> list[tuple[str, dict[str, Any], str]]:
    """Extract (name, args, call_id) from a response."""
    tool_calls = msg.get("tool_calls") or []
    results: list[tuple[str, dict[str, Any], str]] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        call_id = tc.get("id", "")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        else:
            args = raw_args if isinstance(raw_args, dict) else {}
        if name:
            results.append((name, args, call_id))
    return results


# ------------------------------------------------------------------
# Server health check
# ------------------------------------------------------------------
def check_server_health(base_url: str) -> bool:
    """Return True if the server /health endpoint responds 200."""
    try:
        r = _requests.get(f"{base_url}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ------------------------------------------------------------------
# Robust chat with retry
# ------------------------------------------------------------------
def robust_chat(
    server: VLLMServer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0,
    max_tokens: int = 1024,
    tool_choice: str = "auto",
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call server.chat with retry on failure.

    Returns the message dict or raises on persistent failure.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            msg = server.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
            )
            return msg
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                # Check if server is still alive
                if not check_server_health(server.base_url):
                    print(
                        f"    [WARN] Server unhealthy after attempt "
                        f"{attempt + 1}, waiting 10s..."
                    )
                    time.sleep(10)
                    if not check_server_health(server.base_url):
                        raise RuntimeError(
                            f"Server died: {exc}"
                        ) from exc
                else:
                    print(
                        f"    [WARN] Request failed attempt "
                        f"{attempt + 1}: {exc}, retrying..."
                    )
                    time.sleep(2)
    raise RuntimeError(f"All retries failed: {last_err}") from last_err


# ------------------------------------------------------------------
# Sample selection -- specific categories per experiment design
# ------------------------------------------------------------------
def select_samples(db_path: str) -> list[dict[str, Any]]:
    """Select 20 samples from the self-consistency DB.

    Categories:
      10 - predicted incident_zone, GT is nominal (dominant error)
       3 - predicted incident_zone, GT IS incident_zone (true positive)
       3 - predicted nominal, GT is nominal (true negative)
       2 - predicted flooded/flagger wrong
       2 - predicted correctly for non-nominal classes
    """
    import random
    random.seed(42)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    def _fetch(query: str, params: tuple = ()) -> list[dict]:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # Cat 1: pred=incident_zone, gt=nominal (10 samples)
    cat1_all = _fetch(
        "SELECT * FROM predictions "
        "WHERE predicted_scene='incident_zone' AND scene_type_gt='nominal' "
        "ORDER BY sample_id"
    )
    cat1 = random.sample(cat1_all, min(10, len(cat1_all)))

    # Cat 2: pred=incident_zone, gt=incident_zone (3 samples)
    cat2_all = _fetch(
        "SELECT * FROM predictions "
        "WHERE predicted_scene='incident_zone' "
        "AND scene_type_gt='incident_zone' "
        "ORDER BY sample_id"
    )
    cat2 = random.sample(cat2_all, min(3, len(cat2_all)))

    # Cat 3: pred=nominal, gt=nominal (3 samples)
    cat3_all = _fetch(
        "SELECT * FROM predictions "
        "WHERE predicted_scene='nominal' AND scene_type_gt='nominal' "
        "ORDER BY sample_id"
    )
    cat3 = random.sample(cat3_all, min(3, len(cat3_all)))

    # Cat 4: pred=flooded/flagger wrong (2 samples)
    cat4_all = _fetch(
        "SELECT * FROM predictions "
        "WHERE (predicted_scene='flooded' AND scene_type_gt != 'flooded') "
        "OR (predicted_scene='flagger' AND scene_type_gt != 'flagger') "
        "ORDER BY sample_id"
    )
    cat4 = random.sample(cat4_all, min(2, len(cat4_all)))

    # Cat 5: correct non-nominal (2 samples)
    cat5_all = _fetch(
        "SELECT * FROM predictions "
        "WHERE predicted_scene=scene_type_gt AND predicted_scene != 'nominal' "
        "ORDER BY sample_id"
    )
    cat5 = random.sample(cat5_all, min(2, len(cat5_all)))

    conn.close()

    # Tag each sample with its category for analysis
    for s in cat1:
        s["category"] = "false_incident_zone"
    for s in cat2:
        s["category"] = "true_incident_zone"
    for s in cat3:
        s["category"] = "true_nominal"
    for s in cat4:
        s["category"] = "wrong_flooded_flagger"
    for s in cat5:
        s["category"] = "correct_non_nominal"

    samples = cat1 + cat2 + cat3 + cat4 + cat5

    print(f"Selected {len(samples)} samples:")
    cat_counts = Counter(s.get("category", "?") for s in samples)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")
    for s in samples:
        print(
            f"  ID={s['sample_id']:5d} "
            f"pred={s['predicted_scene']:18s} "
            f"gt={s['scene_type_gt']:18s} "
            f"cat={s.get('category', '?')}"
        )
    return samples


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
TURN1_TEMPLATE = """\
A driving scene classification model analyzed a dashcam image and predicted:
- Scene: {predicted_scene}
- Longitudinal action: {predicted_long_action}
- Lateral action: {predicted_lat_action}

Use the available tools to verify these predictions, then provide your final answer.

Provide your answer in this format:
FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REASON: <brief explanation>"""


# ------------------------------------------------------------------
# Behavior classification
# ------------------------------------------------------------------
def classify_behavior(
    original_scene: str,
    final_scene: str | None,
    response_text: str,
    tool_calls_made: list[dict],
) -> str:
    """Classify model behavior: IGNORES / BLINDLY_DEFERS / ACTUALLY_REASONS."""
    if not tool_calls_made:
        return "IGNORES"

    lower = response_text.lower() if response_text else ""

    # Check for evidence of reasoning about tool results
    reasoning_keywords = [
        "base rate", "prior", "rare", "confusion", "confused",
        "compatible", "typical", "co-occurrence", "cooccurrence",
        "feasible", "waypoint", "error rate", "commonly confused",
        "however", "but", "although", "despite", "considering",
        "based on", "according to", "the tool", "statistics",
        "data shows", "data suggests", "evidence",
    ]
    has_reasoning = any(kw in lower for kw in reasoning_keywords)

    if final_scene and final_scene != original_scene:
        # Model revised its prediction
        return "ACTUALLY_REASONS" if has_reasoning else "BLINDLY_DEFERS"
    # Model kept original prediction
    return "ACTUALLY_REASONS" if has_reasoning else "IGNORES"


# ------------------------------------------------------------------
# Multi-turn execution for a single sample
# ------------------------------------------------------------------
def run_single_sample(
    server: VLLMServer,
    sample: dict[str, Any],
    level: int,
) -> dict[str, Any]:
    """Run the 3-turn verification flow for one sample at one level.

    Turn 1: Send prediction + tools -> model calls tools
    Turn 2: Feed back tool results at specified level -> model may call more
    Turn 3: Model gives final answer
    """
    predicted_scene = sample["predicted_scene"]
    predicted_long = sample["predicted_long_action"]
    predicted_lat = sample["predicted_lat_action"]

    prompt = TURN1_TEMPLATE.format(
        predicted_scene=predicted_scene,
        predicted_long_action=predicted_long,
        predicted_lat_action=predicted_lat,
    )

    tools = get_tools_for_level(level)
    conversation: list[dict[str, Any]] = [
        {"role": "user", "content": prompt}
    ]

    tool_calls_made: list[dict[str, Any]] = []
    all_rounds: list[dict[str, Any]] = []
    max_rounds = 2  # Keep context manageable (2 rounds max)
    final_text = ""
    error_msg = None

    for round_num in range(max_rounds):
        try:
            msg = robust_chat(
                server,
                conversation,
                tools=tools,
                temperature=0,
                max_tokens=1024,
                tool_choice="auto",
            )
        except Exception as exc:
            error_msg = str(exc)
            break

        calls = extract_tool_calls(msg)
        content = msg.get("content") or ""

        round_info = {
            "round": round_num + 1,
            "has_tool_calls": len(calls) > 0,
            "tool_calls": [
                {"name": n, "args": a} for n, a, _ in calls
            ],
            "content": content,
        }
        all_rounds.append(round_info)

        if not calls:
            # Model responded with text -- this is the final answer
            final_text = content
            break

        # Execute tool calls and build Turn 2 messages.
        # Clean the assistant message to only include fields
        # that vLLM accepts (avoid extra keys like refusal,
        # annotations, etc. that cause 400 errors).
        clean_msg: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": msg.get("tool_calls", []),
        }
        if msg.get("content"):
            clean_msg["content"] = msg["content"]
        conversation.append(clean_msg)

        for name, args, call_id in calls:
            try:
                tool_result = execute_tool_v2(
                    name, args, level=level
                )
            except Exception as exc:
                tool_result = {"error": str(exc)}

            tool_calls_made.append({
                "name": name,
                "args": args,
                "result": tool_result,
                "round": round_num + 1,
            })

            conversation.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tool_result),
            })
    else:
        # Exhausted all rounds -- force a final text response
        try:
            final_msg = robust_chat(
                server,
                conversation,
                tools=tools,
                temperature=0,
                max_tokens=1024,
                tool_choice="none",
            )
            final_text = final_msg.get("content") or ""
        except Exception as exc:
            error_msg = str(exc)

    return {
        "error": error_msg,
        "tool_calls_made": tool_calls_made,
        "response_text": final_text,
        "rounds": len(all_rounds),
        "all_rounds": all_rounds,
    }


# ------------------------------------------------------------------
# Run all samples for one model at one level
# ------------------------------------------------------------------
def run_level(
    server: VLLMServer,
    model_name: str,
    samples: list[dict[str, Any]],
    level: int,
) -> list[dict[str, Any]]:
    """Run all samples for one model at one level."""
    print(
        f"\n--- {model_name} | Level {level} ---"
    )
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        gt_scene = sample["scene_type_gt"]
        original_pred = sample["predicted_scene"]

        # Health check before each request
        if not check_server_health(server.base_url):
            print(
                f"  [WARN] Server unhealthy before sample {sid}, "
                f"waiting 15s..."
            )
            time.sleep(15)
            if not check_server_health(server.base_url):
                print("  [ERROR] Server dead. Marking remaining as errors.")
                for j in range(i, len(samples)):
                    s = samples[j]
                    results.append({
                        "sample_id": s["sample_id"],
                        "category": s.get("category", "?"),
                        "gt_scene": s["scene_type_gt"],
                        "original_prediction": s["predicted_scene"],
                        "original_correct": s["predicted_scene"] == s["scene_type_gt"],
                        "model": model_name,
                        "level": level,
                        "final_scene": None,
                        "final_long_action": None,
                        "final_lat_action": None,
                        "scene_correct": None,
                        "revised": False,
                        "net_effect": 0,
                        "behavior": "IGNORES",
                        "num_tool_calls": 0,
                        "tools_called": [],
                        "rounds": 0,
                        "response_text": "",
                        "full_response_text": "",
                        "error": "Server died",
                        "latency_s": 0.0,
                    })
                return results

        t0 = time.time()
        outcome = run_single_sample(server, sample, level)
        elapsed = time.time() - t0

        parsed = parse_response(outcome["response_text"])
        final_scene = parsed["predicted_scene"]

        # Did the model revise?
        revised = (
            final_scene is not None
            and final_scene != original_pred
        )

        # Was the revision correct?
        scene_correct = (
            final_scene == gt_scene
            if final_scene is not None
            else None
        )

        # Was the original prediction correct?
        original_correct = original_pred == gt_scene

        # Net effect: +1 if revision fixed, -1 if revision broke
        if revised and scene_correct and not original_correct:
            net_effect = 1  # save
        elif revised and not scene_correct and original_correct:
            net_effect = -1  # break
        else:
            net_effect = 0  # no change or neutral

        # Classify behavior
        behavior = classify_behavior(
            original_pred,
            final_scene,
            outcome["response_text"],
            outcome["tool_calls_made"],
        )

        rec = {
            "sample_id": sid,
            "category": sample.get("category", "?"),
            "gt_scene": gt_scene,
            "original_prediction": original_pred,
            "original_correct": original_correct,
            "model": model_name,
            "level": level,
            "final_scene": final_scene,
            "final_long_action": parsed["predicted_long_action"],
            "final_lat_action": parsed["predicted_lat_action"],
            "scene_correct": scene_correct,
            "revised": revised,
            "net_effect": net_effect,
            "behavior": behavior,
            "num_tool_calls": len(outcome["tool_calls_made"]),
            "tools_called": [
                tc["name"] for tc in outcome["tool_calls_made"]
            ],
            "rounds": outcome["rounds"],
            "response_text": outcome["response_text"][:800],
            "full_response_text": outcome["response_text"],
            "error": outcome["error"],
            "latency_s": round(elapsed, 2),
        }
        results.append(rec)

        # Print progress
        if scene_correct:
            status = "OK"
        elif scene_correct is False:
            status = "WRONG"
        else:
            status = "UNPARSE"

        rev_str = "REV" if revised else "   "
        net_str = (
            f"+{net_effect}" if net_effect > 0
            else str(net_effect) if net_effect < 0
            else " 0"
        )
        tools_str = (
            ", ".join(rec["tools_called"][:3])
            if rec["tools_called"]
            else "none"
        )
        err_str = f" ERR={outcome['error'][:40]}" if outcome["error"] else ""
        print(
            f"  [{i + 1:2d}/{len(samples)}] "
            f"sid={sid:5d} "
            f"gt={gt_scene:16s} "
            f"orig={original_pred:16s} "
            f"final={str(final_scene):16s} "
            f"{status:7s} {rev_str} net={net_str} "
            f"beh={behavior:16s} "
            f"tc={rec['num_tool_calls']} "
            f"({tools_str}) "
            f"{elapsed:.1f}s{err_str}"
        )

        # Small delay between requests to avoid overwhelming the server
        time.sleep(0.5)

    return results


# ------------------------------------------------------------------
# Metrics computation
# ------------------------------------------------------------------
def compute_level_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute metrics for one model at one level."""
    n = len(results)
    if n == 0:
        return {}

    # Tool call rate
    with_tools = sum(1 for r in results if r["num_tool_calls"] > 0)
    tool_call_rate = with_tools / n

    # Tool selection counts
    tool_counter: Counter = Counter()
    for r in results:
        for t in r["tools_called"]:
            tool_counter[t] += 1

    # Revision rate
    revisions = sum(1 for r in results if r["revised"])
    revision_rate = revisions / n

    # Revision accuracy (when revised, was it correct?)
    revised_results = [r for r in results if r["revised"]]
    if revised_results:
        revision_correct = sum(
            1 for r in revised_results if r["scene_correct"]
        )
        revision_accuracy = revision_correct / len(revised_results)
    else:
        revision_correct = 0
        revision_accuracy = 0.0

    # Net improvement
    saves = sum(1 for r in results if r["net_effect"] > 0)
    breaks = sum(1 for r in results if r["net_effect"] < 0)
    net_improvement = saves - breaks

    # Scene accuracy (final)
    parseable = [r for r in results if r["scene_correct"] is not None]
    correct = sum(1 for r in parseable if r["scene_correct"])
    accuracy = correct / len(parseable) if parseable else 0
    unparsed = n - len(parseable)

    # Baseline accuracy (original predictions)
    baseline_correct = sum(1 for r in results if r["original_correct"])
    baseline_accuracy = baseline_correct / n

    # Behavior breakdown
    behavior_counts = Counter(r["behavior"] for r in results)

    # Accuracy by category
    cat_metrics: dict[str, dict[str, Any]] = {}
    for cat in set(r["category"] for r in results):
        cat_results = [r for r in results if r["category"] == cat]
        cat_parseable = [
            r for r in cat_results if r["scene_correct"] is not None
        ]
        cat_correct = sum(
            1 for r in cat_parseable if r["scene_correct"]
        )
        cat_revisions = sum(1 for r in cat_results if r["revised"])
        cat_saves = sum(
            1 for r in cat_results if r["net_effect"] > 0
        )
        cat_breaks = sum(
            1 for r in cat_results if r["net_effect"] < 0
        )
        cat_metrics[cat] = {
            "n": len(cat_results),
            "correct": cat_correct,
            "accuracy": (
                round(cat_correct / len(cat_parseable), 3)
                if cat_parseable else 0
            ),
            "revisions": cat_revisions,
            "saves": cat_saves,
            "breaks": cat_breaks,
            "net": cat_saves - cat_breaks,
        }

    return {
        "n_samples": n,
        "tool_call_rate": round(tool_call_rate, 3),
        "n_with_tools": with_tools,
        "tool_selection": dict(tool_counter.most_common()),
        "revision_rate": round(revision_rate, 3),
        "n_revisions": revisions,
        "revision_accuracy": round(revision_accuracy, 3),
        "n_revision_correct": revision_correct,
        "saves": saves,
        "breaks": breaks,
        "net_improvement": net_improvement,
        "scene_accuracy": round(accuracy, 3),
        "n_correct": correct,
        "n_parseable": len(parseable),
        "n_unparsed": unparsed,
        "baseline_accuracy": round(baseline_accuracy, 3),
        "baseline_correct": baseline_correct,
        "behavior_counts": dict(behavior_counts),
        "category_metrics": cat_metrics,
    }


# ------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------
def print_performance_curve(
    all_metrics: dict[str, dict[str, Any]],
) -> None:
    """Print the performance curve table."""
    print("\n" + "=" * 100)
    print("TOOL REASONING LEVELS: PERFORMANCE CURVE")
    print("=" * 100)

    header = (
        f"{'Level':>5s} | "
        f"{'2B Accuracy':>12s} | {'2B Revisions':>12s} | {'2B Net':>7s} | "
        f"{'8B Accuracy':>12s} | {'8B Revisions':>12s} | {'8B Net':>7s}"
    )
    print(header)
    print("-" * len(header))

    for level in [1, 2, 3, 4]:
        parts = []
        for model in ["2B", "8B"]:
            key = f"{model}|L{level}"
            m = all_metrics.get(key, {})
            if m:
                n_c = m["n_correct"]
                n_p = m["n_parseable"]
                acc_pct = m["scene_accuracy"] * 100
                rev_n = m["n_revisions"]
                net = m["net_improvement"]
                parts.append(
                    f"{acc_pct:5.1f}% ({n_c}/{n_p})"
                )
                parts.append(f"    {rev_n:2d}/20     ")
                net_str = f"+{net}" if net > 0 else str(net)
                parts.append(f"  {net_str:>4s} ")
            else:
                parts.extend(["     N/A     ", "     N/A     ", "   N/A "])

        print(
            f"  {level:>3d} | "
            f"{parts[0]:>12s} | {parts[1]:>12s} | {parts[2]:>7s} | "
            f"{parts[3]:>12s} | {parts[4]:>12s} | {parts[5]:>7s}"
        )

    # Print baseline row
    parts_bl = []
    for model in ["2B", "8B"]:
        key = f"{model}|L1"
        m = all_metrics.get(key, {})
        if m:
            bl_c = m["baseline_correct"]
            acc_pct = m["baseline_accuracy"] * 100
            parts_bl.append(f"{acc_pct:5.1f}% ({bl_c}/20)")
            parts_bl.append("     ---     ")
            parts_bl.append("   ---  ")
        else:
            parts_bl.extend(["     N/A     ", "     N/A     ", "   N/A "])
    print("-" * len(header))
    print(
        f" base | "
        f"{parts_bl[0]:>12s} | {parts_bl[1]:>12s} | {parts_bl[2]:>7s} | "
        f"{parts_bl[3]:>12s} | {parts_bl[4]:>12s} | {parts_bl[5]:>7s}"
    )

    # Behavior breakdown
    print("\n" + "-" * 100)
    print("BEHAVIOR BREAKDOWN")
    print("-" * 100)
    beh_header = (
        f"{'Level':>5s} | {'Model':>5s} | "
        f"{'IGNORES':>8s} | {'BLINDLY_DEFERS':>15s} | "
        f"{'ACTUALLY_REASONS':>17s}"
    )
    print(beh_header)
    print("-" * len(beh_header))

    for level in [1, 2, 3, 4]:
        for model in ["2B", "8B"]:
            key = f"{model}|L{level}"
            m = all_metrics.get(key, {})
            if m:
                bc = m.get("behavior_counts", {})
                ign = bc.get("IGNORES", 0)
                bd = bc.get("BLINDLY_DEFERS", 0)
                ar = bc.get("ACTUALLY_REASONS", 0)
                print(
                    f"  {level:>3d} | {model:>5s} | "
                    f"{ign:>8d} | {bd:>15d} | {ar:>17d}"
                )

    # Category breakdown for key category
    print("\n" + "-" * 100)
    print("FALSE INCIDENT_ZONE RECOVERY (10 samples where pred=IZ, gt=nominal)")
    print("-" * 100)
    cat_header = (
        f"{'Level':>5s} | {'Model':>5s} | "
        f"{'Correct':>8s} | {'Revisions':>10s} | "
        f"{'Saves':>6s} | {'Breaks':>7s} | {'Net':>4s}"
    )
    print(cat_header)
    print("-" * len(cat_header))

    for level in [1, 2, 3, 4]:
        for model in ["2B", "8B"]:
            key = f"{model}|L{level}"
            m = all_metrics.get(key, {})
            if m:
                cm = m.get("category_metrics", {}).get(
                    "false_incident_zone", {}
                )
                if cm:
                    print(
                        f"  {level:>3d} | {model:>5s} | "
                        f"  {cm['correct']:>2d}/10 | "
                        f"    {cm['revisions']:>2d}/10 | "
                        f"  {cm['saves']:>4d} | "
                        f"  {cm['breaks']:>5d} | "
                        f"  {cm['net']:>+3d}"
                    )

    print("=" * 100)


def print_qualitative_examples(
    all_results: list[dict[str, Any]],
) -> None:
    """Print qualitative examples showing how responses differ across levels."""
    print("\n" + "=" * 100)
    print("QUALITATIVE EXAMPLES: HOW RESPONSES DIFFER ACROSS LEVELS")
    print("=" * 100)

    # Group by (model, sample_id)
    by_sample: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in all_results:
        by_sample[(r["model"], r["sample_id"])].append(r)

    # Find interesting examples
    shown = 0
    for (model, sid), recs in sorted(by_sample.items()):
        if shown >= 3:
            break
        recs_sorted = sorted(recs, key=lambda x: x["level"])

        # Find samples where at least one level revised or variation exists
        has_revision = any(r["revised"] for r in recs_sorted)
        final_scenes = set(r.get("final_scene") for r in recs_sorted)
        has_variation = len(final_scenes - {None}) > 1
        if not (has_revision or has_variation):
            continue

        shown += 1
        first = recs_sorted[0]
        print(
            f"\n--- Example {shown}: {model} | "
            f"Sample {sid} | "
            f"GT={first['gt_scene']} | "
            f"Original={first['original_prediction']} ---"
        )
        for r in recs_sorted:
            rev_mark = " [REVISED]" if r["revised"] else ""
            if r["scene_correct"] is None:
                correct_mark = " UNPARSED"
            elif r["scene_correct"]:
                correct_mark = " CORRECT"
            else:
                correct_mark = " WRONG"
            print(
                f"  Level {r['level']}: "
                f"final={str(r['final_scene']):16s}"
                f"{correct_mark}{rev_mark} "
                f"| behavior={r['behavior']}"
            )
            # Show first 200 chars of response
            text = r.get("response_text", "")[:200]
            if text:
                text_oneline = text.replace("\n", " ")
                print(f"    Response: {text_oneline}...")

    if shown == 0:
        print("  No examples with variation across levels found.")

    print("=" * 100)


# ------------------------------------------------------------------
# Run one model across all 4 levels (restart server each level)
# ------------------------------------------------------------------
def run_model_all_levels(
    model_name: str,
    model_path: str,
    gpu_id: int,
    port: int,
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run all 4 levels for one model, restarting server per level."""
    results: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}

    for level in [1, 2, 3, 4]:
        print(
            f"\n{'='*60}"
            f"\n{model_name} Level {level}: Starting server on "
            f"GPU {gpu_id}, port {port}..."
            f"\n{'='*60}"
        )
        server = VLLMServer(
            model_path=model_path,
            port=port,
            gpu_id=gpu_id,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
            enable_tools=True,
        )
        try:
            server.start(timeout=420)
            level_results = run_level(
                server=server,
                model_name=model_name,
                samples=samples,
                level=level,
            )
            results.extend(level_results)

            key = f"{model_name}|L{level}"
            m = compute_level_metrics(level_results)
            metrics[key] = m
            print(
                f"  >> {key}: "
                f"acc={m['scene_accuracy']:.0%}"
                f", revisions={m['n_revisions']}"
                f", net={m['net_improvement']:+d}"
            )
        finally:
            print(f"  Stopping {model_name} server (level {level})...")
            server.stop()
            time.sleep(5)

    return results, metrics


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------
def main() -> None:
    """Entry point for Tool Reasoning Levels experiment."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"Tool Reasoning Levels Experiment -- {ts}")
    print("Multi-turn text-only verification experiment.")
    print("4 levels x 2 models x 20 samples = 160 total runs.")
    print("Server restarts between levels for reliability.")
    print()

    # ----------------------------------------------------------
    # 1. Load samples
    # ----------------------------------------------------------
    print("Loading samples from self-consistency DB...")
    samples = select_samples(SC_DB_PATH)
    if len(samples) < 20:
        print(f"WARNING: Only {len(samples)} samples (wanted 20)")

    # ----------------------------------------------------------
    # 2. Run both models sequentially
    # ----------------------------------------------------------
    all_results: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, Any]] = {}

    # --- 2B model ---
    r2b, m2b = run_model_all_levels(
        "2B", MODEL_2B, GPU_2B, PORT_2B, samples
    )
    all_results.extend(r2b)
    all_metrics.update(m2b)
    time.sleep(10)  # Extra sleep before switching models

    # --- 8B model ---
    r8b, m8b = run_model_all_levels(
        "8B", MODEL_8B, GPU_8B, PORT_8B, samples
    )
    all_results.extend(r8b)
    all_metrics.update(m8b)

    # ----------------------------------------------------------
    # 3. Print results
    # ----------------------------------------------------------
    print_performance_curve(all_metrics)
    print_qualitative_examples(all_results)

    # ----------------------------------------------------------
    # 4. Save full results
    # ----------------------------------------------------------
    detailed_for_save = []
    for r in all_results:
        save_rec = {
            k: v for k, v in r.items()
            if k != "full_response_text"
        }
        detailed_for_save.append(save_rec)

    output = {
        "experiment": "tool_reasoning_levels",
        "timestamp": ts,
        "note": (
            "Multi-turn text-only verification. 4 levels of tool "
            "reasoning scaffolding x 2 model sizes x 20 samples. "
            "Server restarted between levels for reliability."
        ),
        "n_samples": len(samples),
        "n_runs": len(all_results),
        "levels": [1, 2, 3, 4],
        "models": ["2B", "8B"],
        "samples": [
            {
                "sample_id": s["sample_id"],
                "category": s.get("category"),
                "gt_scene": s["scene_type_gt"],
                "original_prediction": s.get("predicted_scene"),
                "original_long_action": s.get(
                    "predicted_long_action"
                ),
                "original_lat_action": s.get(
                    "predicted_lat_action"
                ),
                "generated_text": s.get("generated_text"),
            }
            for s in samples
        ],
        "metrics": all_metrics,
        "detailed_results": detailed_for_save,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"  Total runs: {len(all_results)}")
    print(f"  Results file: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
