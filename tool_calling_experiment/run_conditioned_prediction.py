#!/usr/bin/env python3
"""Conditioned prediction experiment.

Tests whether the fine-tuned Qwen3-VL-2B model can USE scene
corrections when given them in the prompt.

For 20 nominal samples (fine_class=negative_incident_zone) that
the model is likely to misclassify as incident_zone, we run two
conditions:

Condition A (baseline): Send the standard prompt with the image.
Condition B (conditioned): Modify the prompt to tell the model
the scene is nominal, and ask it to predict accordingly.

We compare the ODD token and action predictions across conditions.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time

import requests  # type: ignore[import-not-found]

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
MODEL_PATH = "/workspace/vllm/models/checkpoint/"
DATASET_PATH = "/workspace/vllm/models/dataset/"
SERVER_URL = "http://localhost:8300"
OUTPUT_PATH = (
    "/workspace/vllm/tool_calling_experiment"
    "/conditioned_prediction_results.json"
)

# 20 samples: odd_label=nominal, fine_class=negative_incident_zone
NUM_SAMPLES = 20

# ODD token mapping
ODD_TOKEN_MAP = {
    "nominal": "<|odd_nominal|>",
    "incident_zone": "<|odd_incident|>",
    "flooded": "<|odd_flood|>",
    "flagger": "<|odd_flagger|>",
    "mounted_police": "<|odd_policehorse|>",
}

ODD_REVERSE_MAP = {v: k for k, v in ODD_TOKEN_MAP.items()}

# The original ODD instruction block in the training prompt
_ORIG_ODD_BLOCK = (
    "Before the actions block, output exactly one"
    " ODD (Operational Design Domain) token:\n"
    "- <|odd_nominal|> - normal driving conditions\n"
    "- <|odd_flood|> - flooded roads visible\n"
    "- <|odd_incident|> - incident zone"
    " (accident, emergency vehicles)\n"
    "- <|odd_policehorse|> - police on horseback"
    " present\n"
    "- <|odd_flagger|> - human flagger"
    " directing traffic"
)

_COND_ODD_BLOCK = (
    "The scene has been classified as NOMINAL"
    " (normal driving conditions). "
    "You MUST output <|odd_nominal|> as the"
    " ODD token.\n\n"
    "ODD tokens reference:\n"
    "- <|odd_nominal|> - normal driving conditions\n"
    "- <|odd_flood|> - flooded roads visible\n"
    "- <|odd_incident|> - incident zone"
    " (accident, emergency vehicles)\n"
    "- <|odd_policehorse|> - police on horseback"
    " present\n"
    "- <|odd_flagger|> - human flagger"
    " directing traffic"
)


def parse_odd_token(text: str) -> str | None:
    """Extract ODD scene label from model output."""
    for token, label in ODD_REVERSE_MAP.items():
        if token in text:
            return label
    return None


def parse_actions(text: str) -> list[str]:
    """Extract action tokens from model output."""
    return re.findall(r"<\|action_x_[^|]+\|>", text)


def image_bytes_to_base64_url(img_bytes: bytes) -> str:
    """Convert raw image bytes to a data URL."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _convert_content(
    sample: dict,
    user_msg: dict,
    rewrite_odd: bool = False,
) -> list[dict]:
    """Build content parts from a user message.

    If *rewrite_odd* is True, the ODD instruction block is
    replaced with the conditioned version.
    """
    parts: list[dict] = []
    for part in user_msg["content"]:
        if part["type"] == "text":
            text = part["text"]
            if rewrite_odd and _ORIG_ODD_BLOCK in text:
                text = text.replace(
                    _ORIG_ODD_BLOCK, _COND_ODD_BLOCK
                )
            parts.append({"type": "text", "text": text})
        elif part["type"] == "image":
            img_key = part["path"]
            img_bytes = sample[img_key]
            url = image_bytes_to_base64_url(img_bytes)
            parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
    return parts


def build_standard_messages(sample: dict) -> list[dict]:
    """Build the standard prompt from an MDS sample."""
    user_msg = sample["messages"][0]
    content = _convert_content(sample, user_msg)
    return [{"role": "user", "content": content}]


def build_conditioned_messages(
    sample: dict,
) -> list[dict]:
    """Build a conditioned prompt telling the model nominal."""
    user_msg = sample["messages"][0]
    content = _convert_content(
        sample, user_msg, rewrite_odd=True
    )
    return [{"role": "user", "content": content}]


def send_request(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """Send a chat completion request to vLLM."""
    payload = {
        "model": MODEL_PATH,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _run_trial(
    sample: dict,
    meta: dict,
    trial_num: int,
    total: int,
    sample_idx: int,
) -> dict:
    """Run both conditions for one sample."""
    print(
        f"\n--- Sample {trial_num + 1}/{total} "
        f"(dataset idx={sample_idx}) ---"
    )
    print(f"  GT ODD: {meta.get('odd_label')}")
    print(f"  Fine class: {meta.get('fine_class')}")
    uri = meta.get("chum_uri", "N/A")
    print(f"  Chum URI: {uri}")

    # Condition A: Standard prompt (baseline)
    print("  Running Condition A (baseline)...")
    t0 = time.time()
    msgs_a = build_standard_messages(sample)
    response_a = send_request(msgs_a)
    time_a = time.time() - t0
    odd_a = parse_odd_token(response_a)
    actions_a = parse_actions(response_a)
    print(f"    Response: {response_a[:180]}")
    print(f"    ODD predicted: {odd_a}")
    print(f"    Actions: {len(actions_a)}")
    print(f"    Time: {time_a:.2f}s")

    # Condition B: Conditioned prompt
    print("  Running Condition B (conditioned)...")
    t0 = time.time()
    msgs_b = build_conditioned_messages(sample)
    response_b = send_request(msgs_b)
    time_b = time.time() - t0
    odd_b = parse_odd_token(response_b)
    actions_b = parse_actions(response_b)
    print(f"    Response: {response_b[:180]}")
    print(f"    ODD predicted: {odd_b}")
    print(f"    Actions: {len(actions_b)}")
    print(f"    Time: {time_b:.2f}s")

    return {
        "trial_num": trial_num,
        "dataset_index": sample_idx,
        "chum_uri": meta.get("chum_uri", ""),
        "gt_odd_label": meta.get("odd_label", ""),
        "fine_class": meta.get("fine_class", ""),
        "gt_long_action": meta.get("long_action", ""),
        "gt_lat_action": meta.get("lat_action", ""),
        "condition_a": {
            "response": response_a,
            "odd_predicted": odd_a,
            "actions": actions_a,
            "time_s": round(time_a, 2),
        },
        "condition_b": {
            "response": response_b,
            "odd_predicted": odd_b,
            "actions": actions_b,
            "time_s": round(time_b, 2),
        },
        "scene_changed": odd_a != odd_b,
        "a_correct_scene": odd_a == "nominal",
        "b_correct_scene": odd_b == "nominal",
        "actions_changed": actions_a != actions_b,
    }


def _pct(num: int, den: int) -> float:
    """Percentage, safe against zero division."""
    return round(100 * num / den, 1) if den else 0.0


def _compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics from results."""
    total = len(results)
    a_correct = sum(
        1 for r in results if r["a_correct_scene"]
    )
    b_correct = sum(
        1 for r in results if r["b_correct_scene"]
    )
    scene_changed = sum(
        1 for r in results if r["scene_changed"]
    )
    actions_changed = sum(
        1 for r in results if r["actions_changed"]
    )

    a_incident = sum(
        1
        for r in results
        if r["condition_a"]["odd_predicted"]
        == "incident_zone"
    )
    b_incident = sum(
        1
        for r in results
        if r["condition_b"]["odd_predicted"]
        == "incident_zone"
    )
    follows = sum(
        1
        for r in results
        if r["condition_a"]["odd_predicted"]
        == "incident_zone"
        and r["condition_b"]["odd_predicted"] == "nominal"
    )
    ignores = sum(
        1
        for r in results
        if r["condition_a"]["odd_predicted"]
        == "incident_zone"
        and r["condition_b"]["odd_predicted"]
        == "incident_zone"
    )

    odd_a: dict[str, int] = {}
    odd_b: dict[str, int] = {}
    for r in results:
        oa = r["condition_a"]["odd_predicted"] or "none"
        ob = r["condition_b"]["odd_predicted"] or "none"
        odd_a[oa] = odd_a.get(oa, 0) + 1
        odd_b[ob] = odd_b.get(ob, 0) + 1

    return {
        "experiment": "conditioned_prediction",
        "description": (
            "Tests whether the fine-tuned model responds"
            " to scene corrections in the prompt."
        ),
        "total_samples": total,
        "condition_a_baseline": {
            "correct_scene_count": a_correct,
            "correct_scene_pct": _pct(a_correct, total),
            "incident_zone_count": a_incident,
            "odd_distribution": odd_a,
        },
        "condition_b_conditioned": {
            "correct_scene_count": b_correct,
            "correct_scene_pct": _pct(b_correct, total),
            "incident_zone_count": b_incident,
            "odd_distribution": odd_b,
        },
        "comparison": {
            "scene_changed_count": scene_changed,
            "scene_changed_pct": _pct(
                scene_changed, total
            ),
            "actions_changed_count": actions_changed,
            "actions_changed_pct": _pct(
                actions_changed, total
            ),
            "follows_correction_count": follows,
            "follows_correction_pct": _pct(
                follows, a_incident
            ),
            "ignores_correction_count": ignores,
            "ignores_correction_pct": _pct(
                ignores, a_incident
            ),
        },
    }


def _print_summary(summary: dict, results: list[dict]):
    """Print a human-readable summary."""
    total = summary["total_samples"]
    sa = summary["condition_a_baseline"]
    sb = summary["condition_b_conditioned"]
    sc = summary["comparison"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal samples: {total}")

    print("\nCondition A (baseline):")
    print(
        f"  Scene correct (nominal): "
        f"{sa['correct_scene_count']}/{total} "
        f"({sa['correct_scene_pct']}%)"
    )
    a_iz = sa["incident_zone_count"]
    print(f"  Predicted incident_zone: {a_iz}/{total}")
    print(f"  ODD distribution: {sa['odd_distribution']}")

    print("\nCondition B (conditioned: told scene=nominal):")
    print(
        f"  Scene correct (nominal): "
        f"{sb['correct_scene_count']}/{total} "
        f"({sb['correct_scene_pct']}%)"
    )
    b_iz = sb["incident_zone_count"]
    print(f"  Predicted incident_zone: {b_iz}/{total}")
    print(f"  ODD distribution: {sb['odd_distribution']}")

    print("\nComparison:")
    print(
        f"  Scene changed: "
        f"{sc['scene_changed_count']}/{total} "
        f"({sc['scene_changed_pct']}%)"
    )
    print(
        f"  Actions changed: "
        f"{sc['actions_changed_count']}/{total} "
        f"({sc['actions_changed_pct']}%)"
    )
    if a_iz > 0:
        print(
            f"\n  Of {a_iz} baseline incident_zone:"
        )
        print(
            f"    Followed correction: "
            f"{sc['follows_correction_count']} "
            f"({sc['follows_correction_pct']}%)"
        )
        print(
            f"    Ignored correction: "
            f"{sc['ignores_correction_count']} "
            f"({sc['ignores_correction_pct']}%)"
        )

    # Qualitative examples
    print("\n" + "=" * 70)
    print("QUALITATIVE EXAMPLES")
    print("=" * 70)

    changed = [
        r for r in results if r["scene_changed"]
    ][:3]
    if changed:
        n = len(changed)
        print(f"\n--- Scene CHANGED ({n} shown) ---")
        for ex in changed:
            idx = ex["dataset_index"]
            fc = ex["fine_class"]
            print(f"\n  idx={idx} ({fc}):")
            oa = ex["condition_a"]["odd_predicted"]
            ra = ex["condition_a"]["response"][:100]
            print(f"    Baseline:    ODD={oa}, {ra}")
            ob = ex["condition_b"]["odd_predicted"]
            rb = ex["condition_b"]["response"][:100]
            print(f"    Conditioned: ODD={ob}, {rb}")

    unchanged = [
        r for r in results if not r["scene_changed"]
    ][:3]
    if unchanged:
        n = len(unchanged)
        print(f"\n--- Scene UNCHANGED ({n} shown) ---")
        for ex in unchanged:
            idx = ex["dataset_index"]
            fc = ex["fine_class"]
            print(f"\n  idx={idx} ({fc}):")
            oa = ex["condition_a"]["odd_predicted"]
            ra = ex["condition_a"]["response"][:100]
            print(f"    Baseline:    ODD={oa}, {ra}")
            ob = ex["condition_b"]["odd_predicted"]
            rb = ex["condition_b"]["response"][:100]
            print(f"    Conditioned: ODD={ob}, {rb}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    if a_iz > 0:
        rate = sc["follows_correction_pct"]
        if rate > 50:
            print(
                f"\nThe model RESPONDS to corrective "
                f"prompting: {rate}% corrected."
            )
        elif rate > 20:
            print(
                f"\nThe model PARTIALLY responds: "
                f"{rate}% corrected."
            )
        else:
            print(
                f"\nThe model IGNORES corrective "
                f"prompting: only {rate}% corrected."
            )
    else:
        print(
            "\nBaseline already predicted nominal "
            "for all samples -- no correction needed."
        )


def main():
    """Run the conditioned prediction experiment."""
    print("=" * 70)
    print("CONDITIONED PREDICTION EXPERIMENT")
    print("=" * 70)

    # Check server health
    try:
        r = requests.get(
            f"{SERVER_URL}/health", timeout=5
        )
        assert r.status_code == 200  # noqa: S101
        print(f"Server healthy at {SERVER_URL}")
    except Exception as e:
        print(f"ERROR: Server not healthy: {e}")
        sys.exit(1)

    # Load dataset
    from streaming import StreamingDataset  # type: ignore[import-not-found]

    print(f"Loading dataset from {DATASET_PATH}...")
    ds = StreamingDataset(
        local=DATASET_PATH, shuffle=False
    )
    print(f"Dataset size: {len(ds)}")

    # Find samples
    candidates: list[int] = []
    for i in range(len(ds)):
        meta = ds[i]["metadata"]
        is_nominal = meta.get("odd_label") == "nominal"
        is_neg_iz = (
            meta.get("fine_class")
            == "negative_incident_zone"
        )
        if is_nominal and is_neg_iz:
            candidates.append(i)
        if len(candidates) >= NUM_SAMPLES:
            break

    print(
        f"Selected {len(candidates)} samples "
        f"(negative_incident_zone, GT=nominal)"
    )
    print(f"Sample indices: {candidates}")

    # Run experiments
    results = []
    for trial_num, sample_idx in enumerate(candidates):
        sample = ds[sample_idx]
        meta = sample["metadata"]
        result = _run_trial(
            sample,
            meta,
            trial_num,
            len(candidates),
            sample_idx,
        )
        results.append(result)

    # Compute summary
    summary = _compute_summary(results)
    output = {"summary": summary, "samples": results}

    # Write results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")

    # Print summary
    _print_summary(summary, results)


if __name__ == "__main__":
    main()
