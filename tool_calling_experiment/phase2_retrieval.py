#!/usr/bin/env python3
# ruff: noqa: E501,E402
# type: ignore[import-not-found]
"""Phase 2 Similar Scene Retrieval Deep Dive: Tasks 43-50.

Tasks:
    43: Retrieval-augmented scene classification (2B, 100 false-IZ samples)
    44: Same with 8B
    45: Retrieval neighbor quality (100 samples)
    46: k=1 vs k=2 vs k=3 (2B, 100 samples x 3 conditions)
    47: Retrieval for rare classes (2B, 50 truly-rare samples)
    48: Adversarial neighbors (2B, 50 samples w/ disagreeing nearest neighbor)
    49: Retrieval + zoom combined (2B, 100 samples)
    50: Token cost analysis

Usage:
    python tool_calling_experiment/phase2_retrieval.py
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sqlite3
import sys
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
# Avoid shadowing installed vllm
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]

from orchestrator import ToolCallingOrchestrator, parse_prediction  # noqa: E402
from server_utils import VLLMServer  # noqa: E402
from visual_tools import (  # noqa: E402
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    TOOL_SIMILAR_SCENES,
    TOOL_ZOOM,
    find_similar_scenes,
    image_to_base64,
    load_sample_image,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
PORT_2B = 8346
PORT_8B = 8347
GPU_2B = 7
GPU_8B = 7  # Reuse same GPU since servers run sequentially

DATASET_PATH = "/workspace/vllm/models/dataset"
SC_DB_PATH = os.path.join(
    os.path.dirname(_DIR),
    "self_consistency_experiment",
    "self_consistency.db",
)

RESULTS_PATH = os.path.join(_DIR, "phase2_retrieval_results.json")
SCENE_METADATA_PATH = os.path.join(_DIR, "scene_metadata.json")

VALID_SCENES = {"nominal", "flooded", "incident_zone", "mounted_police", "flagger"}
SYSTEM_PROMPT = "The image is 504x336 pixels."

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


def _load_db_data() -> dict[int, dict[str, str]]:
    """Load GT data and predictions from self-consistency DB."""
    data: dict[int, dict[str, str]] = {}
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT sample_id, scene_type_gt, long_action_gt, lat_action_gt, "
        "predicted_scene FROM predictions"
    ).fetchall()
    for sid, scene_gt, long_gt, lat_gt, pred in rows:
        data[sid] = {
            "scene_type_gt": scene_gt,
            "long_action_gt": long_gt,
            "lat_action_gt": lat_gt,
            "predicted_scene": pred,
        }
    conn.close()
    return data


def _get_false_iz_samples(db_data: dict, n: int = 100) -> list[int]:
    """Get sample IDs where baseline predicted incident_zone but GT is not."""
    candidates = [
        sid for sid, d in db_data.items()
        if d["predicted_scene"] == "incident_zone"
        and d["scene_type_gt"] != "incident_zone"
    ]
    random.shuffle(candidates)
    return candidates[:n]


def _get_rare_class_samples(db_data: dict, n: int = 50) -> list[int]:
    """Get samples from rare classes: flooded, incident_zone, flagger, mounted_police."""
    rare_classes = {"flooded", "incident_zone", "flagger", "mounted_police"}
    candidates = [
        sid for sid, d in db_data.items()
        if d["scene_type_gt"] in rare_classes
    ]
    random.shuffle(candidates)
    return candidates[:n]


def _load_faiss_metadata() -> list[dict[str, Any]]:
    """Load FAISS index metadata."""
    with open(INDEX_METADATA_PATH, "rb") as f:
        return pickle.load(f)


def _get_adversarial_samples(
    faiss_metadata: list[dict],
    db_data: dict,
    n: int = 50,
) -> list[int]:
    """Find samples where nearest FAISS neighbor has a DIFFERENT GT scene type.

    We need to search the FAISS index for each candidate and check if
    the nearest neighbor disagrees.
    """
    import faiss  # type: ignore[import-not-found]

    index = faiss.read_index(FAISS_INDEX_PATH)

    # Build sample_id -> FAISS row index map
    sid_to_faiss_idx: dict[int, int] = {}
    for i, meta in enumerate(faiss_metadata):
        sid_to_faiss_idx[meta["dataset_index"]] = i

    # Get all embeddings
    n_total = index.ntotal
    dim = index.d
    all_vecs = np.zeros((n_total, dim), dtype=np.float32)
    for i in range(n_total):
        all_vecs[i] = index.reconstruct(i)

    adversarial = []
    # Search for nearest neighbor (k=2: self + nearest)
    for sid, data in db_data.items():
        if sid not in sid_to_faiss_idx:
            continue
        fidx = sid_to_faiss_idx[sid]
        query = all_vecs[fidx:fidx+1]
        scores, indices = index.search(query, 5)

        for j in range(5):
            nn_idx = int(indices[0][j])
            nn_score = float(scores[0][j])
            if nn_score > 0.999:
                continue  # skip self
            if nn_idx < 0 or nn_idx >= len(faiss_metadata):
                continue
            nn_scene = faiss_metadata[nn_idx]["scene_type_gt"]
            if nn_scene != data["scene_type_gt"]:
                adversarial.append(sid)
            break  # only check nearest non-self neighbor

        if len(adversarial) >= n:
            break

    return adversarial[:n]


def _build_retrieval_prompt_with_neighbors(
    query_image_path: str,
    k: int = 3,
    exclude_self_idx: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run FAISS retrieval and build message list with neighbor images.

    Returns (extra_messages, retrieval_info).
    extra_messages: list of dicts to inject into the conversation
    retrieval_info: metadata about the retrieval
    """
    t0 = time.monotonic()
    result = find_similar_scenes(query_image_path, k=k + 2)
    search_time_ms = (time.monotonic() - t0) * 1000

    if "error" in result:
        return [], {"error": result["error"], "search_time_ms": search_time_ms}

    similar = result.get("similar_images", [])

    # Filter out self if needed
    if exclude_self_idx is not None:
        similar = [
            s for s in similar
            if s["dataset_index"] != exclude_self_idx
        ]

    similar = similar[:k]

    # Build neighbor context text and images
    neighbor_descriptions = []
    neighbor_images_b64 = []
    neighbor_info = []

    for i, s in enumerate(similar):
        desc = (
            f"Similar scene {i+1}: "
            f"scene_type={s['ground_truth_scene']}, "
            f"long_action={s['ground_truth_long_action']}, "
            f"lat_action={s['ground_truth_lat_action']}, "
            f"similarity={s['similarity_score']:.3f}"
        )
        neighbor_descriptions.append(desc)

        if s.get("image"):
            b64 = image_to_base64(s["image"])
            neighbor_images_b64.append(b64)
        else:
            neighbor_images_b64.append(None)

        neighbor_info.append({
            "dataset_index": s["dataset_index"],
            "ground_truth_scene": s["ground_truth_scene"],
            "ground_truth_long_action": s["ground_truth_long_action"],
            "ground_truth_lat_action": s["ground_truth_lat_action"],
            "similarity_score": s["similarity_score"],
        })

    # Build a context message with neighbor images
    context_parts: list[dict[str, Any]] = []
    context_parts.append({
        "type": "text",
        "text": (
            "Here are similar scenes from a reference database "
            "with their known labels:\n\n"
            + "\n".join(neighbor_descriptions)
        ),
    })

    for i, b64 in enumerate(neighbor_images_b64):
        if b64:
            context_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })

    consensus = result.get("consensus", {})

    retrieval_info = {
        "k_requested": k,
        "k_returned": len(similar),
        "neighbors": neighbor_info,
        "consensus": consensus,
        "search_time_ms": round(search_time_ms, 1),
    }

    return context_parts, retrieval_info


def _run_retrieval_classification(
    server_url: str,
    sample_ids: list[int],
    k: int = 3,
    db_data: dict | None = None,
    model_label: str = "2B",
    use_zoom: bool = False,
    max_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """Run retrieval-augmented classification on a list of samples.

    For each sample:
    1. Load the query image
    2. Run FAISS retrieval for k neighbors
    3. Build conversation with neighbor images + labels
    4. Ask the model to classify

    Returns list of per-sample result dicts.
    """
    import requests

    results: list[dict[str, Any]] = []

    prompt_text = (
        "Classify this driving scene. "
        "I'll show you similar scenes from a reference database "
        "with their known labels. Use them to inform your classification.\n\n"
        "Scene types: nominal, flooded, incident_zone, mounted_police, flagger.\n\n"
        "After analysis, output:\n"
        "FINAL_SCENE: <scene_type>\n"
        "FINAL_LONG_ACTION: <action>\n"
        "FINAL_LAT_ACTION: <action>"
    )

    tool_defs = []
    tool_fns = {}
    if use_zoom:
        tool_defs = [TOOL_ZOOM, TOOL_SIMILAR_SCENES]
        tool_fns = {
            "zoom_region": zoom_region,
            "find_similar_scenes": find_similar_scenes,
        }

    for i, sid in enumerate(sample_ids):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{model_label}] Sample {i+1}/{len(sample_ids)} (sid={sid})...")

        try:
            img_path = load_sample_image(sid)
        except Exception as e:
            results.append({
                "sample_id": sid,
                "error": f"Failed to load image: {e}",
            })
            continue

        gt_scene = db_data[sid]["scene_type_gt"] if db_data else "unknown"

        # Get retrieval neighbors
        t_ret_start = time.monotonic()
        context_parts, retrieval_info = _build_retrieval_prompt_with_neighbors(
            img_path, k=k, exclude_self_idx=sid,
        )
        t_ret_end = time.monotonic()
        retrieval_ms = (t_ret_end - t_ret_start) * 1000

        if "error" in retrieval_info:
            results.append({
                "sample_id": sid,
                "gt_scene": gt_scene,
                "error": f"Retrieval failed: {retrieval_info['error']}",
            })
            continue

        # Build conversation
        # System message
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # User message with query image + prompt
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(img_path)}",
                },
            },
        ]
        # Add neighbor context
        user_content.extend(context_parts)

        messages.append({"role": "user", "content": user_content})

        # Call model
        t_gen_start = time.monotonic()
        try:
            payload: dict[str, Any] = {
                "model": model_label,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if use_zoom and tool_defs:
                payload["tools"] = tool_defs
                payload["tool_choice"] = "auto"

            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            usage = resp.json().get("usage", {})
        except Exception as e:
            results.append({
                "sample_id": sid,
                "gt_scene": gt_scene,
                "error": f"Model call failed: {e}",
                "retrieval_info": retrieval_info,
            })
            continue
        t_gen_end = time.monotonic()
        gen_ms = (t_gen_end - t_gen_start) * 1000

        # If using tools and model requested a tool call, run orchestrator instead
        if use_zoom and msg.get("tool_calls"):
            # Use the orchestrator for multi-turn
            orch = ToolCallingOrchestrator(
                server_url=server_url,
                tools=tool_fns,
                tool_definitions=tool_defs,
                max_tool_rounds=3,
                temperature=0,
                max_tokens=max_tokens,
            )
            # Re-run with orchestrator, providing the context in the prompt
            neighbor_text = "\n".join([
                f"Similar scene {j+1}: scene_type={n['ground_truth_scene']}, "
                f"similarity={n['similarity_score']:.3f}"
                for j, n in enumerate(retrieval_info.get("neighbors", []))
            ])
            combined_prompt = (
                f"{prompt_text}\n\n"
                f"Reference database results:\n{neighbor_text}\n\n"
                "You also have a zoom_region tool available to inspect "
                "specific parts of the image more closely."
            )
            orch_result = orch.run(
                image_path=img_path,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=combined_prompt,
            )
            content = orch_result["final_text"]
            gen_ms = orch_result["generation_ms"]

        # Parse prediction
        prediction = parse_prediction(content)
        predicted_scene = prediction.get("scene")
        correct = (predicted_scene == gt_scene) if predicted_scene else False

        # Check neighbor consensus
        neighbor_scenes = [
            n["ground_truth_scene"]
            for n in retrieval_info.get("neighbors", [])
        ]
        neighbor_agreement = (
            sum(1 for ns in neighbor_scenes if ns == gt_scene)
            / len(neighbor_scenes)
            if neighbor_scenes else 0
        )

        result_entry = {
            "sample_id": sid,
            "gt_scene": gt_scene,
            "predicted_scene": predicted_scene,
            "correct": correct,
            "retrieval_info": retrieval_info,
            "neighbor_scenes": neighbor_scenes,
            "neighbor_agreement_with_gt": round(neighbor_agreement, 3),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(gen_ms, 1),
            "final_text": content[:500],
            "usage": usage if 'usage' in dir() else {},
        }
        results.append(result_entry)

    return results


# ===================================================================
# Task 43: Retrieval-augmented scene classification (2B, 100 false-IZ)
# ===================================================================
def run_task43(server_2b_url: str, db_data: dict) -> dict[str, Any]:
    _print_separator("TASK 43: Retrieval-Augmented Classification (2B, 100 false-IZ)")

    sample_ids = _get_false_iz_samples(db_data, n=100)
    print(f"  Selected {len(sample_ids)} false-IZ samples")

    # GT distribution
    gt_dist: dict[str, int] = {}
    for sid in sample_ids:
        gt = db_data[sid]["scene_type_gt"]
        gt_dist[gt] = gt_dist.get(gt, 0) + 1
    print(f"  GT distribution: {gt_dist}")

    results = _run_retrieval_classification(
        server_url=server_2b_url,
        sample_ids=sample_ids,
        k=3,
        db_data=db_data,
        model_label="2B",
    )

    # Compute metrics
    correct = sum(1 for r in results if r.get("correct"))
    total = sum(1 for r in results if "predicted_scene" in r and r.get("predicted_scene"))
    errors = sum(1 for r in results if "error" in r)
    accuracy = correct / total if total > 0 else 0

    # Baseline: all predicted as incident_zone (0% correct since these are false-IZ)
    baseline_correct = sum(
        1 for sid in sample_ids
        if db_data[sid]["scene_type_gt"] == "incident_zone"
    )
    baseline_accuracy = baseline_correct / len(sample_ids)

    # Per-class accuracy
    per_class: dict[str, dict[str, int]] = {}
    for r in results:
        gt = r.get("gt_scene", "unknown")
        if gt not in per_class:
            per_class[gt] = {"correct": 0, "total": 0}
        if r.get("predicted_scene"):
            per_class[gt]["total"] += 1
            if r.get("correct"):
                per_class[gt]["correct"] += 1

    summary = {
        "task": "task43_retrieval_classification_2b",
        "model": "2B",
        "n_samples": len(sample_ids),
        "k": 3,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "improvement": round(accuracy - baseline_accuracy, 4),
        "correct": correct,
        "total_with_prediction": total,
        "errors": errors,
        "gt_distribution": gt_dist,
        "per_class_accuracy": {
            k: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0
            for k, v in per_class.items()
        },
        "samples": results,
    }

    print("\n  --- Task 43 Summary ---")
    print(f"  Accuracy with retrieval: {accuracy:.1%} ({correct}/{total})")
    print(f"  Baseline (all IZ): {baseline_accuracy:.1%}")
    print(f"  Improvement: {accuracy - baseline_accuracy:+.1%}")
    print(f"  Per-class: {summary['per_class_accuracy']}")
    return summary


# ===================================================================
# Task 44: Same with 8B
# ===================================================================
def run_task44(server_8b_url: str, db_data: dict, task43_samples: list[int]) -> dict[str, Any]:
    _print_separator("TASK 44: Retrieval-Augmented Classification (8B, 100 false-IZ)")

    sample_ids = task43_samples  # Same samples for comparison
    print(f"  Using same {len(sample_ids)} false-IZ samples as Task 43")

    results = _run_retrieval_classification(
        server_url=server_8b_url,
        sample_ids=sample_ids,
        k=3,
        db_data=db_data,
        model_label="8B",
    )

    correct = sum(1 for r in results if r.get("correct"))
    total = sum(1 for r in results if r.get("predicted_scene"))
    errors = sum(1 for r in results if "error" in r)
    accuracy = correct / total if total > 0 else 0

    baseline_correct = sum(
        1 for sid in sample_ids
        if db_data[sid]["scene_type_gt"] == "incident_zone"
    )
    baseline_accuracy = baseline_correct / len(sample_ids)

    per_class: dict[str, dict[str, int]] = {}
    for r in results:
        gt = r.get("gt_scene", "unknown")
        if gt not in per_class:
            per_class[gt] = {"correct": 0, "total": 0}
        if r.get("predicted_scene"):
            per_class[gt]["total"] += 1
            if r.get("correct"):
                per_class[gt]["correct"] += 1

    summary = {
        "task": "task44_retrieval_classification_8b",
        "model": "8B",
        "n_samples": len(sample_ids),
        "k": 3,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "improvement": round(accuracy - baseline_accuracy, 4),
        "correct": correct,
        "total_with_prediction": total,
        "errors": errors,
        "per_class_accuracy": {
            k: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0
            for k, v in per_class.items()
        },
        "samples": results,
    }

    print("\n  --- Task 44 Summary ---")
    print(f"  Accuracy with retrieval (8B): {accuracy:.1%} ({correct}/{total})")
    print(f"  Baseline (all IZ): {baseline_accuracy:.1%}")
    print(f"  Improvement: {accuracy - baseline_accuracy:+.1%}")
    return summary


# ===================================================================
# Task 45: Retrieval neighbor quality (100 samples)
# ===================================================================
def run_task45(server_2b_url: str, db_data: dict) -> dict[str, Any]:
    _print_separator("TASK 45: Retrieval Neighbor Quality Analysis")

    # Use a mix of samples
    all_sids = list(db_data.keys())
    random.shuffle(all_sids)
    sample_ids = all_sids[:100]

    print(f"  Analyzing neighbor quality for {len(sample_ids)} samples")

    results: list[dict[str, Any]] = []
    total_neighbor_match = 0
    total_neighbors = 0
    agree_correct = 0
    agree_total = 0
    disagree_correct = 0
    disagree_total = 0

    for i, sid in enumerate(sample_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Sample {i+1}/{len(sample_ids)} (sid={sid})...")

        try:
            img_path = load_sample_image(sid)
        except Exception as e:
            results.append({"sample_id": sid, "error": str(e)})
            continue

        gt_scene = db_data[sid]["scene_type_gt"]

        # Run FAISS retrieval
        context_parts, retrieval_info = _build_retrieval_prompt_with_neighbors(
            img_path, k=3, exclude_self_idx=sid,
        )

        if "error" in retrieval_info:
            results.append({"sample_id": sid, "error": retrieval_info["error"]})
            continue

        neighbors = retrieval_info.get("neighbors", [])
        neighbor_scenes = [n["ground_truth_scene"] for n in neighbors]

        # What % of neighbors share the GT scene?
        match_count = sum(1 for ns in neighbor_scenes if ns == gt_scene)
        match_ratio = match_count / len(neighbor_scenes) if neighbor_scenes else 0
        total_neighbor_match += match_count
        total_neighbors += len(neighbor_scenes)

        # Is there agreement (majority matches GT)?
        agree = match_ratio > 0.5

        # Now ask the model to classify (to see if agreement helps accuracy)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Classify this driving scene. Similar scenes from reference:\n"
                    + "\n".join([
                        f"  Similar {j+1}: {ns} (score={n['similarity_score']:.3f})"
                        for j, (ns, n) in enumerate(zip(neighbor_scenes, neighbors))
                    ])
                    + "\n\nTypes: nominal, flooded, incident_zone, mounted_police, flagger."
                    "\n\nFINAL_SCENE: "
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(img_path)}",
                },
            },
        ]
        messages.append({"role": "user", "content": user_content})

        try:
            import requests
            resp = requests.post(
                f"{server_2b_url}/v1/chat/completions",
                json={
                    "model": "2B",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 256,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content", "")
            prediction = parse_prediction(content)
            predicted = prediction.get("scene")
            correct = (predicted == gt_scene) if predicted else False
        except Exception as e:
            predicted = None
            correct = False
            content = f"Error: {e}"

        if predicted:
            if agree:
                agree_total += 1
                if correct:
                    agree_correct += 1
            else:
                disagree_total += 1
                if correct:
                    disagree_correct += 1

        results.append({
            "sample_id": sid,
            "gt_scene": gt_scene,
            "predicted_scene": predicted,
            "correct": correct,
            "neighbor_scenes": neighbor_scenes,
            "neighbor_match_ratio": round(match_ratio, 3),
            "neighbors_agree_with_gt": agree,
        })

    overall_match_rate = (
        total_neighbor_match / total_neighbors
        if total_neighbors > 0 else 0
    )
    agree_accuracy = (
        agree_correct / agree_total if agree_total > 0 else 0
    )
    disagree_accuracy = (
        disagree_correct / disagree_total if disagree_total > 0 else 0
    )

    summary = {
        "task": "task45_neighbor_quality",
        "n_samples": len(sample_ids),
        "overall_neighbor_match_rate": round(overall_match_rate, 4),
        "total_neighbor_match": total_neighbor_match,
        "total_neighbors": total_neighbors,
        "accuracy_when_neighbors_agree": round(agree_accuracy, 4),
        "accuracy_when_neighbors_disagree": round(disagree_accuracy, 4),
        "agree_count": agree_total,
        "disagree_count": disagree_total,
        "correlation_insight": (
            f"When neighbors agree with GT (n={agree_total}): "
            f"{agree_accuracy:.1%} accuracy. "
            f"When neighbors disagree (n={disagree_total}): "
            f"{disagree_accuracy:.1%} accuracy."
        ),
        "samples": results,
    }

    print("\n  --- Task 45 Summary ---")
    print(f"  Neighbor match rate: {overall_match_rate:.1%}")
    print(f"  Accuracy when neighbors agree: {agree_accuracy:.1%} (n={agree_total})")
    print(f"  Accuracy when neighbors disagree: {disagree_accuracy:.1%} (n={disagree_total})")
    return summary


# ===================================================================
# Task 46: k=1 vs k=2 vs k=3 (2B, 100 samples x 3 conditions)
# ===================================================================
def run_task46(server_2b_url: str, db_data: dict) -> dict[str, Any]:
    _print_separator("TASK 46: k=1 vs k=2 vs k=3 (2B)")

    all_sids = list(db_data.keys())
    random.shuffle(all_sids)
    sample_ids = all_sids[:100]

    print(f"  Testing k=1,2,3 on {len(sample_ids)} samples")

    all_k_results: dict[int, list[dict[str, Any]]] = {}

    for k_val in [1, 2, 3]:
        print(f"\n  --- k={k_val} ---")
        results = _run_retrieval_classification(
            server_url=server_2b_url,
            sample_ids=sample_ids,
            k=k_val,
            db_data=db_data,
            model_label="2B",
        )
        all_k_results[k_val] = results

    # Compute accuracy per k
    k_accuracies = {}
    k_details = {}
    for k_val, results in all_k_results.items():
        correct = sum(1 for r in results if r.get("correct"))
        total = sum(1 for r in results if r.get("predicted_scene"))
        accuracy = correct / total if total > 0 else 0
        k_accuracies[k_val] = round(accuracy, 4)
        k_details[k_val] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
        }

    summary = {
        "task": "task46_k_comparison",
        "model": "2B",
        "n_samples": len(sample_ids),
        "k_accuracies": k_accuracies,
        "k_details": k_details,
        "best_k": max(k_accuracies, key=lambda x: k_accuracies[x]),
        "insight": (
            f"k=1: {k_accuracies.get(1, 0):.1%}, "
            f"k=2: {k_accuracies.get(2, 0):.1%}, "
            f"k=3: {k_accuracies.get(3, 0):.1%}"
        ),
        "per_k_samples": {
            str(k): results
            for k, results in all_k_results.items()
        },
    }

    print("\n  --- Task 46 Summary ---")
    for k_val in [1, 2, 3]:
        acc = k_accuracies.get(k_val, 0)
        det = k_details.get(k_val, {})
        print(f"  k={k_val}: {acc:.1%} ({det.get('correct', 0)}/{det.get('total', 0)})")
    print(f"  Best k: {summary['best_k']}")
    return summary


# ===================================================================
# Task 47: Retrieval for rare classes (2B, 50 rare samples)
# ===================================================================
def run_task47(server_2b_url: str, db_data: dict) -> dict[str, Any]:
    _print_separator("TASK 47: Retrieval for Rare Classes (2B, 50 rare)")

    sample_ids = _get_rare_class_samples(db_data, n=50)
    print(f"  Selected {len(sample_ids)} rare-class samples")

    gt_dist: dict[str, int] = {}
    for sid in sample_ids:
        gt = db_data[sid]["scene_type_gt"]
        gt_dist[gt] = gt_dist.get(gt, 0) + 1
    print(f"  GT distribution: {gt_dist}")

    # With retrieval
    results_with = _run_retrieval_classification(
        server_url=server_2b_url,
        sample_ids=sample_ids,
        k=3,
        db_data=db_data,
        model_label="2B",
    )

    # Without retrieval (baseline: just the image)
    import requests
    results_without: list[dict[str, Any]] = []

    print("\n  Running baseline (no retrieval)...")
    for i, sid in enumerate(sample_ids):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [baseline] Sample {i+1}/{len(sample_ids)}...")

        try:
            img_path = load_sample_image(sid)
        except Exception:
            results_without.append({"sample_id": sid, "error": "load failed"})
            continue

        gt_scene = db_data[sid]["scene_type_gt"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Classify this driving scene.\n"
                            "Types: nominal, flooded, incident_zone, "
                            "mounted_police, flagger.\n\n"
                            "FINAL_SCENE: "
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_to_base64(img_path)}",
                        },
                    },
                ],
            },
        ]

        try:
            resp = requests.post(
                f"{server_2b_url}/v1/chat/completions",
                json={
                    "model": "2B",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 256,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content", "")
            prediction = parse_prediction(content)
            predicted = prediction.get("scene")
            correct = (predicted == gt_scene) if predicted else False
        except Exception:
            predicted = None
            correct = False

        results_without.append({
            "sample_id": sid,
            "gt_scene": gt_scene,
            "predicted_scene": predicted,
            "correct": correct,
        })

    # Compute metrics
    with_correct = sum(1 for r in results_with if r.get("correct"))
    with_total = sum(1 for r in results_with if r.get("predicted_scene"))
    without_correct = sum(1 for r in results_without if r.get("correct"))
    without_total = sum(1 for r in results_without if r.get("predicted_scene"))

    with_acc = with_correct / with_total if with_total > 0 else 0
    without_acc = without_correct / without_total if without_total > 0 else 0

    # Per-class breakdown
    per_class_with: dict[str, dict[str, int]] = {}
    per_class_without: dict[str, dict[str, int]] = {}

    for r in results_with:
        gt = r.get("gt_scene", "unknown")
        if gt not in per_class_with:
            per_class_with[gt] = {"correct": 0, "total": 0}
        if r.get("predicted_scene"):
            per_class_with[gt]["total"] += 1
            if r.get("correct"):
                per_class_with[gt]["correct"] += 1

    for r in results_without:
        gt = r.get("gt_scene", "unknown")
        if gt not in per_class_without:
            per_class_without[gt] = {"correct": 0, "total": 0}
        if r.get("predicted_scene"):
            per_class_without[gt]["total"] += 1
            if r.get("correct"):
                per_class_without[gt]["correct"] += 1

    summary = {
        "task": "task47_rare_class_retrieval",
        "model": "2B",
        "n_samples": len(sample_ids),
        "gt_distribution": gt_dist,
        "with_retrieval_accuracy": round(with_acc, 4),
        "without_retrieval_accuracy": round(without_acc, 4),
        "improvement": round(with_acc - without_acc, 4),
        "per_class_with_retrieval": {
            k: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0
            for k, v in per_class_with.items()
        },
        "per_class_without_retrieval": {
            k: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0
            for k, v in per_class_without.items()
        },
        "samples_with_retrieval": results_with,
        "samples_without_retrieval": results_without,
    }

    print("\n  --- Task 47 Summary ---")
    print(f"  With retrieval: {with_acc:.1%} ({with_correct}/{with_total})")
    print(f"  Without retrieval: {without_acc:.1%} ({without_correct}/{without_total})")
    print(f"  Improvement: {with_acc - without_acc:+.1%}")
    print(f"  Per-class (with): {summary['per_class_with_retrieval']}")
    print(f"  Per-class (without): {summary['per_class_without_retrieval']}")
    return summary


# ===================================================================
# Task 48: Adversarial neighbors (2B, 50 samples)
# ===================================================================
def run_task48(
    server_2b_url: str,
    db_data: dict,
    faiss_metadata: list[dict],
) -> dict[str, Any]:
    _print_separator("TASK 48: Adversarial Neighbors (2B, 50 samples)")

    sample_ids = _get_adversarial_samples(faiss_metadata, db_data, n=50)
    print(f"  Found {len(sample_ids)} adversarial samples (NN disagrees with GT)")

    results = _run_retrieval_classification(
        server_url=server_2b_url,
        sample_ids=sample_ids,
        k=1,  # Only nearest neighbor, which disagrees
        db_data=db_data,
        model_label="2B",
    )

    # How often does model follow the misleading neighbor vs trust its own vision?
    followed_neighbor = 0
    trusted_own = 0
    correct_count = 0
    total_with_pred = 0

    for r in results:
        if not r.get("predicted_scene"):
            continue
        total_with_pred += 1

        gt = r.get("gt_scene")
        predicted = r.get("predicted_scene")
        nn_scenes = r.get("neighbor_scenes", [])
        nn_scene = nn_scenes[0] if nn_scenes else None

        if predicted == gt:
            correct_count += 1
            trusted_own += 1
        elif predicted == nn_scene:
            followed_neighbor += 1
        else:
            # Predicted something else entirely
            pass

    accuracy = correct_count / total_with_pred if total_with_pred > 0 else 0
    follow_rate = followed_neighbor / total_with_pred if total_with_pred > 0 else 0
    trust_rate = trusted_own / total_with_pred if total_with_pred > 0 else 0

    summary = {
        "task": "task48_adversarial_neighbors",
        "model": "2B",
        "n_samples": len(sample_ids),
        "accuracy": round(accuracy, 4),
        "followed_misleading_neighbor": followed_neighbor,
        "followed_rate": round(follow_rate, 4),
        "trusted_own_vision": trusted_own,
        "trust_rate": round(trust_rate, 4),
        "total_with_prediction": total_with_pred,
        "insight": (
            f"Model followed misleading neighbor {follow_rate:.0%} of the time, "
            f"trusted its own vision {trust_rate:.0%} of the time."
        ),
        "samples": results,
    }

    print("\n  --- Task 48 Summary ---")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Followed misleading neighbor: {followed_neighbor}/{total_with_pred} ({follow_rate:.1%})")
    print(f"  Trusted own vision: {trusted_own}/{total_with_pred} ({trust_rate:.1%})")
    return summary


# ===================================================================
# Task 49: Retrieval + zoom combined (2B, 100 samples)
# ===================================================================
def run_task49(server_2b_url: str, db_data: dict) -> dict[str, Any]:
    _print_separator("TASK 49: Retrieval + Zoom Combined (2B, 100 samples)")

    all_sids = list(db_data.keys())
    random.shuffle(all_sids)
    sample_ids = all_sids[:100]

    print(f"  Testing retrieval + zoom on {len(sample_ids)} samples")

    # Combined: retrieval + zoom
    results_combined = _run_retrieval_classification(
        server_url=server_2b_url,
        sample_ids=sample_ids,
        k=3,
        db_data=db_data,
        model_label="2B",
        use_zoom=True,
    )

    # Retrieval only (same samples)
    results_retrieval = _run_retrieval_classification(
        server_url=server_2b_url,
        sample_ids=sample_ids,
        k=3,
        db_data=db_data,
        model_label="2B",
        use_zoom=False,
    )

    combined_correct = sum(1 for r in results_combined if r.get("correct"))
    combined_total = sum(1 for r in results_combined if r.get("predicted_scene"))
    retrieval_correct = sum(1 for r in results_retrieval if r.get("correct"))
    retrieval_total = sum(1 for r in results_retrieval if r.get("predicted_scene"))

    combined_acc = combined_correct / combined_total if combined_total > 0 else 0
    retrieval_acc = retrieval_correct / retrieval_total if retrieval_total > 0 else 0

    summary = {
        "task": "task49_retrieval_plus_zoom",
        "model": "2B",
        "n_samples": len(sample_ids),
        "combined_accuracy": round(combined_acc, 4),
        "retrieval_only_accuracy": round(retrieval_acc, 4),
        "improvement": round(combined_acc - retrieval_acc, 4),
        "combined_correct": combined_correct,
        "retrieval_correct": retrieval_correct,
        "insight": (
            f"Combined (retrieval+zoom): {combined_acc:.1%}, "
            f"Retrieval only: {retrieval_acc:.1%}. "
            f"Improvement: {combined_acc - retrieval_acc:+.1%}"
        ),
        "samples_combined": results_combined,
        "samples_retrieval": results_retrieval,
    }

    print("\n  --- Task 49 Summary ---")
    print(f"  Combined (retrieval+zoom): {combined_acc:.1%}")
    print(f"  Retrieval only: {retrieval_acc:.1%}")
    print(f"  Improvement: {combined_acc - retrieval_acc:+.1%}")
    return summary


# ===================================================================
# Task 50: Token cost analysis
# ===================================================================
def run_task50(server_2b_url: str, db_data: dict) -> dict[str, Any]:
    _print_separator("TASK 50: Token Cost Analysis")

    import requests

    # Use 10 samples for cost analysis
    all_sids = list(db_data.keys())
    random.shuffle(all_sids)
    sample_ids = all_sids[:10]

    print(f"  Measuring token costs on {len(sample_ids)} samples")

    cost_data: dict[int, list[dict[str, Any]]] = {}

    for k_val in [1, 2, 3]:
        print(f"\n  --- k={k_val} ---")
        cost_data[k_val] = []

        for i, sid in enumerate(sample_ids):
            print(f"  [k={k_val}] Sample {i+1}/{len(sample_ids)}...")

            try:
                img_path = load_sample_image(sid)
            except Exception:
                continue

            # Measure embedding time
            t_emb_start = time.monotonic()
            context_parts, retrieval_info = _build_retrieval_prompt_with_neighbors(
                img_path, k=k_val, exclude_self_idx=sid,
            )
            t_emb_end = time.monotonic()
            embedding_search_ms = (t_emb_end - t_emb_start) * 1000

            if "error" in retrieval_info:
                continue

            # Build messages and measure token costs
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            user_content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        "Classify this driving scene with the similar scenes shown.\n"
                        "Types: nominal, flooded, incident_zone, mounted_police, flagger.\n"
                        "FINAL_SCENE: "
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_to_base64(img_path)}",
                    },
                },
            ]
            user_content.extend(context_parts)
            messages.append({"role": "user", "content": user_content})

            t_gen_start = time.monotonic()
            try:
                resp = requests.post(
                    f"{server_2b_url}/v1/chat/completions",
                    json={
                        "model": "2B",
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": 256,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                usage = result.get("usage", {})
            except Exception:
                usage = {}
            t_gen_end = time.monotonic()
            generation_ms = (t_gen_end - t_gen_start) * 1000

            cost_data[k_val].append({
                "sample_id": sid,
                "k": k_val,
                "embedding_search_ms": round(embedding_search_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "faiss_search_ms": retrieval_info.get("search_time_ms", 0),
            })

    # Aggregate per k
    k_summaries: dict[int, dict[str, Any]] = {}
    for k_val, entries in cost_data.items():
        if not entries:
            k_summaries[k_val] = {"error": "no data"}
            continue

        avg_prompt = np.mean([e["prompt_tokens"] for e in entries])
        avg_completion = np.mean([e["completion_tokens"] for e in entries])
        avg_total = np.mean([e["total_tokens"] for e in entries])
        avg_emb = np.mean([e["embedding_search_ms"] for e in entries])
        avg_gen = np.mean([e["generation_ms"] for e in entries])
        avg_faiss = np.mean([e["faiss_search_ms"] for e in entries])

        k_summaries[k_val] = {
            "avg_prompt_tokens": round(float(avg_prompt), 1),
            "avg_completion_tokens": round(float(avg_completion), 1),
            "avg_total_tokens": round(float(avg_total), 1),
            "avg_embedding_search_ms": round(float(avg_emb), 1),
            "avg_generation_ms": round(float(avg_gen), 1),
            "avg_faiss_search_ms": round(float(avg_faiss), 1),
            "n_samples": len(entries),
        }

    summary = {
        "task": "task50_token_cost_analysis",
        "model": "2B",
        "per_k_costs": k_summaries,
        "raw_data": {str(k): v for k, v in cost_data.items()},
        "insight": (
            f"k=1: ~{k_summaries.get(1, {}).get('avg_total_tokens', 'N/A')} tokens, "
            f"k=2: ~{k_summaries.get(2, {}).get('avg_total_tokens', 'N/A')} tokens, "
            f"k=3: ~{k_summaries.get(3, {}).get('avg_total_tokens', 'N/A')} tokens"
        ),
    }

    print("\n  --- Task 50 Summary ---")
    for k_val in [1, 2, 3]:
        ks = k_summaries.get(k_val, {})
        print(
            f"  k={k_val}: "
            f"avg_prompt={ks.get('avg_prompt_tokens', 'N/A')}, "
            f"avg_total={ks.get('avg_total_tokens', 'N/A')}, "
            f"avg_emb_ms={ks.get('avg_embedding_search_ms', 'N/A')}, "
            f"avg_gen_ms={ks.get('avg_generation_ms', 'N/A')}"
        )
    return summary


# ===================================================================
# Save scene_metadata.json
# ===================================================================
def _save_scene_metadata(faiss_metadata: list[dict]) -> None:
    """Save metadata to scene_metadata.json for external use."""
    entries = []
    for m in faiss_metadata:
        entries.append({
            "sample_id": m["dataset_index"],
            "scene_type_gt": m["scene_type_gt"],
            "long_action_gt": m["long_action_gt"],
            "lat_action_gt": m["lat_action_gt"],
        })
    with open(SCENE_METADATA_PATH, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Saved scene metadata to {SCENE_METADATA_PATH}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-server", action="store_true",
        help="Skip server start/stop (use pre-started servers)",
    )
    parser.add_argument(
        "--tasks", type=str, default="43,44,45,46,47,48,49,50",
        help="Comma-separated list of tasks to run",
    )
    args = parser.parse_args()

    tasks_to_run = set(int(t) for t in args.tasks.split(","))

    start_time = time.time()

    print("=" * 70)
    print("  Phase 2 Similar Scene Retrieval Deep Dive -- Tasks 43-50")
    print("=" * 70)

    # Verify FAISS index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"\nERROR: FAISS index not found at {FAISS_INDEX_PATH}")
        print("Run: python tool_calling_experiment/visual_tools.py build-index")
        sys.exit(1)
    if not os.path.exists(INDEX_METADATA_PATH):
        print(f"\nERROR: Index metadata not found at {INDEX_METADATA_PATH}")
        sys.exit(1)

    print(f"\nFAISS index found: {FAISS_INDEX_PATH}")

    # Load DB data
    print("Loading self-consistency DB...")
    db_data = _load_db_data()
    print(f"  Loaded {len(db_data)} samples")

    # Load FAISS metadata
    print("Loading FAISS metadata...")
    faiss_metadata = _load_faiss_metadata()
    print(f"  Loaded {len(faiss_metadata)} metadata entries")

    # Save scene_metadata.json
    _save_scene_metadata(faiss_metadata)

    url_2b = f"http://localhost:{PORT_2B}"
    url_8b = f"http://localhost:{PORT_8B}"

    all_results: dict[str, Any] = {
        "experiment": "phase2_retrieval_tasks43to50",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {"2B": MODEL_2B, "8B": MODEL_8B},
        "faiss_index_path": FAISS_INDEX_PATH,
        "total_indexed": len(faiss_metadata),
    }

    # Track task43 samples for task44
    task43_samples: list[int] = []

    # --- Phase A: 2B tasks (single GPU) ---
    _2b_tasks = {43, 45, 46, 47, 48, 49, 50} & tasks_to_run
    if _2b_tasks:
        server_2b = None
        if not args.no_server:
            print("\nStarting 2B server...")
            server_2b = VLLMServer(
                model_path=MODEL_2B,
                port=PORT_2B,
                gpu_id=GPU_2B,
                max_model_len=8192,
                gpu_memory_utilization=0.8,
                enable_tools=True,
            )
            print(f"  Starting 2B on GPU {GPU_2B}, port {PORT_2B}...")
            server_2b.start(timeout=600)
        else:
            print("\nUsing pre-started 2B server...")

        try:
            if 43 in tasks_to_run:
                t43 = run_task43(url_2b, db_data)
                all_results["task43"] = t43
                task43_samples = [
                    s["sample_id"] for s in t43.get("samples", [])
                    if "sample_id" in s
                ]

            if 45 in tasks_to_run:
                all_results["task45"] = run_task45(url_2b, db_data)

            if 46 in tasks_to_run:
                all_results["task46"] = run_task46(url_2b, db_data)

            if 47 in tasks_to_run:
                all_results["task47"] = run_task47(url_2b, db_data)

            if 48 in tasks_to_run:
                all_results["task48"] = run_task48(
                    url_2b, db_data, faiss_metadata,
                )

            if 49 in tasks_to_run:
                all_results["task49"] = run_task49(url_2b, db_data)

            if 50 in tasks_to_run:
                all_results["task50"] = run_task50(url_2b, db_data)

        finally:
            if server_2b is not None:
                print("\nStopping 2B server...")
                server_2b.stop()
                time.sleep(5)

    # --- Phase B: 8B task (single GPU, same or different) ---
    if 44 in tasks_to_run:
        if not task43_samples:
            task43_samples = _get_false_iz_samples(db_data, n=100)

        server_8b = None
        if not args.no_server:
            print("\nStarting 8B server...")
            server_8b = VLLMServer(
                model_path=MODEL_8B,
                port=PORT_8B,
                gpu_id=GPU_8B,
                max_model_len=8192,
                gpu_memory_utilization=0.8,
                enable_tools=True,
            )
            print(f"  Starting 8B on GPU {GPU_8B}, port {PORT_8B}...")
            server_8b.start(timeout=600)
        else:
            print("\nUsing pre-started 8B server...")

        try:
            t44 = run_task44(url_8b, db_data, task43_samples)
            all_results["task44"] = t44
        finally:
            if server_8b is not None:
                print("\nStopping 8B server...")
                server_8b.stop()

    # Save results
    elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = round(elapsed, 1)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Total elapsed: {elapsed/60:.1f} minutes")

    # Final summary
    _print_separator("FINAL SUMMARY -- Phase 2 Tasks 43-50")

    for task_key in ["task43", "task44", "task45", "task46", "task47", "task48", "task49", "task50"]:
        t = all_results.get(task_key, {})
        if not t:
            continue
        task_name = t.get("task", task_key)
        print(f"\n{task_key}: {task_name}")
        if "accuracy" in t:
            print(f"  Accuracy: {t['accuracy']:.1%}")
        if "improvement" in t:
            print(f"  Improvement: {t['improvement']:+.1%}")
        if "insight" in t:
            print(f"  Insight: {t['insight']}")
        if "correlation_insight" in t:
            print(f"  Correlation: {t['correlation_insight']}")


if __name__ == "__main__":
    main()
