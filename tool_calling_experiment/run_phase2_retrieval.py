#!/usr/bin/env python3
# ruff: noqa: E501,E402,I001
# type: ignore[import-not-found]
"""Phase 2: Similar Scene Retrieval Deep Dive. Run independently with: nohup python3 -u run_phase2_retrieval.py &

Tasks:
    43: Retrieval-augmented classification (2B, 100 false-IZ)
    44: Same with 8B (100 false-IZ)
    46: k=1 vs k=2 vs k=3 (2B, 100 samples x 3 conditions)
    47: Retrieval for rare classes (2B, 50 rare samples)
    48: Adversarial neighbors (2B, 50 samples w/ disagreeing NN)

Servers: GPU 6 (2B port 8406), GPU 7 (8B port 8407)
Saves to: phase2_retrieval_final.json

Requires FAISS index: tool_calling_experiment/scene_index.faiss
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

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_PARENT = os.path.dirname(_DIR)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _PARENT]

from orchestrator import parse_prediction
from robust_runner import RobustServer
from visual_tools import (
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    find_similar_scenes,
    image_to_base64,
    load_sample_image,
)

DB_PATH = "/workspace/vllm/self_consistency_experiment/self_consistency.db"
RESULTS_PATH = os.path.join(_DIR, "phase2_retrieval_final.json")

# Server config
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_2B = 6
GPU_8B = 7
PORT_2B = 8406
PORT_8B = 8407

SYSTEM_PROMPT = "The image is 504x336 pixels."

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_all(all_results: dict[str, Any]) -> None:
    """Save all results to disk."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [saved to {RESULTS_PATH}]")


def _load_db_data() -> dict[int, dict[str, str]]:
    """Load GT data and predictions from self-consistency DB."""
    data: dict[int, dict[str, str]] = {}
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
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


def _get_false_iz(db_data: dict[int, dict[str, str]], n: int = 100) -> list[int]:
    """Sample IDs where baseline predicted incident_zone but GT is not."""
    candidates = [
        sid for sid, d in db_data.items()
        if d["predicted_scene"] == "incident_zone"
        and d["scene_type_gt"] != "incident_zone"
    ]
    random.shuffle(candidates)
    return candidates[:n]


def _get_rare_samples(db_data: dict[int, dict[str, str]], n: int = 50) -> list[int]:
    """Samples from rare classes."""
    rare = {"flooded", "incident_zone", "flagger", "mounted_police"}
    candidates = [sid for sid, d in db_data.items() if d["scene_type_gt"] in rare]
    random.shuffle(candidates)
    return candidates[:n]


def _build_neighbor_context(
    img_path: str, k: int, exclude_idx: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run FAISS retrieval and build context parts + info."""
    t0 = time.monotonic()
    result = find_similar_scenes(img_path, k=k + 2)
    search_ms = (time.monotonic() - t0) * 1000

    if "error" in result:
        return [], {"error": result["error"], "search_time_ms": search_ms}

    similar = result.get("similar_images", [])
    if exclude_idx is not None:
        similar = [s for s in similar if s["dataset_index"] != exclude_idx]
    similar = similar[:k]

    descs = []
    neighbor_info = []
    context_parts: list[dict[str, Any]] = []

    for i, s in enumerate(similar):
        descs.append(
            f"Similar scene {i + 1}: scene_type={s['ground_truth_scene']}, "
            f"similarity={s['similarity_score']:.3f}"
        )
        neighbor_info.append({
            "dataset_index": s["dataset_index"],
            "ground_truth_scene": s["ground_truth_scene"],
            "similarity_score": s["similarity_score"],
        })
        if s.get("image"):
            b64 = image_to_base64(s["image"])
            context_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    text_part = {
        "type": "text",
        "text": (
            "Similar scenes from a reference database with known labels:\n\n"
            + "\n".join(descs)
        ),
    }

    info = {
        "k_requested": k,
        "k_returned": len(similar),
        "neighbors": neighbor_info,
        "consensus": result.get("consensus", {}),
        "search_time_ms": round(search_ms, 1),
    }
    return [text_part, *context_parts], info


def _run_retrieval_classification(
    server: RobustServer,
    sample_ids: list[int],
    k: int,
    db_data: dict[int, dict[str, str]],
    model_label: str,
) -> list[dict[str, Any]]:
    """Run retrieval-augmented classification on samples."""
    prompt_text = (
        "Classify this driving scene. "
        "I'll show you similar scenes from a reference database with known labels.\n\n"
        "Scene types: nominal, flooded, incident_zone, mounted_police, flagger.\n\n"
        "After analysis, output:\nFINAL_SCENE: <scene_type>"
    )

    results: list[dict[str, Any]] = []

    for i, sid in enumerate(sample_ids):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{model_label}] Sample {i + 1}/{len(sample_ids)} (sid={sid})...")
        if i > 0 and i % 25 == 0:
            server.ensure_healthy()

        try:
            img_path = load_sample_image(sid)
        except Exception as e:
            results.append({"sample_id": sid, "error": f"load failed: {e}"})
            continue

        gt_scene = db_data[sid]["scene_type_gt"]

        context_parts, ret_info = _build_neighbor_context(img_path, k=k, exclude_idx=sid)
        if "error" in ret_info:
            results.append({"sample_id": sid, "gt_scene": gt_scene, "error": ret_info["error"]})
            continue

        # Build messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(img_path)}"}},
        ]
        user_content.extend(context_parts)
        messages.append({"role": "user", "content": user_content})

        t0 = time.monotonic()
        msg = server.chat(messages, max_tokens=512)
        gen_ms = (time.monotonic() - t0) * 1000

        if msg is None:
            results.append({"sample_id": sid, "gt_scene": gt_scene, "error": "chat returned None"})
            continue

        content = msg.get("content", "")
        prediction = parse_prediction(content)
        predicted = prediction.get("scene")
        correct = (predicted == gt_scene) if predicted else False

        neighbor_scenes = [n["ground_truth_scene"] for n in ret_info.get("neighbors", [])]

        results.append({
            "sample_id": sid,
            "gt_scene": gt_scene,
            "predicted_scene": predicted,
            "correct": correct,
            "neighbor_scenes": neighbor_scenes,
            "retrieval_ms": ret_info.get("search_time_ms", 0),
            "generation_ms": round(gen_ms, 1),
        })

    return results


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------
def run_task43(server: RobustServer, db_data: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Task 43: Retrieval-augmented classification (2B, 100 false-IZ)."""
    print("\n=== Task 43: Retrieval classification (2B, 100 false-IZ) ===")
    sample_ids = _get_false_iz(db_data, 100)
    print(f"  Selected {len(sample_ids)} false-IZ samples")

    results = _run_retrieval_classification(server, sample_ids, k=3, db_data=db_data, model_label="2B")

    correct = sum(1 for r in results if r.get("correct"))
    total = sum(1 for r in results if r.get("predicted_scene"))
    accuracy = correct / total if total > 0 else 0
    # Baseline: all predicted incident_zone = 0% correct for false-IZ
    baseline_correct = sum(1 for sid in sample_ids if db_data[sid]["scene_type_gt"] == "incident_zone")
    baseline_acc = baseline_correct / len(sample_ids) if sample_ids else 0

    summary = {
        "total": len(sample_ids), "correct": correct,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "improvement": round(accuracy - baseline_acc, 4),
    }
    print(f"  T43: accuracy={accuracy:.1%}, baseline={baseline_acc:.1%}, improvement={accuracy - baseline_acc:+.1%}")
    return {"summary": summary, "samples": results, "_sample_ids": sample_ids}


def run_task44(server: RobustServer, db_data: dict[int, dict[str, str]], sample_ids: list[int]) -> dict[str, Any]:
    """Task 44: Same as 43 with 8B."""
    print("\n=== Task 44: Retrieval classification (8B, 100 false-IZ) ===")
    print(f"  Using same {len(sample_ids)} samples as Task 43")

    results = _run_retrieval_classification(server, sample_ids, k=3, db_data=db_data, model_label="8B")

    correct = sum(1 for r in results if r.get("correct"))
    total = sum(1 for r in results if r.get("predicted_scene"))
    accuracy = correct / total if total > 0 else 0
    baseline_correct = sum(1 for sid in sample_ids if db_data[sid]["scene_type_gt"] == "incident_zone")
    baseline_acc = baseline_correct / len(sample_ids) if sample_ids else 0

    summary = {
        "total": len(sample_ids), "correct": correct,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "improvement": round(accuracy - baseline_acc, 4),
    }
    print(f"  T44: accuracy={accuracy:.1%}, improvement={accuracy - baseline_acc:+.1%}")
    return {"summary": summary, "samples": results}


def run_task46(server: RobustServer, db_data: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Task 46: k=1 vs k=2 vs k=3."""
    print("\n=== Task 46: k=1 vs k=2 vs k=3 (2B, 100 samples) ===")
    all_sids = list(db_data.keys())
    random.shuffle(all_sids)
    sample_ids = all_sids[:100]

    k_accs: dict[int, float] = {}
    k_details: dict[int, dict[str, Any]] = {}

    for k_val in [1, 2, 3]:
        print(f"\n  --- k={k_val} ---")
        results = _run_retrieval_classification(server, sample_ids, k=k_val, db_data=db_data, model_label="2B")
        correct = sum(1 for r in results if r.get("correct"))
        total = sum(1 for r in results if r.get("predicted_scene"))
        acc = correct / total if total > 0 else 0
        k_accs[k_val] = round(acc, 4)
        k_details[k_val] = {"accuracy": round(acc, 4), "correct": correct, "total": total}

    best_k = max(k_accs, key=lambda x: k_accs[x])
    summary = {
        "total": len(sample_ids),
        "k_accuracies": k_accs,
        "k_details": k_details,
        "best_k": best_k,
    }
    for kv in [1, 2, 3]:
        print(f"  k={kv}: {k_accs[kv]:.1%}")
    print(f"  Best k: {best_k}")
    return {"summary": summary}


def run_task47(server: RobustServer, db_data: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Task 47: Retrieval for rare classes."""
    print("\n=== Task 47: Retrieval for rare classes (2B, 50 rare) ===")
    sample_ids = _get_rare_samples(db_data, 50)
    print(f"  Selected {len(sample_ids)} rare samples")

    gt_dist: dict[str, int] = {}
    for sid in sample_ids:
        gt = db_data[sid]["scene_type_gt"]
        gt_dist[gt] = gt_dist.get(gt, 0) + 1
    print(f"  GT distribution: {gt_dist}")

    # With retrieval
    results_with = _run_retrieval_classification(server, sample_ids, k=3, db_data=db_data, model_label="2B")

    # Without retrieval (baseline)
    print("\n  Running baseline (no retrieval)...")
    results_without: list[dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        if (i + 1) % 10 == 0:
            print(f"  [baseline] {i + 1}/{len(sample_ids)}...")
        try:
            img_path = load_sample_image(sid)
        except Exception:
            results_without.append({"sample_id": sid, "error": "load failed"})
            continue
        gt_scene = db_data[sid]["scene_type_gt"]
        b64 = image_to_base64(img_path)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Classify this driving scene.\nTypes: nominal, flooded, incident_zone, mounted_police, flagger.\n\nFINAL_SCENE: "},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ]
        msg = server.chat(messages, max_tokens=256)
        content = msg.get("content", "") if msg else ""
        pred = parse_prediction(content)
        predicted = pred.get("scene")
        correct = (predicted == gt_scene) if predicted else False
        results_without.append({"sample_id": sid, "gt_scene": gt_scene, "predicted_scene": predicted, "correct": correct})

    with_correct = sum(1 for r in results_with if r.get("correct"))
    with_total = sum(1 for r in results_with if r.get("predicted_scene"))
    without_correct = sum(1 for r in results_without if r.get("correct"))
    without_total = sum(1 for r in results_without if r.get("predicted_scene"))

    with_acc = with_correct / with_total if with_total > 0 else 0
    without_acc = without_correct / without_total if without_total > 0 else 0

    summary = {
        "total": len(sample_ids),
        "gt_distribution": gt_dist,
        "with_retrieval_accuracy": round(with_acc, 4),
        "without_retrieval_accuracy": round(without_acc, 4),
        "improvement": round(with_acc - without_acc, 4),
    }
    print(f"  T47: with={with_acc:.1%}, without={without_acc:.1%}, improvement={with_acc - without_acc:+.1%}")
    return {"summary": summary, "samples_with": results_with, "samples_without": results_without}


def run_task48(server: RobustServer, db_data: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Task 48: Adversarial neighbors (NN disagrees with GT)."""
    print("\n=== Task 48: Adversarial neighbors (2B, 50 samples) ===")

    import faiss

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(INDEX_METADATA_PATH, "rb") as f:
        faiss_meta = pickle.load(f)

    # Build sid -> faiss idx map
    sid_to_fidx: dict[int, int] = {}
    for i, m in enumerate(faiss_meta):
        sid_to_fidx[m["dataset_index"]] = i

    n_total = index.ntotal
    dim = index.d
    all_vecs = np.zeros((n_total, dim), dtype=np.float32)
    for i in range(n_total):
        all_vecs[i] = index.reconstruct(i)

    adversarial: list[int] = []
    for sid, data in db_data.items():
        if sid not in sid_to_fidx:
            continue
        fidx = sid_to_fidx[sid]
        query = all_vecs[fidx:fidx + 1]
        scores, indices = index.search(query, 5)
        for j in range(5):
            nn_idx = int(indices[0][j])
            nn_score = float(scores[0][j])
            if nn_score > 0.999:
                continue
            if 0 <= nn_idx < len(faiss_meta):
                nn_scene = faiss_meta[nn_idx]["scene_type_gt"]
                if nn_scene != data["scene_type_gt"]:
                    adversarial.append(sid)
            break
        if len(adversarial) >= 50:
            break

    sample_ids = adversarial[:50]
    print(f"  Found {len(sample_ids)} adversarial samples")

    results = _run_retrieval_classification(server, sample_ids, k=1, db_data=db_data, model_label="2B")

    followed = 0
    trusted = 0
    correct_count = 0
    total_pred = 0

    for r in results:
        if not r.get("predicted_scene"):
            continue
        total_pred += 1
        gt = r.get("gt_scene")
        predicted = r.get("predicted_scene")
        nn_scenes = r.get("neighbor_scenes", [])
        nn_scene = nn_scenes[0] if nn_scenes else None
        if predicted == gt:
            correct_count += 1
            trusted += 1
        elif predicted == nn_scene:
            followed += 1

    accuracy = correct_count / total_pred if total_pred > 0 else 0
    follow_rate = followed / total_pred if total_pred > 0 else 0
    trust_rate = trusted / total_pred if total_pred > 0 else 0

    summary = {
        "total": len(sample_ids),
        "accuracy": round(accuracy, 4),
        "followed_misleading_neighbor": followed,
        "follow_rate": round(follow_rate, 4),
        "trusted_own_vision": trusted,
        "trust_rate": round(trust_rate, 4),
    }
    print(f"  T48: accuracy={accuracy:.1%}, followed_nn={follow_rate:.1%}, trusted_own={trust_rate:.1%}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Similar Scene Retrieval Deep Dive (Tasks 43, 44, 46, 47, 48)")
    print("=" * 70)

    # Verify FAISS index
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"ERROR: FAISS index not found at {FAISS_INDEX_PATH}")
        print("Run: python tool_calling_experiment/visual_tools.py build-index")
        sys.exit(1)
    print(f"FAISS index found: {FAISS_INDEX_PATH}")

    # Load DB
    print("Loading self-consistency DB...")
    db_data = _load_db_data()
    print(f"  Loaded {len(db_data)} samples")

    all_results: dict[str, Any] = {
        "experiment": "phase2_retrieval_final",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    task43_sample_ids: list[int] = []

    # === 2B Tasks (GPU 6) ===
    SKIP_2B = True
    if not SKIP_2B:
        print("\n--- Starting 2B server ---")
        server_2b = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
        server_2b.start(timeout=1800)

        try:
            # Task 43
            t43 = run_task43(server_2b, db_data)
            all_results["task43"] = {"summary": t43["summary"], "samples": t43["samples"]}
            task43_sample_ids = t43.get("_sample_ids", [])
            _save_all(all_results)

            # Task 46
            t46 = run_task46(server_2b, db_data)
            all_results["task46"] = t46
            _save_all(all_results)

            # Task 47
            t47 = run_task47(server_2b, db_data)
            all_results["task47"] = t47
            _save_all(all_results)

            # Task 48
            t48 = run_task48(server_2b, db_data)
            all_results["task48"] = t48
            _save_all(all_results)

        finally:
            print("\n--- Stopping 2B server ---")
            server_2b.stop()
            time.sleep(5)

    # === 8B Task (GPU 7) ===
    if not task43_sample_ids:
        task43_sample_ids = _get_false_iz(db_data, 100)

    print("\n--- Starting 8B server ---")
    server_8b = RobustServer(MODEL_8B, gpu_id=GPU_8B, port=PORT_8B)
    server_8b.start(timeout=1800)

    try:
        t44 = run_task44(server_8b, db_data, task43_sample_ids)
        all_results["task44"] = t44
        _save_all(all_results)
    finally:
        print("\n--- Stopping 8B server ---")
        server_8b.stop()

    elapsed = time.time() - start_time
    print(f"\n=== ALL TASKS COMPLETE ({elapsed / 60:.1f} min) ===")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
