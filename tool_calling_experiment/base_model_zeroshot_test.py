#!/usr/bin/env python3
# ruff: noqa: E501
# type: ignore
"""Zero-shot evaluation of base Qwen3-VL models on SceneIQ.

Tests whether the base (non-fine-tuned) Qwen3-VL 2B and 8B
models can classify driving scenes from zero-shot prompting.

Starts vLLM servers, sends 10 diverse samples with images,
and records results.
"""

import base64
import json
import os
import re
import subprocess
import time

import requests
from streaming import StreamingDataset

# ---- Configuration ----
DATASET_PATH = "/workspace/vllm/models/dataset"
OUTPUT_PATH = (
    "/workspace/vllm/tool_calling_experiment"
    "/base_model_zeroshot_results.json"
)

MODELS = {
    "qwen3vl_2b_base": {
        "path": "/fsx/models/Qwen3-VL-2B-Instruct",
        "port": 8301,
        "gpu": "3",
    },
    "qwen3vl_8b_base": {
        "path": "/fsx/models/Qwen3-VL-8B-Instruct",
        "port": 8302,
        "gpu": "4",
    },
}

# 10 diverse test samples from the MDS dataset
# Format: (dataset_index, description)
TEST_SAMPLES = [
    (2, "nominal / nominal_triggers (easy)"),
    (4, "nominal / nominal_triggers (easy)"),
    (0, "nominal / neg_incident_zone (trigger)"),
    (13, "nominal / neg_incident_zone (trigger)"),
    (1, "flagger / flagger (stop)"),
    (8, "flagger / flagger (proceed)"),
    (59, "flooded / flooded (slowdown)"),
    (60, "flooded / flooded (slowdown)"),
    (49, "incident_zone / accident (stop)"),
    (61, "mounted_police / mounted_police"),
]

# Zero-shot prompt for base model
ZEROSHOT_PROMPT = (
    "You are analyzing a dashcam image from an autonomous"
    " vehicle. Classify the driving scene and predict the"
    " appropriate action.\n"
    "\n"
    "Scene types:\n"
    "- nominal: Normal driving conditions with no obstacles"
    " or hazards requiring action\n"
    "- flooded: Water on the road surface requiring slowdown"
    " or stop\n"
    "- incident_zone: Active incident (crash, emergency"
    " vehicles, road closure) requiring lane change or stop\n"
    "- mounted_police: Police on horseback, treat as normal"
    " driving\n"
    "- flagger: Human traffic controller directing vehicles\n"
    "\n"
    "Actions:\n"
    "- Longitudinal: null (maintain speed), stop, slowdown,"
    " proceed\n"
    "- Lateral: null (stay in lane), lc_left (lane change"
    " left), lc_right (lane change right)\n"
    "\n"
    "Important: nominal is by far the most common scene type"
    " (~78% of cases). incident_zone is rare (~3.7%). Be"
    " cautious about predicting incident_zone -- most scenes"
    " with traffic cones or barriers are actually nominal.\n"
    "\n"
    "Respond in this exact format (3 lines, no other text):\n"
    "Scene: <scene_type>\n"
    "Long_action: <action>\n"
    "Lat_action: <action>"
)

VALID_SCENES = {
    "nominal",
    "flooded",
    "incident_zone",
    "mounted_police",
    "flagger",
}
VALID_LONG = {"null", "stop", "slowdown", "proceed"}
VALID_LAT = {"null", "lc_left", "lc_right"}


def parse_response(text):
    """Parse model response into structured predictions."""
    result = {
        "raw_text": text,
        "predicted_scene": None,
        "predicted_long_action": None,
        "predicted_lat_action": None,
        "format_valid": False,
    }

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        m = re.match(
            r"Scene:\s*(\S+)", line, re.IGNORECASE
        )
        if m:
            result["predicted_scene"] = (
                m.group(1).lower().strip()
            )
        m = re.match(
            r"Long_action:\s*(\S+)", line, re.IGNORECASE
        )
        if m:
            result["predicted_long_action"] = (
                m.group(1).lower().strip()
            )
        m = re.match(
            r"Lat_action:\s*(\S+)", line, re.IGNORECASE
        )
        if m:
            result["predicted_lat_action"] = (
                m.group(1).lower().strip()
            )

    result["scene_valid"] = (
        result["predicted_scene"] in VALID_SCENES
    )
    result["long_valid"] = (
        result["predicted_long_action"] in VALID_LONG
    )
    result["lat_valid"] = (
        result["predicted_lat_action"] in VALID_LAT
    )
    result["format_valid"] = all([
        result["scene_valid"],
        result["long_valid"],
        result["lat_valid"],
    ])

    return result


def start_server(model_name, config):
    """Start a vLLM server for a model."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    # Remove /workspace/vllm from PYTHONPATH so the
    # installed vllm package is used instead of the
    # local source tree (which lacks compiled _C ext).
    pp = env.get("PYTHONPATH", "")
    parts = [
        p
        for p in pp.split(":")
        if p and "/workspace/vllm" not in p
    ]
    env["PYTHONPATH"] = ":".join(parts)

    vllm_bin = "/home/mketkar/.local/bin/vllm"
    cmd = [
        vllm_bin,
        "serve",
        config["path"],
        "--trust-remote-code",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--port",
        str(config["port"]),
        "--gpu-memory-utilization",
        "0.8",
    ]

    log_path = f"/tmp/vllm_{model_name}.log"
    log_file = open(log_path, "w")  # noqa: SIM115
    print(
        f"Starting {model_name} on GPU "
        f"{config['gpu']} port {config['port']}..."
    )
    print(f"  Log: {log_path}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd="/tmp",
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(port, timeout=300):
    """Wait for vLLM server to become ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except (
            requests.ConnectionError,
            requests.Timeout,
        ):
            pass
        time.sleep(5)
    return False


def load_sample(ds, idx):
    """Load a sample, extracting images and metadata."""
    sample = ds[idx]
    meta = sample["metadata"]
    messages = sample["messages"]

    # Extract images as base64
    images = []
    img_keys = [
        "image_0000",
        "image_0001",
        "image_0002",
        "image_0003",
    ]
    for key in img_keys:
        img_bytes = sample.get(key)
        if img_bytes is not None and len(img_bytes) > 0:
            b64 = base64.b64encode(img_bytes).decode(
                "utf-8"
            )
            images.append(
                f"data:image/jpeg;base64,{b64}"
            )

    # Ground truth from metadata
    odd_label = meta.get("odd_label", "unknown")
    fine_class = meta.get("fine_class", "unknown")
    long_action = meta.get("long_action", "null")
    lat_action = meta.get("lat_action", "null")

    # odd_label matches scene_type directly
    scene_type_gt = odd_label

    # Get the fine-tuned model's ground truth response
    asst_content = messages[1]["content"]
    ft_response = ""
    if isinstance(asst_content, list):
        for c in asst_content:
            if "text" in c:
                ft_response += c["text"]
    elif isinstance(asst_content, str):
        ft_response = asst_content

    # Parse fine-tuned model's ODD token
    ft_scene = None
    odd_map = {
        "<|odd_nominal|>": "nominal",
        "<|odd_flood|>": "flooded",
        "<|odd_incident|>": "incident_zone",
        "<|odd_policehorse|>": "mounted_police",
        "<|odd_flagger|>": "flagger",
    }
    for token, scene in odd_map.items():
        if token in ft_response:
            ft_scene = scene
            break

    return {
        "dataset_index": idx,
        "chum_uri": meta.get("chum_uri", ""),
        "scene_type_gt": scene_type_gt,
        "fine_class": fine_class,
        "long_action_gt": long_action,
        "lat_action_gt": lat_action,
        "ft_predicted_scene": ft_scene,
        "ft_response_snippet": ft_response[:100],
        "images": images,
    }


def get_model_id(port):
    """Get the model ID from the vLLM server."""
    url = f"http://localhost:{port}/v1/models"
    resp = requests.get(url, timeout=10)
    return resp.json()["data"][0]["id"]


def query_model(port, model_id, images, prompt):
    """Send a multimodal query to vLLM server."""
    content = []
    for img_url in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": img_url},
        })
    content.append({
        "type": "text",
        "text": prompt,
    })

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    url = f"http://localhost:{port}/v1/chat/completions"
    try:
        resp = requests.post(
            url, json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def run_test(port, model_name, model_id, samples):
    """Run all test samples through a model."""
    results = []
    for i, sample_info in enumerate(samples):
        desc = TEST_SAMPLES[i][1]
        idx = sample_info["dataset_index"]
        gt_s = sample_info["scene_type_gt"]
        gt_lo = sample_info["long_action_gt"]
        gt_la = sample_info["lat_action_gt"]
        print(f"  [{i+1}/10] idx={idx} ({desc})")
        print(
            f"    GT: scene={gt_s}, "
            f"long={gt_lo}, lat={gt_la}"
        )

        raw_response = query_model(
            port,
            model_id,
            sample_info["images"],
            ZEROSHOT_PROMPT,
        )
        parsed = parse_response(raw_response)

        scene_ok = (
            parsed["predicted_scene"]
            == sample_info["scene_type_gt"]
        )
        result = {
            "dataset_index": idx,
            "description": desc,
            "chum_uri": sample_info["chum_uri"],
            "scene_type_gt": gt_s,
            "fine_class": sample_info["fine_class"],
            "long_action_gt": gt_lo,
            "lat_action_gt": gt_la,
            "ft_predicted_scene": (
                sample_info["ft_predicted_scene"]
            ),
            "model": model_name,
            "raw_response": raw_response,
            "predicted_scene": parsed["predicted_scene"],
            "predicted_long_action": (
                parsed["predicted_long_action"]
            ),
            "predicted_lat_action": (
                parsed["predicted_lat_action"]
            ),
            "scene_correct": scene_ok,
            "scene_valid": parsed["scene_valid"],
            "format_valid": parsed["format_valid"],
        }
        results.append(result)

        tag = "CORRECT" if scene_ok else "WRONG"
        ps = parsed["predicted_scene"]
        pl = parsed["predicted_long_action"]
        pa = parsed["predicted_lat_action"]
        print(
            f"    Pred: scene={ps} [{tag}], "
            f"long={pl}, lat={pa}"
        )
        if not parsed["format_valid"]:
            print(
                f"    FORMAT ISSUE - raw: "
                f"{raw_response[:150]}"
            )

    return results


def compute_summary(results_2b, results_8b):
    """Compute summary statistics."""
    summary = {}
    pairs = [
        ("qwen3vl_2b_base", results_2b),
        ("qwen3vl_8b_base", results_8b),
    ]
    for model_name, results in pairs:
        n = len(results)
        if n == 0:
            summary[model_name] = {}
            continue
        scene_correct = sum(
            1 for r in results if r["scene_correct"]
        )
        scene_valid = sum(
            1 for r in results if r["scene_valid"]
        )
        format_valid = sum(
            1 for r in results if r["format_valid"]
        )

        by_gt = {}
        for r in results:
            gt = r["scene_type_gt"]
            if gt not in by_gt:
                by_gt[gt] = {"total": 0, "correct": 0}
            by_gt[gt]["total"] += 1
            if r["scene_correct"]:
                by_gt[gt]["correct"] += 1

        pred_dist = {}
        for r in results:
            p = r["predicted_scene"] or "INVALID"
            pred_dist[p] = pred_dist.get(p, 0) + 1

        hallucinated = [
            r["predicted_scene"]
            for r in results
            if (
                r["predicted_scene"]
                and r["predicted_scene"] not in VALID_SCENES
            )
        ]

        iz_pred = sum(
            1
            for r in results
            if r["predicted_scene"] == "incident_zone"
        )
        iz_actual = sum(
            1
            for r in results
            if r["scene_type_gt"] == "incident_zone"
        )

        pct = scene_correct / n
        summary[model_name] = {
            "overall_accuracy": (
                f"{scene_correct}/{n} = {pct:.1%}"
            ),
            "scene_valid_rate": (
                f"{scene_valid}/{n} = "
                f"{scene_valid / n:.1%}"
            ),
            "format_valid_rate": (
                f"{format_valid}/{n} = "
                f"{format_valid / n:.1%}"
            ),
            "per_class_accuracy": {
                gt: f"{v['correct']}/{v['total']}"
                for gt, v in sorted(by_gt.items())
            },
            "predicted_distribution": pred_dist,
            "hallucinated_scenes": hallucinated,
            "incident_zone_predictions": iz_pred,
            "incident_zone_actual": iz_actual,
        }

    return summary


def print_summary(summary, samples, comparison):
    """Print the final summary report."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    models = ["qwen3vl_2b_base", "qwen3vl_8b_base"]
    for model_name in models:
        s = summary.get(model_name, {})
        if not s:
            print(
                f"\n{model_name}: NO RESULTS "
                "(server may have failed)"
            )
            continue
        print(f"\n--- {model_name} ---")
        print(f"  Overall accuracy: {s['overall_accuracy']}")
        print(f"  Valid scene rate: {s['scene_valid_rate']}")
        print(f"  Format valid:     {s['format_valid_rate']}")
        print(f"  Per-class: {s['per_class_accuracy']}")
        pred = s["predicted_distribution"]
        print(f"  Predicted dist:   {pred}")
        if s["hallucinated_scenes"]:
            h = s["hallucinated_scenes"]
            print(f"  HALLUCINATED:     {h}")
        iz_p = s["incident_zone_predictions"]
        iz_a = s["incident_zone_actual"]
        print(
            f"  incident_zone predicted: {iz_p} "
            f"(actual: {iz_a})"
        )

    print("\n--- Comparison Table ---")
    hdr = (
        f"{'Idx':>4} {'GT Scene':<16} "
        f"{'Fine Class':<25} {'FT Pred':<16} "
        f"{'2B Pred':<16} {'8B Pred':<16}"
    )
    print(hdr)
    print("-" * 100)
    for row in comparison:
        gt = row["ground_truth_scene"]
        fc = row["fine_class"]
        ft = row.get("ft_predicted_scene", "?")
        b2 = row.get("base_2b_predicted_scene", "?")
        b8 = row.get("base_8b_predicted_scene", "?")
        b2m = " OK" if row.get("base_2b_scene_correct") else " XX"
        b8m = " OK" if row.get("base_8b_scene_correct") else " XX"
        b2_str = (b2 or "?")
        b8_str = (b8 or "?")
        print(
            f"{row['dataset_index']:>4} {gt:<16} "
            f"{fc:<25} {ft:<16} "
            f"{b2_str:<13}{b2m} {b8_str:<13}{b8m}"
        )

    print("\n--- Key Questions ---")
    for model_name in models:
        s = summary.get(model_name, {})
        if not s:
            continue
        acc_str = s["overall_accuracy"]
        parts = acc_str.split("=")
        acc_pct = parts[1].strip() if len(parts) == 2 else "?"
        print(f"\n  {model_name}:")
        print(
            f"    Accuracy: {acc_pct} "
            "(random = 20% for 5 classes)"
        )
        vr = s["scene_valid_rate"]
        print(f"    Produces valid scene types: {vr}")
        h = s["hallucinated_scenes"]
        print(f"    Hallucinated scenes: {len(h)} ({h})")

    print("\nDone.")


def main():
    """Run the full zero-shot evaluation."""
    print("=" * 70)
    print("BASE MODEL ZERO-SHOT SCENE CLASSIFICATION TEST")
    print("=" * 70)

    # ---- Load dataset ----
    print("\nLoading MDS dataset...")
    ds = StreamingDataset(
        local=DATASET_PATH, shuffle=False
    )
    print(f"Dataset size: {len(ds)}")

    # ---- Load test samples ----
    print("\nLoading 10 test samples...")
    samples = []
    for idx, desc in TEST_SAMPLES:
        sample = load_sample(ds, idx)
        samples.append(sample)
        gt = sample["scene_type_gt"]
        fc = sample["fine_class"]
        ft = sample["ft_predicted_scene"]
        ni = len(sample["images"])
        print(
            f"  idx={idx}: gt={gt}/{fc}, "
            f"ft_pred={ft}, images={ni}"
        )

    # ---- Run models SEQUENTIALLY ----
    # Each model: start server, wait, test, kill.
    # This avoids GPU conflicts and memory issues.
    all_results = {}
    for model_name, config in MODELS.items():
        port = config["port"]
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")

        proc = start_server(model_name, config)
        print(
            f"  Waiting for {model_name} "
            f"on port {port} (up to 15 min)..."
        )
        if not wait_for_server(port, timeout=900):
            print(
                f"  ERROR: {model_name} failed."
                f" Check /tmp/vllm_{model_name}.log"
            )
            log_path = f"/tmp/vllm_{model_name}.log"
            if os.path.exists(log_path):
                with open(log_path) as f:
                    tail = f.readlines()
                    for ln in tail[-20:]:
                        print(f"    {ln.rstrip()}")
            proc.terminate()
            all_results[model_name] = []
            continue

        print(f"  {model_name} is READY")

        # Get model ID for API calls
        try:
            mid = get_model_id(port)
            print(f"  Model ID = {mid}")
        except Exception:
            mid = config["path"]
            print(f"  Using path as model ID: {mid}")

        print(f"\nRunning {model_name} tests...")
        results = run_test(
            port, model_name, mid, samples
        )
        all_results[model_name] = results

        # Kill this server before starting the next
        print(f"\nStopping {model_name}...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Also kill any leftover Ray/Engine processes
        subprocess.run(
            ["pkill", "-9", "-f", f"port.*{port}"],
            capture_output=True,
        )
        subprocess.run(
            ["pkill", "-9", "-f", "EngineCore"],
            capture_output=True,
        )
        print(f"  {model_name} stopped.")
        # Wait for GPU memory to free
        time.sleep(10)

    # ---- Compute summary ----
    r2b = all_results.get("qwen3vl_2b_base", [])
    r8b = all_results.get("qwen3vl_8b_base", [])
    summary = compute_summary(r2b, r8b)

    # ---- Build comparison table ----
    comparison = []
    for i in range(len(TEST_SAMPLES)):
        row = {
            "dataset_index": TEST_SAMPLES[i][0],
            "description": TEST_SAMPLES[i][1],
            "ground_truth_scene": (
                samples[i]["scene_type_gt"]
            ),
            "fine_class": samples[i]["fine_class"],
            "long_action_gt": (
                samples[i]["long_action_gt"]
            ),
            "lat_action_gt": samples[i]["lat_action_gt"],
            "ft_predicted_scene": (
                samples[i]["ft_predicted_scene"]
            ),
        }
        if i < len(r2b):
            row["base_2b_predicted_scene"] = (
                r2b[i]["predicted_scene"]
            )
            row["base_2b_scene_correct"] = (
                r2b[i]["scene_correct"]
            )
            row["base_2b_raw_response"] = (
                r2b[i]["raw_response"]
            )
        if i < len(r8b):
            row["base_8b_predicted_scene"] = (
                r8b[i]["predicted_scene"]
            )
            row["base_8b_scene_correct"] = (
                r8b[i]["scene_correct"]
            )
            row["base_8b_raw_response"] = (
                r8b[i]["raw_response"]
            )
        comparison.append(row)

    # ---- Output ----
    output = {
        "experiment": (
            "base_model_zeroshot_scene_classification"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_tested": {
            "qwen3vl_2b_base": (
                MODELS["qwen3vl_2b_base"]["path"]
            ),
            "qwen3vl_8b_base": (
                MODELS["qwen3vl_8b_base"]["path"]
            ),
        },
        "num_samples": len(TEST_SAMPLES),
        "prompt_used": ZEROSHOT_PROMPT,
        "summary": summary,
        "comparison_table": comparison,
        "raw_results": {
            "qwen3vl_2b_base": r2b,
            "qwen3vl_8b_base": r8b,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")

    print_summary(summary, samples, comparison)


if __name__ == "__main__":
    main()
