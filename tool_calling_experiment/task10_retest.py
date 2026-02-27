#!/usr/bin/env python3
# ruff: noqa: E501,E402
# type: ignore[import-not-found]
"""Task 10 Retest: Road geometry comprehension post-JPEG fix.

Tests that the model receives and references geometry analysis data
when analyze_road_geometry now outputs JPEG (not PNG).

10 samples, base 2B model, analyze_road_geometry tool only.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure sibling modules are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from orchestrator import ToolCallingOrchestrator  # noqa: E402, I001
from server_utils import VLLMServer  # noqa: E402
from visual_tools import (  # noqa: E402
    TOOL_ROAD_GEOMETRY,
    analyze_road_geometry,
    load_sample_image,
    load_sample_metadata,
)

RESULTS_PATH = os.path.join(_DIR, "task10_retest_results.json")

BASE_2B_MODEL = "/fsx/models/Qwen3-VL-2B-Instruct"
GPU_ID = 7
PORT = 8340

SYSTEM_PROMPT = (
    "You are a driving scene analyst. "
    "The image is 504x336 pixels. "
    "You have access to a road geometry analysis tool. "
    "Use it to understand the road structure before classifying the scene."
)

USER_PROMPT = (
    "Analyze this driving scene. First, use the analyze_road_geometry tool to "
    "understand the road structure. Then based on what you see in the image AND "
    "the geometry analysis, classify the scene.\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)


def main():
    print("=" * 70)
    print("TASK 10 RETEST: Road geometry comprehension (JPEG fix)")
    print("=" * 70)
    print(f"  Model: {BASE_2B_MODEL}")
    print(f"  GPU: {GPU_ID}, Port: {PORT}")
    print("  Samples: 10")
    print()

    # Start server
    server = VLLMServer(
        model_path=BASE_2B_MODEL,
        port=PORT,
        gpu_id=GPU_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        enable_tools=True,
    )

    try:
        print("Starting vLLM server...")
        server.start(timeout=600)
        print(f"Server healthy on port {PORT}")

        # Build orchestrator with road geometry tool only
        orch = ToolCallingOrchestrator(
            server_url=f"http://localhost:{PORT}",
            tools={"analyze_road_geometry": analyze_road_geometry},
            tool_definitions=[TOOL_ROAD_GEOMETRY],
            max_tool_rounds=3,
            temperature=0,
            max_tokens=1024,
        )

        # Run 10 samples
        results = []
        sample_indices = list(range(10))

        for i, sid in enumerate(sample_indices):
            print(f"\n  [{i+1}/10] Sample {sid}...")

            try:
                img_path = load_sample_image(sid)
            except Exception as e:
                print(f"    ERROR loading image: {e}")
                results.append({"sample_id": sid, "error": f"Image load failed: {e}"})
                continue

            try:
                meta = load_sample_metadata(sid)
                fine_class = meta.get("fine_class", "unknown")
            except Exception:
                fine_class = "unknown"

            # Check server health before each request
            if not server.is_healthy():
                print("    Server unhealthy, restarting...")
                server.restart(timeout=600)

            try:
                r = orch.run(
                    image_path=img_path,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=USER_PROMPT,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"    ERROR in orchestrator: {e}")
                results.append({"sample_id": sid, "error": str(e)})
                continue

            # Extract key info
            final_text = r.get("final_text", "")
            final_pred = r.get("final_prediction", {})
            tool_calls = r.get("tool_calls", [])
            num_tools = r.get("num_tool_calls", 0)

            # Check if model called analyze_road_geometry
            called_geometry = any(
                tc.get("tool_name") == "analyze_road_geometry"
                for tc in tool_calls
            )

            # Check if model referenced geometry data in its response
            reasoning_text = r.get("reasoning_text", "")
            references_geometry = any(
                kw in reasoning_text.lower()
                for kw in [
                    "lane", "curvature", "vanishing", "drivable",
                    "road_curvature", "num_lanes", "boundary",
                    "straight", "gentle", "sharp",
                ]
            )

            # Check if annotated image was returned to the model
            geometry_result_has_image = any(
                tc.get("result_has_image", False)
                for tc in tool_calls
                if tc.get("tool_name") == "analyze_road_geometry"
            )

            # Get geometry metadata from tool call result
            geometry_metadata = None
            for tc in tool_calls:
                if tc.get("tool_name") == "analyze_road_geometry":
                    geometry_metadata = tc.get("result_metadata", {})
                    break

            entry = {
                "sample_id": sid,
                "fine_class": fine_class,
                "called_geometry": called_geometry,
                "references_geometry": references_geometry,
                "geometry_result_has_image": geometry_result_has_image,
                "geometry_metadata": geometry_metadata,
                "num_tool_calls": num_tools,
                "final_prediction": final_pred,
                "final_text_snippet": final_text[:500],
                "reasoning_snippet": reasoning_text[:500],
                "latency_ms": r.get("latency_ms"),
                "error": r.get("error"),
            }
            results.append(entry)

            scene = final_pred.get("scene", "?") if final_pred else "?"
            print(f"    called_geometry={called_geometry}, "
                  f"references={references_geometry}, "
                  f"has_image={geometry_result_has_image}, "
                  f"scene={scene}")
            if geometry_metadata:
                print(f"    curvature={geometry_metadata.get('road_curvature', '?')}, "
                      f"lanes={geometry_metadata.get('num_lanes_detected', '?')}, "
                      f"lines={geometry_metadata.get('total_lines_detected', '?')}")

        # Aggregate summary
        n = len(results)
        n_called = sum(1 for r in results if r.get("called_geometry"))
        n_references = sum(1 for r in results if r.get("references_geometry"))
        n_has_image = sum(1 for r in results if r.get("geometry_result_has_image"))
        n_errors = sum(1 for r in results if r.get("error"))

        summary = {
            "task": "task10_retest_geometry_jpeg",
            "total_samples": n,
            "called_geometry_tool": n_called,
            "call_rate": round(n_called / n, 3) if n > 0 else 0,
            "references_geometry_data": n_references,
            "reference_rate": round(n_references / n, 3) if n > 0 else 0,
            "received_annotated_image": n_has_image,
            "image_rate": round(n_has_image / n, 3) if n > 0 else 0,
            "errors": n_errors,
            "verdict": (
                "PASS: Model calls geometry tool and references its data"
                if n_called >= 7 and n_references >= 5
                else "PARTIAL: Model calls tool but may not reference data"
                if n_called >= 5
                else "FAIL: Model rarely calls geometry tool"
            ),
        }

        print(f"\n{'='*70}")
        print("TASK 10 RETEST SUMMARY")
        print(f"{'='*70}")
        print(json.dumps(summary, indent=2))

        # Save results
        output = {
            "experiment": "task10_retest",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": summary,
            "samples": results,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {RESULTS_PATH}")

    finally:
        print("\nStopping server...")
        server.stop()
        print("Server stopped.")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
