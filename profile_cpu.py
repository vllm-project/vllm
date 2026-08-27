"""Profile the per-step CPU cost of MiniCPM-RobotTrack (pixels-in path).

Instruments the real, persistent mm processor used in production: times
``_call_hf_processor`` (resize+normalize+hash, i.e. the whole CPU mm step) and
the total ``llm.embed`` wall time, for a rolling 32-frame window with the
feature cache ON (frame_cache_size=64).

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python profile_cpu.py [--frames N]
"""
import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=36)
    args = parser.parse_args()

    from vllm import LLM

    llm = LLM(
        model="/cache/zhanghao/model/MiniCPM-RobotTrack",
        runner="pooling",
        dtype="float32",
        max_model_len=512,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.45,
        trust_remote_code=False,
        hf_overrides={
            "dino_model": "/cache/zhanghao/model/dinov3-vits16-pretrain-lvd1689m",
            "siglip_model": "/cache/zhanghao/model/siglip-so400m-patch14-384",
            "image_size": 384,
            "frame_cache_size": 64,
        },
    )
    renderer = llm.llm_engine.input_processor.renderer
    processor = renderer.mm_processor
    print("processor:", type(processor).__name__)

    # Instrument the real _call_hf_processor + the full `apply` (includes
    # MultiModalKwargsItems.from_hf_inputs + mm hashing) on the persistent
    # processor, so per-step wall time is attributed correctly.
    original = type(processor)._call_hf_processor
    original_apply = type(processor).apply
    info = processor.info
    original_prepare = info.prepare_pixels
    calls = {"hf": 0, "apply": 0, "frames_normalized": 0}
    hf_time = {"ms": 0.0}
    apply_time = {"ms": 0.0}

    def timed_call_hf(*a, **kw):
        t0 = time.perf_counter()
        out = original(*a, **kw)
        hf_time["ms"] += (time.perf_counter() - t0) * 1e3
        calls["hf"] += 1
        return out

    def timed_apply(*a, **kw):
        t0 = time.perf_counter()
        out = original_apply(*a, **kw)
        apply_time["ms"] += (time.perf_counter() - t0) * 1e3
        calls["apply"] += 1
        return out

    def counting_prepare(frames):
        calls["frames_normalized"] += len(frames)
        return original_prepare(frames)

    type(processor)._call_hf_processor = timed_call_hf
    type(processor).apply = timed_apply
    info.prepare_pixels = counting_prepare

    paths = sorted(Path("track-image/0").glob("*.jpg"))[: args.frames]
    window: deque[Image.Image] = deque(maxlen=32)
    step_ms = []
    for path in paths:
        window.append(Image.open(path).convert("RGB"))
        frames = list(window)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mm = {"image": {"frames": [np.asarray(f) for f in frames]}}
        out = llm.embed([{"prompt": "Follow the person.", "multi_modal_data": mm}])
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)

    warm = slice(32, None) if len(step_ms) > 32 else slice(0, None)
    steady = step_ms[warm]
    print(f"\nsteps={len(step_ms)} steady(n={len(steady)}) "
          f"mean={np.mean(steady):.1f}ms p50={np.median(steady):.1f}ms")
    n_hf, n_ap = max(calls["hf"], 1), max(calls["apply"], 1)
    print(f"_call_hf_processor (CPU resize/normalize) mean={hf_time['ms']/n_hf:.1f}ms/step")
    print(f"processor.apply (full mm: +from_hf_inputs +mm-hash) mean={apply_time['ms']/n_ap:.1f}ms/step")
    print(f"  -> mm machinery beyond _call_hf_processor: "
          f"{(apply_time['ms']-hf_time['ms'])/n_ap:.1f}ms/step")
    print(f"prepare_pixels input mean={calls['frames_normalized']/n_hf:.1f} frames/step")
    print(f"-> CPU mm (apply) share of steady step: "
          f"{apply_time['ms']/len(step_ms)/np.mean(steady)*100:.0f}%")


if __name__ == "__main__":
    main()
