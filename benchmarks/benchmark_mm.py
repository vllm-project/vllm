#!/usr/bin/env python3

import argparse
import time
from typing import Optional

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Pillow (PIL) is required for this benchmark. Please install with `pip install pillow`."
    ) from exc

from vllm import LLM, SamplingParams


def create_solid_image(width: int, height: int, rgb: tuple[int, int, int]) -> Image.Image:
    """Create a simple solid-color RGB image using Pillow."""
    return Image.new("RGB", (width, height), rgb)


def load_image(path: Optional[str], fallback_rgb: tuple[int, int, int]) -> Image.Image:
    if path:
        return Image.open(path).convert("RGB")
    return create_solid_image(512, 512, fallback_rgb)


def run_benchmark(
    model: str,
    image1_path: Optional[str],
    image2_path: Optional[str],
    max_tokens: int,
    runs: int,
    warmup: int,
    trust_remote_code: bool,
    max_model_len: Optional[int],
) -> None:
    image_1 = load_image(image1_path, (220, 20, 60))  # crimson
    image_2 = load_image(image2_path, (65, 105, 225))  # royal blue

    llm = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
        **({"max_model_len": max_model_len} if max_model_len is not None else {}),
    )

    sampling_params = SamplingParams(max_tokens=max_tokens)

    payload = [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_2},
        },
    ]

    # Warmup runs (not timed)
    for _ in range(max(0, warmup)):
        _ = llm.generate(payload, sampling_params=sampling_params)

    # Timed runs
    latencies_s: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        outputs = llm.generate(payload, sampling_params=sampling_params)
        t1 = time.perf_counter()
        latencies_s.append(t1 - t0)

    # Print one set of outputs for sanity check
    print("=== Sample outputs ===")
    for i, o in enumerate(outputs):
        print(f"Request {i}: {o.outputs[0].text.strip()}")

    # Report latency stats
    avg = sum(latencies_s) / len(latencies_s) if latencies_s else float("nan")
    p50 = sorted(latencies_s)[len(latencies_s) // 2] if latencies_s else float("nan")
    p95 = sorted(latencies_s)[max(0, int(len(latencies_s) * 0.95) - 1)] if latencies_s else float("nan")

    print("\n=== Benchmark results ===")
    print(f"Model: {model}")
    print(f"Runs: {runs} (warmup: {warmup})")
    print(f"Max tokens: {max_tokens}")
    print(f"Latency (s): avg={avg:.3f}, p50={p50:.3f}, p95={p95:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple end-to-end multimodal (image) benchmark using vLLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Model name or path.",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="Path to first image file (defaults to a synthetic image).",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=None,
        help="Path to second image file (defaults to a synthetic image).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens to generate per request.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (not timed).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to transformers to trust remote code if required by the model.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model length override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        model=args.model,
        image1_path=args.image1,
        image2_path=args.image2,
        max_tokens=args.max_tokens,
        runs=args.runs,
        warmup=args.warmup,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()

