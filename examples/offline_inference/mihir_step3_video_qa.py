# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Aggressive benchmark for Step3 Video QA / Inference.

This script demonstrates how to perform inference over video content using the
Step3 architecture. It extracts frames from a video (or generates dummy frames)
and feeds them as a sequence of images to the model.

Usage:
    python step3_video_qa.py --model stepfun-ai/step3-fp8 --video-path path/to/video.mp4 --question "Describe the video."
    python step3_video_qa.py --use-dummy-video --num-frames 16 --question "What is happening?"
    python step3_video_qa.py --use-baby-video --num-frames 8

    To test Hybrid Attention (Step3 Architecture):
    Ensure your model config (config.json) has "use_hybrid_step3_attn": true.
    You can pass a local checkpoint path to --model.

Requirements:
    pip install opencv-python
"""

import argparse
import numpy as np
import os
import sys
import time
from typing import List, Optional
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.video import VideoAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

def get_frames_from_video(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    Extracts frames from a video file uniformly.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required to process video files. Please install it.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Try to count manually or just read until end if streaming
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        total_frames = len(frames)
        cap.release()
        # If num_frames is specified, sample uniformly
        if num_frames < total_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            return [frames[i] for i in indices]
        return frames

    # Random access if possible
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            print(f"Warning: Could not read frame {i}")
    
    cap.release()
    return frames

def generate_dummy_frames(num_frames: int = 8, width: int = 728, height: int = 728) -> List[Image.Image]:
    """Generates random noise frames for benchmarking."""
    print(f"Generating {num_frames} dummy frames ({width}x{height})...")
    frames = []
    for _ in range(num_frames):
        # Generate random noise
        data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frames.append(Image.fromarray(data))
    return frames

def main(args):
    # 1. Prepare Data (Frames)
    if args.use_dummy_video:
        frames = generate_dummy_frames(num_frames=args.num_frames)
    elif args.use_baby_video:
        print("Downloading/Loading 'baby_reading' video asset...")
        video_path = VideoAsset(name="baby_reading").video_path
        print(f"Extracting {args.num_frames} frames from {video_path}...")
        frames = get_frames_from_video(video_path, num_frames=args.num_frames)
    elif args.video_path:
        print(f"Extracting {args.num_frames} frames from {args.video_path}...")
        frames = get_frames_from_video(args.video_path, num_frames=args.num_frames)
    else:
        raise ValueError("Must specify --video-path, --use-dummy-video, or --use-baby-video")

    if not frames:
        raise ValueError("No frames extracted from video.")

    print(f"Prepared {len(frames)} frames for inference.")

    # 2. Prepare Prompt
    # Step3 prompt format for multi-image (video frames treated as images)
    # The <im_patch> placeholder is used for each image.
    # We concatenate them.
    
    # Note: Step3 might have specific system prompts. Using the one from examples.
    prompt = (
        "<｜begin of sentence｜> You are a helpful assistant. <|BOT|>user\n "
        f"{'<im_patch>' * len(frames)}{args.question} <|EOT|><|BOT|"
        ">assistant\n<think>\n"
    )

    # 3. Initialize LLM
    print(f"Initializing LLM with model: {args.model}")
    
    # Aggressive configuration
    engine_args = EngineArgs(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": len(frames)},
        reasoning_parser="step3", # Enable Step3 reasoning parser if applicable
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )

    llm = LLM(**engine_args)

    # 4. Run Inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    print("Starting inference...")
    start_time = time.perf_counter()
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": frames},
        },
        sampling_params=sampling_params,
    )
    end_time = time.perf_counter()
    duration = end_time - start_time

    # 5. Print Results and Metrics
    print("\n" + "=" * 50)
    print("Generated Output (Step3 Video QA):")
    print("=" * 50)
    
    total_input_tokens = 0
    total_output_tokens = 0

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)
        
        # Count tokens
        total_input_tokens += len(o.prompt_token_ids)
        total_output_tokens += len(o.outputs[0].token_ids)
    
    total_tokens = total_input_tokens + total_output_tokens
    tokens_per_sec = total_tokens / duration if duration > 0 else 0
    output_tokens_per_sec = total_output_tokens / duration if duration > 0 else 0

    print("\nBenchmark Metrics:")
    print(f"Time taken: {duration:.2f} s")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Throughput (Total): {tokens_per_sec:.2f} tokens/sec")
    print(f"Throughput (Generation): {output_tokens_per_sec:.2f} tokens/sec")
    print("=" * 50)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Step3 Video QA / Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stepfun-ai/step3-fp8",
        help="Step3 model name or path.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--use-dummy-video",
        action="store_true",
        help="Use generated dummy frames instead of a real video.",
    )
    parser.add_argument(
        "--use-baby-video",
        action="store_true",
        help="Use the sample 'baby_reading' video asset.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe what is happening in this video in detail.",
        help="Question to ask about the video.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample from the video.",
    )
    # Aggressive defaults
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384, # Increased default for "aggressive" benchmark
        help="Maximum number of batched tokens per iteration.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager execution (useful for some aggressive setups or debugging).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0, # Deterministic for benchmarking
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling.",
    )

    args = parser.parse_args()
    main(args)

