# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline demo for the MiniCPM-RobotTrack vision-language-action policy.

MiniCPM-RobotTrack is served as a vLLM pooling model: a single causal forward
whose last (control-query) token drives a trajectory head that regresses eight
``[x, y, yaw]`` waypoints. The model consumes *precomputed* fused DINOv3+SigLIP
features (1536-dim), so the multi-modal input is a dict of feature tensors under
the ``"image"`` modality rather than raw pixels.

Run:
    python examples/pooling/robottrack_minicpm_offline.py \
        --model openbmb/MiniCPM-RobotTrack

Replace ``build_dummy_visual_features`` with your project's DINOv3+SigLIP
preprocessing to get meaningful trajectories; the random features here only
exercise the serving path.
"""

import argparse

import torch

from vllm import LLM

# Fixed for this checkpoint (see the model's config.json / HF model card).
VISION_FEATURE_DIM = 1536
HISTORY_FRAMES = 31
COARSE_TOKENS_PER_FRAME = 4
FINE_TOKENS_CURRENT_FRAME = 64
NUM_WAYPOINTS = 8
ACTION_DIM = 3


def build_dummy_visual_features(
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build one visual bundle: history (coarse) + current (fine) features.

    In a real deployment these come from the DINOv3+SigLIP fusion pipeline.
    ``*_time_indices`` group tokens by frame; one temporal marker is inserted
    before each maximal run of equal indices.
    """
    generator = torch.Generator().manual_seed(seed)
    num_coarse = HISTORY_FRAMES * COARSE_TOKENS_PER_FRAME
    return {
        "coarse_tokens": torch.randn(
            num_coarse, VISION_FEATURE_DIM, generator=generator
        ),
        "coarse_time_indices": torch.arange(HISTORY_FRAMES).repeat_interleave(
            COARSE_TOKENS_PER_FRAME
        ),
        "fine_tokens": torch.randn(
            FINE_TOKENS_CURRENT_FRAME, VISION_FEATURE_DIM, generator=generator
        ),
        "fine_time_indices": torch.full(
            (FINE_TOKENS_CURRENT_FRAME,), HISTORY_FRAMES, dtype=torch.long
        ),
    }


def main(args: argparse.Namespace) -> None:
    llm = LLM(
        model=args.model,
        runner="pooling",
        dtype=args.dtype,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
    )

    prompts = [
        {
            "prompt": args.instruction,
            "multi_modal_data": {"image": build_dummy_visual_features(seed)},
        }
        for seed in range(args.num_prompts)
    ]

    outputs = llm.embed(prompts)

    for i, output in enumerate(outputs):
        trajectory = torch.tensor(output.outputs.embedding).reshape(
            NUM_WAYPOINTS, ACTION_DIM
        )
        print(f"\n=== request {i}: predicted waypoints [x, y, yaw] ===")
        for step, (x, y, yaw) in enumerate(trajectory.tolist()):
            print(f"  t{step}: x={x:+.4f}  y={y:+.4f}  yaw={yaw:+.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openbmb/MiniCPM-RobotTrack")
    parser.add_argument("--instruction", default="Follow the person in the red shirt.")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

# python examples/pooling/robottrack_minicpm_offline.py --model openbmb/MiniCPM-RobotTrack