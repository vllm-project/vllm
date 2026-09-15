# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-image, per-frame tracking demo for MiniCPM-RobotTrack on vLLM.

The DINOv3 + SigLIP visual encoder runs **inside vLLM** (the in-tree vision
tower); the client only maintains the rolling 32-frame window and ships raw
frames. For each frame the inference chain is:

    frame -> (client deque of <=32 raw frames)
          -> vLLM: resize384 + DINOv3 + SigLIP (fused 1536-dim tokens)
                   -> [31 coarse history + 1 fine current] -> pooling forward
          -> 8 [vx, vy, vyaw] waypoints -> velocity overlay -> output/

RobotTrack predicts a trajectory of waypoint *velocities* in the robot's body
frame (x forward, +y left, +yaw counter-clockwise), already scaled by
``output_scale`` (xy by ``xy_scale``, yaw unscaled). The released benchmark
policy divides one waypoint by ``control_dt`` to get m/s and rad/s, then feeds
that to the controller; this demo mirrors that: it consumes a selectable
``--control-waypoint`` (default 1, matching the release) and divides by
``--control-dt`` before drawing.

Example:
    python examples/pooling/robottrack_minicpm_video.py \
        --model openbmb/MiniCPM-RobotTrack \
        --dino facebook/dinov3-vits16-pretrain-lvd1689m \
        --siglip google/siglip-so400m-patch14-384 \
        --images track-image/0 --output output \
        --instruction "Follow the person."
"""

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from vllm import LLM

WINDOW_FRAMES = 32  # 31 history + 1 current
NUM_WAYPOINTS = 8
ACTION_DIM = 3
IMAGE_SIZE = 384


def build_window(frames: deque[Image.Image]) -> dict[str, list[np.ndarray]]:
    """Pack the rolling window (oldest first, current last) as raw frames.

    The tower treats the last frame as the current frame (fine tokens) and the
    earlier frames as coarse history, padding to 31 frames internally, so the
    client just forwards whatever it has buffered.
    """
    return {"frames": [np.asarray(frame.convert("RGB")) for frame in frames]}


def draw_overlay(
    image: Image.Image,
    velocity: torch.Tensor,
    trajectory: torch.Tensor,
    instruction: str,
    frame_idx: int,
    vel_range: float,
) -> Image.Image:
    """Draw the selected waypoint velocity (vx fwd, +vy left) as a BEV arrow.

    ``velocity`` is ``[vx, vy, vyaw]`` in m/s, rad/s (already divided by
    ``control_dt``); ``trajectory`` is the full 8-waypoint velocity trajectory
    for the inset, drawn as a relative walk from the robot origin.
    """
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Main velocity arrow: robot fixed at frame center-bottom, pointing forward.
    cx = image.width // 2
    cy = int(image.height * 0.78)
    arrow_scale = (min(image.width, image.height) * 0.28) / max(vel_range, 1e-6)
    vx, vy, vyaw = velocity.tolist()
    # forward -> up (screen -y), +lateral(left) -> screen -x
    tip = (int(cx - vy * arrow_scale), int(cy - vx * arrow_scale))

    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 90, 90, 255))
    draw.line([cx, cy, tip[0], tip[1]], fill=(80, 200, 255, 255), width=4)
    # yaw heading tick: small perpendicular segment rotated by vyaw.
    hx = int(tip[0] + math.sin(vyaw) * 12)
    hy = int(tip[1] - math.cos(vyaw) * 12)
    draw.line([tip[0], tip[1], hx, hy], fill=(255, 220, 80, 255), width=2)

    # Inset: full velocity trajectory as a relative walk from origin.
    panel = min(120, image.height // 3, image.width // 3)
    margin = 10
    x0 = image.width - panel - margin
    y0 = margin
    ix = x0 + panel // 2
    iy = y0 + panel - panel // 6
    walk_scale = (panel * 0.4) / max(vel_range, 1e-6)
    draw.rectangle([x0, y0, x0 + panel, y0 + panel], fill=(0, 0, 0, 140))
    draw.line([x0, iy, x0 + panel, iy], fill=(90, 90, 90, 180))
    draw.line([ix, y0, ix, y0 + panel], fill=(90, 90, 90, 180))

    pts = [(ix, iy)]
    px, py = 0.0, 0.0
    for v in trajectory.tolist():
        px += v[1] * walk_scale  # accumulate lateral
        py += v[0] * walk_scale  # accumulate forward
        pts.append((int(ix - px), int(iy - py)))
    draw.line(pts, fill=(80, 200, 255, 255), width=2)
    for i, pt in enumerate(pts):
        r = 3 if i else 4
        color = (255, 90, 90, 255) if i == 0 else (80, 200, 255, 255)
        draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color)

    lines = [
        f"frame {frame_idx}  |  {instruction}",
        f"vel  vx={vx:+.3f}  vy={vy:+.3f}  vyaw={vyaw:+.3f}",
    ]
    for i, text in enumerate(lines):
        draw.text((margin, margin + i * 14), text, fill=(255, 255, 255, 255), font=font)
    draw.text((x0 + 4, y0 + 4), "vel walk", fill=(200, 200, 200, 255), font=font)
    return image


def main(args: argparse.Namespace) -> None:
    frame_paths = sorted(Path(args.images).glob("*.jpg")) + sorted(
        Path(args.images).glob("*.png")
    )
    if not frame_paths:
        raise FileNotFoundError(f"no .jpg/.png frames in {args.images}")
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not 0 <= args.control_waypoint < NUM_WAYPOINTS:
        raise ValueError(
            f"--control-waypoint must be in [0, {NUM_WAYPOINTS}), "
            f"got {args.control_waypoint}"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        hf_overrides={
            "dino_model": args.dino,
            "siglip_model": args.siglip,
            "image_size": IMAGE_SIZE,
        },
    )

    frame_window: deque[Image.Image] = deque(maxlen=WINDOW_FRAMES)
    print(f"Running per-frame inference over {len(frame_paths)} frames ...")
    for frame_idx, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB")
        frame_window.append(image)

        window = build_window(frame_window)
        outputs = llm.embed(
            [{"prompt": args.instruction, "multi_modal_data": {"image": window}}]
        )
        trajectory = torch.tensor(outputs[0].outputs.embedding).reshape(
            NUM_WAYPOINTS, ACTION_DIM
        )
        # RobotTrack waypoints are positions over dt; divide by control_dt to get
        # the body-frame velocity the released policy feeds the controller.
        velocity = trajectory[args.control_waypoint].float() / args.control_dt

        annotated = draw_overlay(
            image, velocity, trajectory, args.instruction, frame_idx, args.vel_range
        )
        annotated.save(output_dir / frame_path.name)
        if frame_idx % 10 == 0:
            print(
                f"  frame {frame_idx:4d}/{len(frame_paths)}  "
                f"vel=({velocity[0]:+.3f}, {velocity[1]:+.3f}, "
                f"{velocity[2]:+.3f})"
            )

    print(f"Saved {len(frame_paths)} annotated frames to {output_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openbmb/MiniCPM-RobotTrack")
    parser.add_argument("--dino", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--siglip", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--images", required=True, help="directory of video frames")
    parser.add_argument("--output", default="output")
    parser.add_argument("--instruction", default="Follow the person.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument(
        "--control-waypoint",
        type=int,
        default=1,
        help="which predicted waypoint to drive the overlay (release default 1)",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.1,
        help="control timestep; waypoint / dt gives m/s, rad/s",
    )
    parser.add_argument(
        "--vel-range", type=float, default=1.5, help="velocity half-extent for drawing"
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
