#!/usr/bin/env python3
"""Test script to trigger video loading and see the logs."""

import logging
import os
from pathlib import Path

# Set up logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set the environment variable
os.environ['VLLM_VIDEO_LOADER_BACKEND'] = 'pynvvideocodec'

from vllm.multimodal.video import VideoMediaIO
from vllm.multimodal.image import ImageMediaIO

print("=" * 70)
print("Step 1: Initializing VideoMediaIO")
print("=" * 70)

image_io = ImageMediaIO()
video_io = VideoMediaIO(image_io, num_frames=10)

print("\n" + "=" * 70)
print("Step 2: Loading a video file")
print("=" * 70)

# Try to load a video
video_path = Path('sharegpt4video/panda/PN77MQRGJDs.mp4')
if video_path.exists():
    print(f"\nLoading: {video_path}")
    try:
        frames, metadata = video_io.load_file(video_path)
        print(f"\n✅ Video loaded successfully!")
        print(f"   Frames shape: {frames.shape if hasattr(frames, 'shape') else 'N/A'}")
        print(f"   Metadata: {metadata}")
    except Exception as e:
        print(f"\n❌ Expected exception caught: {type(e).__name__}: {e}")
        print("   (This is the test exception at line 351)")
else:
    print(f"\n⚠️  Video not found: {video_path}")
    print("   Make sure you're running from /workflow/vllm-exp where the symlink exists")

