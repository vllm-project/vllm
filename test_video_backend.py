#!/usr/bin/env python3
"""Test script to check which video backend is being used."""

import sys
import os
from pathlib import Path

# Test video codec detection
def test_codec_detection(video_path, video_io=None):
    print(f"\n{'='*60}")
    print(f"Testing video: {video_path}")
    print(f"{'='*60}")
    
    if not Path(video_path).exists():
        print(f"❌ File does not exist: {video_path}")
        return
    
    print(f"✓ File exists (size: {Path(video_path).stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Read video bytes
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    
    # Test codec detection using actual VideoMediaIO if provided
    if video_io:
        try:
            is_hw_accel = video_io.is_video_code_hw_accelerated(video_bytes)
            codec, containers = video_io.get_video_codec_and_container_from_bytes(video_bytes)
            
            print(f"Codec: {codec}")
            print(f"Containers: {containers}")
            print(f"HW accelerated codecs: {video_io.hw_video_loader.hardware_accelerated_codecs()}")
            print(f"HW accelerated containers: {video_io.hw_video_loader.hardware_accelerated_containers()}")
            
            if is_hw_accel:
                print("✅ This video WILL use PyNvVideoCodec backend")
            else:
                print("⚠️  This video will use OpenCV backend")
                
        except Exception as e:
            print(f"❌ Error checking codec: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Fallback to manual check
        try:
            import io
            import av
            
            bio = io.BytesIO(video_bytes)
            with av.open(bio, mode='r') as c:
                vstreams = [s for s in c.streams if s.type == 'video']
                if not vstreams:
                    print("❌ No video streams found")
                    return
                
                s = vstreams[0]
                format_name = getattr(c.format, 'name', '')
                codec_name = getattr(s.codec_context, 'name', None)
                
                print(f"Codec: {codec_name}")
                print(f"Container: {format_name}")
                    
        except Exception as e:
            print(f"❌ Error checking codec: {e}")
            import traceback
            traceback.print_exc()

# Test VideoMediaIO initialization
def test_video_io():
    print(f"\n{'='*60}")
    print("Testing VideoMediaIO initialization")
    print(f"{'='*60}")
    
    from vllm import envs
    from vllm.multimodal.video import VideoMediaIO, VIDEO_LOADER_REGISTRY, PyNVVideoBackend
    from vllm.multimodal.image import ImageMediaIO
    
    print(f"VLLM_VIDEO_LOADER_BACKEND env: {os.getenv('VLLM_VIDEO_LOADER_BACKEND', 'NOT SET')}")
    print(f"envs.VLLM_VIDEO_LOADER_BACKEND: {envs.VLLM_VIDEO_LOADER_BACKEND}")
    
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io, num_frames=10)
    
    print(f"video_io.video_loader: {video_io.video_loader}")
    print(f"video_io.hw_video_loader: {video_io.hw_video_loader}")
    
    print(f"\nBoth paths point to PyNVVideoBackend: {video_io.video_loader.__class__.__name__ == 'PyNVVideoBackend'}")

if __name__ == "__main__":
    # Test initialization
    test_video_io()
    
    # Create VideoMediaIO instance for testing
    from vllm.multimodal.video import VideoMediaIO
    from vllm.multimodal.image import ImageMediaIO
    
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io, num_frames=10)
    
    # Test video files
    if len(sys.argv) > 1:
        for video_path in sys.argv[1:]:
            test_codec_detection(video_path, video_io)
    else:
        # Try to find a sample video
        import json
        dataset_path = "/workspace/data/sharegpt/llava_v1_5_mix665k_with_video_chatgpt72k_share4video28k.json"
        
        if Path(dataset_path).exists():
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Get first video
            for item in data[:5]:
                if 'video' in item:
                    video_rel_path = item['video']
                    # Try different base paths
                    for base in ['/workflow/vllm-exp', '/workspace/data', '.']:
                        video_path = Path(base) / video_rel_path
                        if video_path.exists():
                            test_codec_detection(str(video_path), video_io)
                            break
                    else:
                        print(f"\n⚠️  Could not find video: {video_rel_path}")
                        print(f"   Tried paths:")
                        for base in ['/workflow/vllm-exp', '/workspace/data', '.']:
                            print(f"     - {Path(base) / video_rel_path}")

