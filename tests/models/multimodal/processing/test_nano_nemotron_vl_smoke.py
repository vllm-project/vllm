# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for Nano-Nemotron-VL model with local checkpoint."""

import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY


# Local model paths
# Path 1: Dynamic resolution approach
MODEL_PATH_DYNAMIC = "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/nhaber/megatron-lm/work/megatron_hf"
# Path 2: Tiling approach
MODEL_PATH_TILING = "/lustre/fs1/portfolios/llmservice/projects/llmservice_deci_vlm/users/nbagrov/vllm_fast_preproc/nemotron_nano_vl/megatron_hf"

# All model paths to test
MODEL_PATHS = [
    ("dynamic", MODEL_PATH_DYNAMIC),
    ("tiling", MODEL_PATH_TILING),
]


def create_test_image(width=512, height=512):
    """Create a simple test image."""
    import numpy as np
    
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(255 * i / height),  # Red gradient
                int(255 * j / width),   # Green gradient
                128,                     # Constant blue
            ]
    return Image.fromarray(img_array, mode='RGB')


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_nano_nemotron_vl_basic_image_preprocessing(model_type, model_path):
    """Test basic image preprocessing with the nano nemotron VL model."""
    
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type.upper()} model: {model_path}")
    print(f"{'=' * 80}")
    
    # Create ModelConfig with local path
    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        limit_mm_per_prompt={"image": 1},
    )
    
    # Create processor
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    
    # Create test image
    test_image = create_test_image(512, 512)
    
    # Test preprocessing
    prompt = "<image>Describe this image."
    mm_data = {"image": [test_image]}
    
    mm_items = processor.info.parse_mm_data(mm_data)
    processed = processor.apply(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})
    
    # Basic validations
    assert "prompt_token_ids" in processed
    assert "mm_kwargs" in processed
    assert len(processed["prompt_token_ids"]) > 0
    
    mm_kwargs = processed["mm_kwargs"].get_data()
    assert "pixel_values_flat" in mm_kwargs
    
    pixel_values = mm_kwargs["pixel_values_flat"]
    if isinstance(pixel_values, list):
        assert len(pixel_values) > 0
        assert isinstance(pixel_values[0], torch.Tensor)
        print(f"✓ Dynamic resolution: {len(pixel_values)} tensor(s)")
        print(f"  First tensor shape: {pixel_values[0].shape}")
    else:
        assert isinstance(pixel_values, torch.Tensor)
        print(f"✓ Standard resolution: shape={pixel_values.shape}")
    
    print(f"✓ Prompt token IDs: {len(processed['prompt_token_ids'])} tokens")
    print("✓ Basic image preprocessing works!")


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_nano_nemotron_vl_preprocessing(model_type, model_path):
    """Test preprocessing pipeline."""
    
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type.upper()} model: {model_path}")
    print(f"{'=' * 80}")
    
    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        limit_mm_per_prompt={"image": 1},
    )
    
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    
    test_image = create_test_image(512, 512)
    prompt = "<image>Describe this image."
    mm_data = {"image": [test_image]}
    
    mm_items = processor.info.parse_mm_data(mm_data)
    processed = processor.apply(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})
    
    assert "mm_kwargs" in processed
    mm_kwargs = processed["mm_kwargs"].get_data()
    assert "pixel_values_flat" in mm_kwargs
    
    print("✓ Preprocessing works!")


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_nano_nemotron_vl_multiple_images(model_type, model_path):
    """Test preprocessing with multiple images."""
    
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type.upper()} model: {model_path}")
    print(f"{'=' * 80}")
    
    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        limit_mm_per_prompt={"image": 3},
    )
    
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    
    # Create multiple test images of different sizes
    images = [
        create_test_image(256, 256),
        create_test_image(512, 512),
        create_test_image(768, 768),
    ]
    
    prompt = "<image><image><image>Describe these images."
    mm_data = {"image": images}
    
    mm_items = processor.info.parse_mm_data(mm_data)
    processed = processor.apply(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})
    
    assert "mm_kwargs" in processed
    mm_kwargs = processed["mm_kwargs"].get_data()
    
    pixel_values = mm_kwargs["pixel_values_flat"]
    if isinstance(pixel_values, list):
        print(f"✓ Processed {len(images)} images → {len(pixel_values)} tensor(s)")
        for idx, pv in enumerate(pixel_values):
            print(f"  Tensor {idx}: shape={pv.shape}, dtype={pv.dtype}")
    else:
        print(f"✓ Processed {len(images)} images → concatenated tensor shape={pixel_values.shape}")
    
    print("✓ Multiple image preprocessing works!")


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_nano_nemotron_vl_preprocessing_timing(model_type, model_path):
    """Time preprocessing pipeline."""
    import time
    
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type.upper()} model: {model_path}")
    print(f"{'=' * 80}")
    
    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        limit_mm_per_prompt={"image": 1},
    )
    
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    
    test_image = create_test_image(512, 512)
    prompt = "<image>Test."
    mm_data = {"image": [test_image]}
    
    mm_items = processor.info.parse_mm_data(mm_data)
    start = time.time()
    processed = processor.apply(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})
    elapsed = time.time() - start
    
    print(f"✓ Preprocessing took {elapsed:.4f}s")
    
    mm_kwargs = processed["mm_kwargs"].get_data()
    pixel_values = mm_kwargs["pixel_values_flat"]
    
    if isinstance(pixel_values, list):
        print(f"  Output: {len(pixel_values)} tensor(s), first shape={pixel_values[0].shape}")
    else:
        print(f"  Output: shape={pixel_values.shape}, dtype={pixel_values.dtype}")


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_image_preprocessing_resolutions(model_type, model_path):
    """Test image preprocessing across multiple resolutions."""
    import time
    
    print(f"\n{'=' * 80}")
    print(f"IMAGE PREPROCESSING - {model_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'=' * 80}")
    
    resolutions = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    
    results = []
    
    for width, height in resolutions:
        print(f"\nTesting Resolution: {width}x{height}")
        
        test_image = create_test_image(width, height)
        prompt = "<image>Describe this image."
        mm_data = {"image": [test_image]}
        
        model_config = ModelConfig(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            seed=0,
            limit_mm_per_prompt={"image": 1},
        )
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        mm_items = processor.info.parse_mm_data(mm_data)
        
        start = time.time()
        processed = processor.apply(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})
        elapsed = time.time() - start
        
        mm_kwargs = processed["mm_kwargs"].get_data()
        pixel_values = mm_kwargs["pixel_values_flat"]
        
        if isinstance(pixel_values, list):
            print(f"  Time: {elapsed:.4f}s  |  {len(pixel_values)} tensor(s), "
                  f"first shape={pixel_values[0].shape}")
        else:
            print(f"  Time: {elapsed:.4f}s  |  shape={pixel_values.shape}")
        
        results.append({
            'resolution': f"{width}x{height}",
            'time': elapsed,
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - IMAGE PREPROCESSING")
    print("=" * 80)
    print(f"{'Resolution':<15} {'Time (s)':<15}")
    print("-" * 40)
    for result in results:
        print(f"{result['resolution']:<15} {result['time']:<15.4f}")
    print("=" * 80)


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_video_preprocessing_resolutions(model_type, model_path):
    """Test video preprocessing at FHD and 4K resolutions."""
    import numpy as np
    import time
    
    print(f"\n{'=' * 80}")
    print(f"VIDEO PREPROCESSING - {model_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'=' * 80}")
    
    test_configs = [
        (1920, 1080, [32, 64, 128]),  # FHD
        (3840, 2160, [32, 64, 128]),  # 4K
    ]
    
    all_results = []
    
    for video_width, video_height, frame_counts in test_configs:
        print(f"\nResolution: {video_width}x{video_height}")
        
        for num_frames in frame_counts:
            video_frames = []
            for i in range(num_frames):
                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                frame[:, :, 0] = int(255 * i / num_frames)
                frame[:, :, 1] = 128
                frame[:, :, 2] = int(255 * (1 - i / num_frames))
                video_frames.append(frame)
            
            video_array = np.array(video_frames, dtype=np.uint8)
            video_metadata = {
                "total_num_frames": num_frames,
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "video_backend": "opencv_dynamic",
                "frames_indices": list(range(num_frames)),
                "do_sample_frames": False,
            }
            
            prompt = "<video>Describe this video."
            mm_data = {"video": [(video_array, video_metadata)]}
            
            model_config = ModelConfig(
                model=model_path,
                trust_remote_code=True,
                dtype="auto",
                seed=0,
                limit_mm_per_prompt={"video": 1},
            )
            processor = MULTIMODAL_REGISTRY.create_processor(model_config)
            mm_items = processor.info.parse_mm_data(mm_data)
            
            start = time.time()
            processed = processor.apply(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})
            elapsed = time.time() - start
            
            mm_kwargs = processed["mm_kwargs"].get_data()
            pixel_values = mm_kwargs.get("pixel_values_flat_video")
            
            if pixel_values is not None:
                print(f"  {num_frames} frames: {elapsed:.4f}s "
                      f"({elapsed/num_frames*1000:.2f}ms/frame)  "
                      f"shape={pixel_values.shape}")
                all_results.append({
                    'resolution': f"{video_width}x{video_height}",
                    'num_frames': num_frames,
                    'time': elapsed,
                    'ms_per_frame': elapsed / num_frames * 1000,
                })
            else:
                print(f"  {num_frames} frames: No video output (model may not support videos)")
                return
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - VIDEO PREPROCESSING")
    print("=" * 80)
    if all_results:
        print(f"{'Resolution':<15} {'Frames':<10} {'Time (s)':<15} {'ms/frame':<12}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['resolution']:<15} {r['num_frames']:<10} "
                  f"{r['time']:<15.4f} {r['ms_per_frame']:<12.2f}")
    print("=" * 80)


if __name__ == "__main__":
    """Run smoke tests manually."""
    print("=" * 80)
    print("Running Nano-Nemotron-VL Smoke Tests")
    print("=" * 80)
    print("Model paths:")
    for model_type, model_path in MODEL_PATHS:
        print(f"  - {model_type.upper()}: {model_path}")
    print("=" * 80)
    
    try:
        for model_type, model_path in MODEL_PATHS:
            print(f"\n{'#' * 80}")
            print(f"TESTING MODEL: {model_type.upper()}")
            print(f"{'#' * 80}")
            
            print("\n[Test 1/5] Basic image preprocessing...")
            test_nano_nemotron_vl_basic_image_preprocessing(model_type, model_path)
            
            print("\n[Test 2/5] Preprocessing pipeline...")
            test_nano_nemotron_vl_preprocessing(model_type, model_path)
            
            print("\n[Test 3/5] Multiple images...")
            test_nano_nemotron_vl_multiple_images(model_type, model_path)
            
            print("\n[Test 4/5] Image preprocessing at multiple resolutions...")
            test_image_preprocessing_resolutions(model_type, model_path)
            
            print("\n[Test 5/5] Video preprocessing at multiple resolutions...")
            test_video_preprocessing_resolutions(model_type, model_path)
            
            print(f"\n All tests passed for {model_type.upper()} model!")
        
        print("\n" + "=" * 80)
        print("All smoke tests passed for all models!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise