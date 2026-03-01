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
def test_nano_nemotron_vl_fast_preprocessing(model_type, model_path):
    """Test fast preprocessing mode."""
    
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type.upper()} model: {model_path}")
    print(f"{'=' * 80}")
    
    # Test with fast_preprocess enabled
    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        mm_processor_kwargs={"fast_preprocess": True},
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
    
    print("✓ Fast preprocessing mode works!")


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
@pytest.mark.parametrize("fast_preprocess", [False, True])
def test_nano_nemotron_vl_preprocessing_comparison(model_type, model_path, fast_preprocess):
    """Compare standard vs fast preprocessing."""
    import time
    
    mode = "FAST" if fast_preprocess else "STANDARD"
    print(f"\n{'=' * 80}")
    print(f"Testing {model_type.upper()} model ({mode}): {model_path}")
    print(f"{'=' * 80}")
    
    model_config = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        mm_processor_kwargs={"fast_preprocess": fast_preprocess},
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
    
    print(f"✓ {mode} preprocessing took {elapsed:.4f}s")
    
    mm_kwargs = processed["mm_kwargs"].get_data()
    pixel_values = mm_kwargs["pixel_values_flat"]
    
    if isinstance(pixel_values, list):
        print(f"  Output: {len(pixel_values)} tensor(s), first shape={pixel_values[0].shape}")
    else:
        print(f"  Output: shape={pixel_values.shape}, dtype={pixel_values.dtype}")


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_compare_image_preprocessing_modes(model_type, model_path):
    """Compare standard vs fast preprocessing for images across multiple resolutions."""
    import numpy as np
    import time
    
    print(f"\n{'=' * 80}")
    print(f"IMAGE PREPROCESSING COMPARISON - {model_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'=' * 80}")
    
    # Test different resolutions
    resolutions = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    
    results = []
    
    for width, height in resolutions:
        print(f"\n{'=' * 80}")
        print(f"Testing Resolution: {width}x{height}")
        print(f"{'=' * 80}")
        
        # Create test image
        test_image = create_test_image(width, height)
        prompt = "<image>Describe this image."
        mm_data = {"image": [test_image]}
        
        # Test with standard preprocessing
        print(f"\n[1/2] Running STANDARD preprocessing...")
        model_config_standard = ModelConfig(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            seed=0,
            mm_processor_kwargs={"fast_preprocess": False},
            limit_mm_per_prompt={"image": 1},
        )
        processor_standard = MULTIMODAL_REGISTRY.create_processor(model_config_standard)
        mm_items_standard = processor_standard.info.parse_mm_data(mm_data)
        
        start = time.time()
        processed_standard = processor_standard.apply(prompt, mm_items=mm_items_standard, hf_processor_mm_kwargs={})
        time_standard = time.time() - start
        
        mm_kwargs_standard = processed_standard["mm_kwargs"].get_data()
        pixel_values_standard = mm_kwargs_standard["pixel_values_flat"]
        
        print(f"  ✓ Time: {time_standard:.4f}s")
        if isinstance(pixel_values_standard, list):
            print(f"  ✓ Output: {len(pixel_values_standard)} tensor(s)")
            print(f"  ✓ First tensor shape: {pixel_values_standard[0].shape}")
            print(f"  ✓ Stats: min={pixel_values_standard[0].min():.4f}, max={pixel_values_standard[0].max():.4f}, mean={pixel_values_standard[0].mean():.4f}")
        else:
            print(f"  ✓ Output shape: {pixel_values_standard.shape}")
            print(f"  ✓ Stats: min={pixel_values_standard.min():.4f}, max={pixel_values_standard.max():.4f}, mean={pixel_values_standard.mean():.4f}")
        
        # Test with fast preprocessing
        print(f"\n[2/2] Running FAST preprocessing...")
        model_config_fast = ModelConfig(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            seed=0,
            mm_processor_kwargs={"fast_preprocess": True},
            limit_mm_per_prompt={"image": 1},
        )
        processor_fast = MULTIMODAL_REGISTRY.create_processor(model_config_fast)
        mm_items_fast = processor_fast.info.parse_mm_data(mm_data)
        
        start = time.time()
        processed_fast = processor_fast.apply(prompt, mm_items=mm_items_fast, hf_processor_mm_kwargs={})
        time_fast = time.time() - start
        
        mm_kwargs_fast = processed_fast["mm_kwargs"].get_data()
        pixel_values_fast = mm_kwargs_fast["pixel_values_flat"]
        
        print(f"  ✓ Time: {time_fast:.4f}s")
        if isinstance(pixel_values_fast, list):
            print(f"  ✓ Output: {len(pixel_values_fast)} tensor(s)")
            print(f"  ✓ First tensor shape: {pixel_values_fast[0].shape}")
            print(f"  ✓ Stats: min={pixel_values_fast[0].min():.4f}, "
                  f"max={pixel_values_fast[0].max():.4f}, "
                  f"mean={pixel_values_fast[0].mean():.4f}")
        else:
            print(f"  ✓ Output shape: {pixel_values_fast.shape}")
            print(f"  ✓ Stats: min={pixel_values_fast.min():.4f}, "
                  f"max={pixel_values_fast.max():.4f}, "
                  f"mean={pixel_values_fast.mean():.4f}")
        
        # Compare tensors numerically
        print(f"\n  📊 Tensor Comparison:")
        
        # Handle both list and tensor formats
        if isinstance(pixel_values_standard, list) and isinstance(pixel_values_fast, list):
            if len(pixel_values_standard) == len(pixel_values_fast):
                shapes_match = all(
                    s.shape == f.shape 
                    for s, f in zip(pixel_values_standard, pixel_values_fast)
                )
                if shapes_match:
                    print(f"  ✓ Shapes match: {len(pixel_values_standard)} tensor(s)")
                    
                    # Compute differences for all tensors
                    all_diffs = []
                    for idx, (std, fast) in enumerate(
                        zip(pixel_values_standard, pixel_values_fast)
                    ):
                        diff = torch.abs(std - fast)
                        all_diffs.append(diff)
                    
                    # Aggregate statistics
                    max_diff = max(d.max().item() for d in all_diffs)
                    mean_diff = torch.cat(
                        [d.flatten() for d in all_diffs]
                    ).mean().item()
                    
                    print(f"     Max absolute diff:  {max_diff:.6f}")
                    print(f"     Mean absolute diff: {mean_diff:.6f}")
                else:
                    print(f"  ⚠ Shape mismatch in list tensors")
                    max_diff = None
                    mean_diff = None
                    shapes_match = False
            else:
                print(f"  ⚠ Different number of tensors: "
                      f"{len(pixel_values_standard)} vs {len(pixel_values_fast)}")
                max_diff = None
                mean_diff = None
                shapes_match = False
        elif not isinstance(pixel_values_standard, list) and not isinstance(
            pixel_values_fast, list
        ):
            shapes_match = pixel_values_standard.shape == pixel_values_fast.shape
            if shapes_match:
                print(f"  ✓ Shapes match: {pixel_values_standard.shape}")
                
                diff = torch.abs(pixel_values_standard - pixel_values_fast)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                print(f"     Max absolute diff:  {max_diff:.6f}")
                print(f"     Mean absolute diff: {mean_diff:.6f}")
            else:
                print(f"  ⚠ Shape mismatch!")
                print(f"     Standard: {pixel_values_standard.shape}")
                print(f"     Fast:     {pixel_values_fast.shape}")
                max_diff = None
                mean_diff = None
                shapes_match = False
        else:
            print(f"  ⚠ Format mismatch: one is list, other is tensor")
            max_diff = None
            mean_diff = None
            shapes_match = False
        
        # Check tolerance
        tolerance = 0.1
        if max_diff is not None:
            within_tolerance = max_diff < tolerance
            if within_tolerance:
                print(f"     ✓ Within tolerance ({tolerance})")
            else:
                print(f"     ⚠ Exceeds tolerance ({tolerance})")
        
        # Calculate speedup
        speedup = time_standard / time_fast if time_fast > 0 else float('inf')
        
        results.append({
            'resolution': f"{width}x{height}",
            'time_standard': time_standard,
            'time_fast': time_fast,
            'speedup': speedup,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'shapes_match': shapes_match,
        })
        
        print(f"\n  ⚡ Speedup: {speedup:.2f}x")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY - IMAGE PREPROCESSING COMPARISON")
    print("=" * 80)
    print(f"{'Resolution':<15} {'Standard (s)':<15} {'Fast (s)':<15} "
          f"{'Speedup':<12} {'Max Diff':<12}")
    print("-" * 80)
    for result in results:
        max_diff_str = (
            f"{result['max_diff']:.6f}" 
            if result['max_diff'] is not None 
            else "N/A"
        )
        print(f"{result['resolution']:<15} "
              f"{result['time_standard']:<15.4f} "
              f"{result['time_fast']:<15.4f} "
              f"{result['speedup']:<12.2f}x "
              f"{max_diff_str:<12}")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print("-" * 80)
    print(f"{'Average Speedup:':<45} {avg_speedup:.2f}x")
    
    # Check if all within tolerance
    all_within_tolerance = all(
        r['max_diff'] is not None and r['max_diff'] < 0.1 
        for r in results
    )
    if all_within_tolerance:
        print(f"{'Numerical Accuracy:':<45} "
              f"✓ All within tolerance (0.1)")
    else:
        print(f"{'Numerical Accuracy:':<45} "
              f"⚠ Some exceed tolerance or N/A")
    
    print("\n✓ Image preprocessing comparison complete!")
    print("=" * 80)


@pytest.mark.skip(reason="Smoke test - only run manually with local model")
@pytest.mark.parametrize("model_type,model_path", MODEL_PATHS)
def test_compare_video_preprocessing_modes(model_type, model_path):
    """Compare standard vs fast preprocessing for videos at FHD and 4K resolutions."""
    import numpy as np
    import time
    
    print(f"\n{'=' * 80}")
    print(f"VIDEO PREPROCESSING COMPARISON - {model_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'=' * 80}")
    
    # Test different resolutions and frame counts
    test_configs = [
        # (width, height, frame_counts)
        (1920, 1080, [32, 64, 128]),  # FHD
        (3840, 2160, [32, 64, 128]),  # 4K
    ]
    
    all_results = []
    
    for video_width, video_height, frame_counts in test_configs:
        print(f"\n{'#' * 80}")
        print(f"TESTING RESOLUTION: {video_width}x{video_height}")
        print(f"{'#' * 80}")
        
        results_for_resolution = []
        
        for num_frames in frame_counts:
            print(f"\n{'=' * 80}")
            print(f"Testing {num_frames} frames at {video_width}x{video_height}")
            print(f"{'=' * 80}")
            
            # Create synthetic video
            video_frames = []
            for i in range(num_frames):
                # Create frames with changing patterns
                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                frame[:, :, 0] = int(255 * i / num_frames)  # Red gradient over time
                frame[:, :, 1] = 128  # Green constant
                frame[:, :, 2] = int(255 * (1 - i / num_frames))  # Blue decreases
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
            
            # Test with standard preprocessing
            print(f"\n[1/2] Running STANDARD video preprocessing...")
            model_config_standard = ModelConfig(
                model=model_path,
                trust_remote_code=True,
                dtype="auto",
                seed=0,
                mm_processor_kwargs={"fast_preprocess": False},
                limit_mm_per_prompt={"video": 1},
            )
            processor_standard = MULTIMODAL_REGISTRY.create_processor(model_config_standard)
            mm_items_standard = processor_standard.info.parse_mm_data(mm_data)
            
            start = time.time()
            processed_standard = processor_standard.apply(prompt, mm_items=mm_items_standard, hf_processor_mm_kwargs={})
            time_standard = time.time() - start
            
            mm_kwargs_standard = processed_standard["mm_kwargs"].get_data()
            pixel_values_standard = mm_kwargs_standard.get("pixel_values_flat_video")
            
            if pixel_values_standard is not None:
                print(f"  ✓ Time: {time_standard:.4f}s ({time_standard/num_frames*1000:.2f}ms/frame)")
                print(f"  ✓ Output shape: {pixel_values_standard.shape}")
                print(f"  ✓ Stats: min={pixel_values_standard.min():.4f}, max={pixel_values_standard.max():.4f}, mean={pixel_values_standard.mean():.4f}")
            else:
                print("  ⚠ No video output (model may not support videos)")
                return
            
            # Test with fast preprocessing
            print(f"\n[2/2] Running FAST video preprocessing...")
            model_config_fast = ModelConfig(
                model=model_path,
                trust_remote_code=True,
                dtype="auto",
                seed=0,
                mm_processor_kwargs={"fast_preprocess": True},
                limit_mm_per_prompt={"video": 1},
            )
            processor_fast = MULTIMODAL_REGISTRY.create_processor(model_config_fast)
            mm_items_fast = processor_fast.info.parse_mm_data(mm_data)
            
            start = time.time()
            processed_fast = processor_fast.apply(prompt, mm_items=mm_items_fast, hf_processor_mm_kwargs={})
            time_fast = time.time() - start
            
            mm_kwargs_fast = processed_fast["mm_kwargs"].get_data()
            pixel_values_fast = mm_kwargs_fast.get("pixel_values_flat_video")
            
            if pixel_values_fast is not None:
                print(f"  ✓ Time: {time_fast:.4f}s ({time_fast/num_frames*1000:.2f}ms/frame)")
                print(f"  ✓ Output shape: {pixel_values_fast.shape}")
                print(f"  ✓ Stats: min={pixel_values_fast.min():.4f}, max={pixel_values_fast.max():.4f}, mean={pixel_values_fast.mean():.4f}")
            
            # Calculate speedup and numerical differences
            speedup = time_standard / time_fast if time_fast > 0 else float('inf')
            
            # Check shape match
            shapes_match = pixel_values_standard.shape == pixel_values_fast.shape
            if shapes_match:
                print(f"\n  ✓ Shapes match: {pixel_values_standard.shape}")
                
                # Compute numerical difference
                diff = torch.abs(pixel_values_standard - pixel_values_fast)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                print(f"  📊 Numerical Difference:")
                print(f"     Max absolute diff:  {max_diff:.6f}")
                print(f"     Mean absolute diff: {mean_diff:.6f}")
                
                # Check if differences are acceptable
                tolerance = 0.1
                within_tolerance = max_diff < tolerance
                if within_tolerance:
                    print(f"     ✓ Within tolerance ({tolerance})")
                else:
                    print(f"     ⚠ Exceeds tolerance ({tolerance})")
            else:
                print(f"\n  ⚠ Shape mismatch!")
                print(f"     Standard: {pixel_values_standard.shape}")
                print(f"     Fast:     {pixel_values_fast.shape}")
                max_diff = None
                mean_diff = None
                shapes_match = False
            
            print(f"\n  ⚡ Speedup: {speedup:.2f}x")
            
            result = {
                'resolution': f"{video_width}x{video_height}",
                'num_frames': num_frames,
                'time_standard': time_standard,
                'time_fast': time_fast,
                'speedup': speedup,
                'time_per_frame_standard': time_standard / num_frames * 1000,
                'time_per_frame_fast': time_fast / num_frames * 1000,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shapes_match': shapes_match,
            }
            results_for_resolution.append(result)
            all_results.append(result)
        
        # Per-resolution summary
        print(f"\n{'=' * 80}")
        print(f"SUMMARY - {video_width}x{video_height}")
        print(f"{'=' * 80}")
        
        if results_for_resolution:
            print(f"{'Frames':<10} {'Standard (s)':<15} {'Fast (s)':<15} {'Speedup':<12} {'Std ms/f':<12} {'Fast ms/f':<12}")
            print("-" * 80)
            for result in results_for_resolution:
                print(f"{result['num_frames']:<10} {result['time_standard']:<15.4f} {result['time_fast']:<15.4f} "
                      f"{result['speedup']:<12.2f}x {result['time_per_frame_standard']:<12.2f} "
                      f"{result['time_per_frame_fast']:<12.2f}")
            
            avg_speedup_res = sum(r['speedup'] for r in results_for_resolution) / len(results_for_resolution)
            print("-" * 80)
            print(f"Average Speedup for {video_width}x{video_height}: {avg_speedup_res:.2f}x")
        else:
            print("  ⚠ No results collected for this resolution")
    
    # Overall summary table
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - VIDEO PREPROCESSING COMPARISON (ALL RESOLUTIONS)")
    print("=" * 80)
    
    if all_results:
        print(f"{'Resolution':<15} {'Frames':<10} {'Standard (s)':<15} {'Fast (s)':<15} {'Speedup':<12} {'Std ms/f':<12} {'Fast ms/f':<12} {'Max Diff':<12}")
        print("-" * 80)
        for result in all_results:
            max_diff_str = f"{result['max_diff']:.6f}" if result['max_diff'] is not None else "N/A"
            print(f"{result['resolution']:<15} {result['num_frames']:<10} {result['time_standard']:<15.4f} {result['time_fast']:<15.4f} "
                  f"{result['speedup']:<12.2f}x {result['time_per_frame_standard']:<12.2f} "
                  f"{result['time_per_frame_fast']:<12.2f} {max_diff_str:<12}")
        
        # Calculate overall statistics
        avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
        print("-" * 80)
        print(f"{'Overall Average Speedup:':<45} {avg_speedup:.2f}x")
        
        all_within_tolerance = all(r['max_diff'] is not None and r['max_diff'] < 0.1 for r in all_results)
        if all_within_tolerance:
            print(f"{'Numerical Accuracy:':<45} ✓ All within tolerance (0.1)")
        else:
            print(f"{'Numerical Accuracy:':<45} ⚠ Some exceed tolerance")
        
        # Resolution-specific averages
        print("\n" + "-" * 80)
        print("Average Speedup by Resolution:")
        print("-" * 80)
        resolutions_tested = list(set(r['resolution'] for r in all_results))
        for res in resolutions_tested:
            res_results = [r for r in all_results if r['resolution'] == res]
            if res_results:
                res_avg_speedup = sum(r['speedup'] for r in res_results) / len(res_results)
                print(f"  {res:<15} {res_avg_speedup:.2f}x")
    else:
        print("  ⚠ No results collected")
    
    print("\n✓ Video preprocessing comparison complete!")
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
            
            print("\n[Test 1/6] Basic image preprocessing...")
            test_nano_nemotron_vl_basic_image_preprocessing(model_type, model_path)
            
            print("\n[Test 2/6] Fast preprocessing mode...")
            test_nano_nemotron_vl_fast_preprocessing(model_type, model_path)
            
            print("\n[Test 3/6] Multiple images...")
            test_nano_nemotron_vl_multiple_images(model_type, model_path)
            
            print("\n[Test 4/6] Simple timing comparison...")
            print("\n  Standard mode:")
            test_nano_nemotron_vl_preprocessing_comparison(model_type, model_path, fast_preprocess=False)
            print("\n  Fast mode:")
            test_nano_nemotron_vl_preprocessing_comparison(model_type, model_path, fast_preprocess=True)
            
            print("\n[Test 5/6] Detailed IMAGE comparison (Standard vs Fast)...")
            test_compare_image_preprocessing_modes(model_type, model_path)
            
            print("\n[Test 6/6] Detailed VIDEO comparison (Standard vs Fast)...")
            test_compare_video_preprocessing_modes(model_type, model_path)
            
            print(f"\n✓ All tests passed for {model_type.upper()} model!")
        
        print("\n" + "=" * 80)
        print("✓ All smoke tests passed for all models!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise