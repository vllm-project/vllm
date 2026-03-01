#!/usr/bin/env python3
"""
Quick smoke test runner for Nano-Nemotron-VL model.

Usage:
    python run_nano_nemotron_smoke_test.py
    
Or run specific tests with pytest:
    pytest tests/models/multimodal/processing/test_nano_nemotron_vl_smoke.py -v -s --no-skip
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from tests.models.multimodal.processing.test_nano_nemotron_vl_smoke import (
        test_nano_nemotron_vl_basic_image_preprocessing,
        test_nano_nemotron_vl_preprocessing,
        test_nano_nemotron_vl_multiple_images,
        test_nano_nemotron_vl_preprocessing_timing,
        test_image_preprocessing_resolutions,
        test_video_preprocessing_resolutions,
        MODEL_PATHS,
    )
    
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
            
            print("\n[Test 2/6] Preprocessing pipeline...")
            test_nano_nemotron_vl_preprocessing(model_type, model_path)
            
            print("\n[Test 3/6] Multiple images...")
            test_nano_nemotron_vl_multiple_images(model_type, model_path)
            
            print("\n[Test 4/6] Preprocessing timing...")
            test_nano_nemotron_vl_preprocessing_timing(model_type, model_path)
            
            print("\n[Test 5/6] Image preprocessing at multiple resolutions...")
            test_image_preprocessing_resolutions(model_type, model_path)
            
            print("\n[Test 6/6] Video preprocessing at multiple resolutions...")
            test_video_preprocessing_resolutions(model_type, model_path)
            
            print(f"\nAll tests passed for {model_type.upper()} model!")
        
        print("\n" + "=" * 80)
        print("All smoke tests passed for all models!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
