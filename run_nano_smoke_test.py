#!/usr/bin/env python3
"""
Quick smoke test runner for Nano-Nemotron-VL model.

Usage:
    python run_nano_nemotron_smoke_test.py
    
Or run specific tests with pytest:
    pytest tests/models/multimodal/processing/test_nano_nemotron_vl_smoke.py -v -s --no-skip
"""
# # ==== START === Setup for attach-helper ====
# import os
# import sys
# try:
#     sys.path.insert(0, os.environ["ATTACH_HELPER_INSTALLATION_PATH"])
#     from attach_helper import debugging_setup
#     debugging_setup()
# except Exception as e:
#     print(f"Error setting up attach-helper: {e}")
#     pass
# # ==== END === Setup for attach-helper ====
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from tests.models.multimodal.processing.test_nano_nemotron_vl_smoke import (
        test_nano_nemotron_vl_basic_image_preprocessing,
        test_nano_nemotron_vl_fast_preprocessing,
        test_nano_nemotron_vl_multiple_images,
        test_nano_nemotron_vl_preprocessing_comparison,
        test_compare_image_preprocessing_modes,
        test_compare_video_preprocessing_modes,
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
            
            print("\n[Test 2/6] Fast preprocessing mode...")
            test_nano_nemotron_vl_fast_preprocessing(model_type, model_path)
            
            print("\n[Test 3/6] Multiple images...")
            test_nano_nemotron_vl_multiple_images(model_type, model_path)
            
            print("\n[Test 4/6] Standard preprocessing (timing)...")
            test_nano_nemotron_vl_preprocessing_comparison(model_type, model_path, fast_preprocess=False)
            
            print("\n[Test 5/6] Fast preprocessing (timing)...")
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
        sys.exit(1)