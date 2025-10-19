#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone test for OCI Go client library."""

import ctypes
import json
from pathlib import Path


class ManifestResult(ctypes.Structure):
    _fields_ = [
        ("manifest_json", ctypes.c_char_p),
        ("error", ctypes.c_char_p),
    ]


def test_library_loading():
    """Test that the library loads successfully."""
    print("Testing library loading...")
    
    lib_dir = Path(__file__).parent
    lib_path = lib_dir / "liboci.so"
    
    if not lib_path.exists():
        lib_path = lib_dir / "liboci.dylib"
    
    if not lib_path.exists():
        print(f"❌ Library not found at {lib_dir}/liboci.{{so,dylib}}")
        print("Run ./build.sh to build the library first")
        return False
    
    try:
        lib = ctypes.CDLL(str(lib_path))
        print(f"✓ Library loaded successfully from {lib_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to load library: {e}")
        return False


def test_function_signatures():
    """Test that exported functions are available."""
    print("\nTesting function signatures...")
    
    lib_dir = Path(__file__).parent
    lib_path = lib_dir / "liboci.so"
    
    if not lib_path.exists():
        lib_path = lib_dir / "liboci.dylib"
    
    lib = ctypes.CDLL(str(lib_path))
    
    # Set up function signatures
    lib.PullManifest.argtypes = [ctypes.c_char_p]
    lib.PullManifest.restype = ManifestResult
    
    print("✓ Function signatures set up successfully")
    return True


def test_public_image_manifest():
    """Test pulling manifest from a public image."""
    print("\nTesting public image manifest pull...")
    print("This test requires network access")
    
    lib_dir = Path(__file__).parent
    lib_path = lib_dir / "liboci.so"
    
    if not lib_path.exists():
        lib_path = lib_dir / "liboci.dylib"
    
    lib = ctypes.CDLL(str(lib_path))
    
    # Set up function signatures
    lib.PullManifest.argtypes = [ctypes.c_char_p]
    lib.PullManifest.restype = ManifestResult
    
    # Test with a small public image (alpine)
    image_ref = b"docker.io/library/alpine:latest"
    print(f"Pulling manifest for: {image_ref.decode()}")
    
    result = lib.PullManifest(image_ref)
    
    if result.error:
        error_msg = result.error.decode("utf-8")
        print(f"❌ Error: {error_msg}")
        return False
    
    if not result.manifest_json:
        print("❌ No manifest returned")
        return False
    
    manifest_str = result.manifest_json.decode("utf-8")
    
    manifest = json.loads(manifest_str)
    
    print(f"✓ Successfully pulled manifest")
    print(f"  Schema version: {manifest.get('schemaVersion')}")
    print(f"  Media type: {manifest.get('mediaType')}")
    print(f"  Config digest: {manifest.get('config', {}).get('digest', 'N/A')[:20]}...")
    print(f"  Number of layers: {len(manifest.get('layers', []))}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("OCI Go Client Library Tests")
    print("=" * 70)
    
    tests = [
        test_library_loading,
        test_function_signatures,
        test_public_image_manifest,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test raised exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
