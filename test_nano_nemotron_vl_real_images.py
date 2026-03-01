# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare slow (PIL BICUBIC) vs fast (torch) preprocessing on real images
for both the tiling path (``dynamic_preprocess``) and the dynamic-resolution
path (``DynamicResolutionImageTiler``).

Usage:
    # Tiling tests only
    pytest test_nano_nemotron_vl_real_images.py \
        --image-dir /path/to/images -v -s

    # Direct CLI (runs both tiling + dynamic-tiler)
    python test_nano_nemotron_vl_real_images.py /path/to/images
"""

import sys
from pathlib import Path

import pytest
import torch
from tqdm import tqdm

from vllm.model_executor.models.nano_nemotron_vl import (
    DynamicResolutionImageTiler,
    dynamic_preprocess,
)

TOLERANCE = 0.06

# Defaults for standalone DynamicTiler construction (no model checkpoint needed)
_PATCH_SIZE = 16
_DOWNSAMPLE_RATIO = 0.5
_MIN_NUM_PATCHES = 10
_MAX_NUM_PATCHES = 0  # unlimited
_MAX_MODEL_LEN = 16384
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_image_paths(directory: str) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    paths = [p for p in root.rglob("*")
        if p.suffix.lower() == ".jpg"
    ]
    return paths


def _make_tiler(fast_preprocess: bool = False) -> DynamicResolutionImageTiler:
    return DynamicResolutionImageTiler(
        max_model_len=_MAX_MODEL_LEN,
        patch_size=_PATCH_SIZE,
        downsample_ratio=_DOWNSAMPLE_RATIO,
        min_num_patches=_MIN_NUM_PATCHES,
        max_num_patches=_MAX_NUM_PATCHES,
        norm_mean=_NORM_MEAN,
        norm_std=_NORM_STD,
        fast_preprocess=fast_preprocess,
    )


# ---------------------------------------------------------------------------
# Tiling path helpers (existing ``dynamic_preprocess``)
# ---------------------------------------------------------------------------

def preprocess_pair(image, image_size=512, max_num_tiles=12):
    slow = dynamic_preprocess(
        image,
        image_size=image_size,
        max_num_tiles=max_num_tiles,
        use_thumbnail=False,
        fast_preprocess=False,
    )
    fast = dynamic_preprocess(
        image,
        image_size=image_size,
        max_num_tiles=max_num_tiles,
        use_thumbnail=False,
        fast_preprocess=True,
    )
    return slow, fast


# ---------------------------------------------------------------------------
# DynamicTiler path helpers
# ---------------------------------------------------------------------------

def dynamic_tiler_preprocess_pair(image, text_prompt_length=20):
    """Run the DynamicResolutionImageTiler in slow then fast mode."""
    tiler_slow = _make_tiler(fast_preprocess=False)
    tiler_fast = _make_tiler(fast_preprocess=True)

    slow_tensors, slow_sizes = tiler_slow._images_to_pixel_values_lst(
        text_prompt_length=text_prompt_length,
        images=[image],
    )
    fast_tensors, fast_sizes = tiler_fast._images_to_pixel_values_lst(
        text_prompt_length=text_prompt_length,
        images=[image],
    )
    return slow_tensors, fast_tensors, slow_sizes, fast_sizes


# ---------------------------------------------------------------------------
# Shared comparison utility
# ---------------------------------------------------------------------------

def compare_patches(slow_patches, fast_patches, label=""):
    assert len(slow_patches) == len(fast_patches), (
        f"{label}: patch count {len(slow_patches)} vs {len(fast_patches)}"
    )
    max_diff = 0.0
    mean_diff = 0.0
    for s, f in zip(slow_patches, fast_patches):
        assert s.shape == f.shape, f"{label}: shape {s.shape} vs {f.shape}"
        abs_diff = (s - f).abs()
        max_diff = max(max_diff, abs_diff.max().item())
        mean_diff += abs_diff.mean().item()
    mean_diff /= len(slow_patches)
    return max_diff, mean_diff


# ---------------------------------------------------------------------------
# pytest fixtures / hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--image-dir",
        action="store",
        default=None,
        help="Directory containing images to test",
    )


@pytest.fixture(scope="session")
def image_dir(request):
    d = request.config.getoption("--image-dir")
    if d is None:
        pytest.skip("--image-dir not provided")
    return d


@pytest.fixture(scope="session")
def image_paths(image_dir):
    paths = collect_image_paths(image_dir)
    if not paths:
        pytest.skip(f"No images found in {image_dir}")
    return paths


# ---------------------------------------------------------------------------
# Tests – tiling path (dynamic_preprocess)
# ---------------------------------------------------------------------------

class TestRealImagesTiling:

    def test_all_images(self, image_paths):
        from PIL import Image

        results = []
        pbar = tqdm(image_paths, desc="Tiling: slow/fast", unit="img")
        for path in pbar:
            pbar.set_postfix_str(path.name[-30:])
            img = Image.open(path).convert("RGB")
            slow, fast = preprocess_pair(img)
            max_d, mean_d = compare_patches(slow, fast, label=path.name)
            results.append((path.name, img.size, len(slow), max_d, mean_d))
            assert max_d < TOLERANCE, (
                f"{path.name}: max_diff {max_d:.6f} >= {TOLERANCE}"
            )

        print(f"\n  Tiling: processed {len(results)} images, all within tolerance.")

    def test_per_image(self, image_paths):
        """Parametrize at runtime so each image is a separate sub-test."""
        from PIL import Image

        failures = []
        for path in tqdm(image_paths, desc="Tiling: per-image", unit="img"):
            img = Image.open(path).convert("RGB")
            slow, fast = preprocess_pair(img)
            max_d, _ = compare_patches(slow, fast, label=path.name)
            if max_d >= TOLERANCE:
                failures.append((path.name, max_d))

        assert not failures, (
            f"{len(failures)} image(s) exceeded tolerance:\n"
            + "\n".join(f"  {n}: {d:.6f}" for n, d in failures)
        )


# ---------------------------------------------------------------------------
# Tests – dynamic-resolution path (DynamicResolutionImageTiler)
# ---------------------------------------------------------------------------

class TestRealImagesDynamicTiler:

    def test_all_images(self, image_paths):
        from PIL import Image

        results = []
        pbar = tqdm(image_paths, desc="DynTiler: slow/fast", unit="img")
        for path in pbar:
            pbar.set_postfix_str(path.name[-30:])
            img = Image.open(path).convert("RGB")
            slow, fast, slow_sz, fast_sz = dynamic_tiler_preprocess_pair(img)
            assert slow_sz == fast_sz, (
                f"{path.name}: feature sizes differ: {slow_sz} vs {fast_sz}"
            )
            max_d, mean_d = compare_patches(slow, fast, label=path.name)
            results.append((path.name, img.size, len(slow), max_d, mean_d))
            assert max_d < TOLERANCE, (
                f"{path.name}: max_diff {max_d:.6f} >= {TOLERANCE}"
            )

        print(f"\n  DynTiler: processed {len(results)} images, all within tolerance.")

    def test_per_image(self, image_paths):
        from PIL import Image

        failures = []
        for path in tqdm(image_paths, desc="DynTiler: per-image", unit="img"):
            img = Image.open(path).convert("RGB")
            slow, fast, _, _ = dynamic_tiler_preprocess_pair(img)
            max_d, _ = compare_patches(slow, fast, label=path.name)
            if max_d >= TOLERANCE:
                failures.append((path.name, max_d))

        assert not failures, (
            f"{len(failures)} image(s) exceeded tolerance:\n"
            + "\n".join(f"  {n}: {d:.6f}" for n, d in failures)
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_tiling(paths):
    from PIL import Image

    print("=" * 80)
    print("  TILING PATH  (dynamic_preprocess)")
    print("=" * 80)
    fail = 0
    for path in tqdm(paths, desc="Tiling", unit="img"):
        img = Image.open(path).convert("RGB")
        slow, fast = preprocess_pair(img)
        max_d, mean_d = compare_patches(slow, fast, label=path.name)
        status = "OK" if max_d < TOLERANCE else "FAIL"
        if max_d >= TOLERANCE:
            fail += 1
        tqdm.write(
            f"[{status}] {path.name:>40s}  {str(img.size):>14s}  "
            f"patches={len(slow):>2d}  "
            f"max_diff={max_d:.6f}  mean_diff={mean_d:.6f}"
        )
    return fail


def _run_dynamic(paths):
    from PIL import Image

    print("=" * 80)
    print("  DYNAMIC-TILER PATH  (DynamicResolutionImageTiler)")
    print("=" * 80)
    fail = 0
    for path in tqdm(paths, desc="DynTiler", unit="img"):
        img = Image.open(path).convert("RGB")
        slow, fast, slow_sz, fast_sz = dynamic_tiler_preprocess_pair(img)
        max_d, mean_d = compare_patches(slow, fast, label=path.name)
        status = "OK" if max_d < TOLERANCE else "FAIL"
        if max_d >= TOLERANCE:
            fail += 1
        sz_match = "Y" if slow_sz == fast_sz else "N"
        tqdm.write(
            f"[{status}] {path.name:>40s}  {str(img.size):>14s}  "
            f"tensors={len(slow):>2d}  sz_match={sz_match}  "
            f"max_diff={max_d:.6f}  mean_diff={mean_d:.6f}"
        )
    return fail


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare slow vs fast preprocessing on real images.",
    )
    parser.add_argument("image_dir", help="Directory containing images to test")
    parser.add_argument(
        "--mode",
        choices=["tiling", "dynamic", "both"],
        default="both",
        help="Which preprocessing path to test (default: both)",
    )
    args = parser.parse_args()

    paths = collect_image_paths(args.image_dir)
    print(f"Found {len(paths)} images in {args.image_dir}\n")

    results = {}
    if args.mode in ("tiling", "both"):
        results["Tiling"] = _run_tiling(paths)
        print()
    if args.mode in ("dynamic", "both"):
        results["DynTiler"] = _run_dynamic(paths)
        print()

    total = len(paths)
    print("=" * 80)
    for name, fail in results.items():
        print(f"{name + ':':12s} {total - fail}/{total} passed  "
              f"({fail} failures)")
    print("=" * 80)