# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Smoke-test preprocessing on real images for both the tiling path
(``dynamic_preprocess``) and the dynamic-resolution path
(``DynamicResolutionImageTiler``).

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


def _make_tiler() -> DynamicResolutionImageTiler:
    return DynamicResolutionImageTiler(
        max_model_len=_MAX_MODEL_LEN,
        patch_size=_PATCH_SIZE,
        downsample_ratio=_DOWNSAMPLE_RATIO,
        min_num_patches=_MIN_NUM_PATCHES,
        max_num_patches=_MAX_NUM_PATCHES,
        norm_mean=_NORM_MEAN,
        norm_std=_NORM_STD,
    )


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

        pbar = tqdm(image_paths, desc="Tiling", unit="img")
        for path in pbar:
            pbar.set_postfix_str(path.name[-30:])
            img = Image.open(path).convert("RGB")
            patches = dynamic_preprocess(
                img,
                image_size=512,
                max_num_tiles=12,
                use_thumbnail=False,
            )
            assert len(patches) > 0, f"{path.name}: no patches produced"
            for p in patches:
                assert isinstance(p, torch.Tensor), (
                    f"{path.name}: patch is not a tensor"
                )

        print(f"\n  Tiling: processed {len(image_paths)} images successfully.")


# ---------------------------------------------------------------------------
# Tests – dynamic-resolution path (DynamicResolutionImageTiler)
# ---------------------------------------------------------------------------

class TestRealImagesDynamicTiler:

    def test_all_images(self, image_paths):
        from PIL import Image

        tiler = _make_tiler()
        pbar = tqdm(image_paths, desc="DynTiler", unit="img")
        for path in pbar:
            pbar.set_postfix_str(path.name[-30:])
            img = Image.open(path).convert("RGB")
            tensors, sizes = tiler._images_to_pixel_values_lst(
                text_prompt_length=20,
                images=[img],
            )
            assert len(tensors) > 0, f"{path.name}: no tensors produced"
            assert len(sizes) > 0, f"{path.name}: no sizes produced"

        print(f"\n  DynTiler: processed {len(image_paths)} images successfully.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_tiling(paths):
    from PIL import Image

    print("=" * 80)
    print("  TILING PATH  (dynamic_preprocess)")
    print("=" * 80)
    for path in tqdm(paths, desc="Tiling", unit="img"):
        img = Image.open(path).convert("RGB")
        patches = dynamic_preprocess(
            img, image_size=512, max_num_tiles=12, use_thumbnail=False,
        )
        tqdm.write(
            f"[OK] {path.name:>40s}  {str(img.size):>14s}  "
            f"patches={len(patches):>2d}"
        )


def _run_dynamic(paths):
    from PIL import Image

    print("=" * 80)
    print("  DYNAMIC-TILER PATH  (DynamicResolutionImageTiler)")
    print("=" * 80)
    tiler = _make_tiler()
    for path in tqdm(paths, desc="DynTiler", unit="img"):
        img = Image.open(path).convert("RGB")
        tensors, sizes = tiler._images_to_pixel_values_lst(
            text_prompt_length=20,
            images=[img],
        )
        tqdm.write(
            f"[OK] {path.name:>40s}  {str(img.size):>14s}  "
            f"tensors={len(tensors):>2d}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test preprocessing on real images.",
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

    if args.mode in ("tiling", "both"):
        _run_tiling(paths)
        print()
    if args.mode in ("dynamic", "both"):
        _run_dynamic(paths)
        print()

    print("=" * 80)
    print(f"Processed {len(paths)} images successfully.")
    print("=" * 80)
