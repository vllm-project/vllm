# Nightly Builds of vLLM Wheels

vLLM maintains a per-commit wheel repository (commonly referred to as "nightly") at `https://wheels.vllm.ai` that provides pre-built wheels for every commit on the `main` branch since `v0.5.3`. This document explains how the nightly wheel index mechanism works.

## Build and Upload Process on CI

### Wheel Building

Wheels are built in the `Release` pipeline (`.buildkite/release-pipeline.yaml`) after a PR is merged into the main branch, with multiple variants:

- **Backend variants**: `cpu` and `cuXXX` (e.g., `cu129`, `cu130`).
- **Architecture variants**: `x86_64` and `aarch64`.

Each build step:

1. Builds the wheel in a Docker container.
2. Renames the wheel filename to use the correct manylinux tag (currently `manylinux_2_31`) for PEP 600 compliance.
3. Uploads the wheel to S3 bucket `vllm-wheels` under `/{commit_hash}/`.

### Index Generation

After uploading each wheel, the `.buildkite/scripts/upload-wheels.sh` script:

1. **Lists all existing wheels** in the commit directory from S3
2. **Generates indices** using `.buildkite/scripts/generate-nightly-index.py`:
    - Parses wheel filenames to extract metadata (version, variant, platform tags).
    - Creates HTML index files (`index.html`) for PyPI compatibility.
    - Generates machine-readable `metadata.json` files.
3. **Uploads indices** to multiple locations (overriding existing ones):
    - `/{commit_hash}/` - Always uploaded for commit-specific access.
    - `/nightly/` - Only for commits on `main` branch (not PRs).
    - `/{version}/` - Only for release wheels (no `dev` in its version).

!!! tip "Handling Concurrent Builds"
    The index generation script can handle multiple variants being built concurrently by always listing all wheels in the commit directory before generating indices, avoiding race conditions.

## Directory Structure

The S3 bucket structure follows this pattern:

```text
s3://vllm-wheels/
├── {commit_hash}/              # Commit-specific wheels and indices
│   ├── vllm-*.whl              # All wheel files
│   ├── index.html              # Project list (default variant)
│   ├── vllm/
│   │   ├── index.html          # Package index (default variant)
│   │   └── metadata.json       # Metadata (default variant)
│   ├── cu129/                  # Variant subdirectory
│   │   ├── index.html          # Project list (cu129 variant)
│   │   └── vllm/
│   │       ├── index.html      # Package index (cu129 variant)
│   │       └── metadata.json   # Metadata (cu129 variant)
│   ├── cu130/                  # Variant subdirectory
│   ├── cpu/                    # Variant subdirectory
│   └── .../                    # More variant subdirectories
├── nightly/                    # Latest main branch wheels (mirror of latest commit)
└── {version}/                  # Release version indices (e.g., 0.11.2)
```

All built wheels are stored in `/{commit_hash}/`, while different indices are generated and reference them.
This avoids duplication of wheel files.

For example, you can specify the following URLs to use different indices:

- `https://wheels.vllm.ai/nightly/cu130` for the latest main branch wheels built with CUDA 13.0.
- `https://wheels.vllm.ai/{commit_hash}` for wheels built at a specific commit (default variant).
- `https://wheels.vllm.ai/0.12.0/cpu` for 0.12.0 release wheels built for CPU variant.

Please note that not all variants are present on every commit. The available variants are subject to change over time, e.g., changing cu130 to cu131.

### Variant Organization

Indices are organized by variant:

- **Default variant**: Wheels without variant suffix (i.e., built with the current `VLLM_MAIN_CUDA_VERSION`) are placed in the root.
- **Variant subdirectories**: Wheels with variant suffixes (e.g., `+cu130`, `.cpu`) are organized in subdirectories.
- **Alias to default**: The default variant can have an alias (e.g., `cu129` for now) for consistency and convenience.

The variant is extracted from the wheel filename (as described in the [file name convention](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention)):

- The variant is encoded in the local version identifier (e.g. `+cu129` or `dev<N>+g<hash>.cu130`).
- Examples:
    - `vllm-0.11.2.dev278+gdbc3d9991-cp38-abi3-manylinux1_x86_64.whl` → default variant
    - `vllm-0.10.2rc2+cu129-cp38-abi3-manylinux2014_aarch64.whl` → `cu129` variant
    - `vllm-0.11.1rc8.dev14+gaa384b3c0.cu130-cp38-abi3-manylinux1_x86_64.whl` → `cu130` variant

## Index Generation Details

The `generate-nightly-index.py` script performs the following:

1. **Parses wheel filenames** using regex to extract:
    - Package name
    - Version (with variant extracted)
    - Python tag, ABI tag, platform tag
    - Build tag (if present)
2. **Groups wheels by variant**, then by package name:
    - Currently only `vllm` is built, but the structure supports multiple packages in the future.
3. **Generates HTML indices** (compliant with the [Simple repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/#simple-repository-api)):
    - Top-level `index.html`: Lists all packages and variant subdirectories
    - Package-level `index.html`: Lists all wheel files for that package
    - Uses relative paths to wheel files for portability
4. **Generates metadata.json**:
    - Machine-readable JSON containing all wheel metadata
    - Includes `path` field with URL-encoded relative path to wheel file
    - Used by `setup.py` to locate compatible pre-compiled wheels during Python-only builds

### Special Handling for AWS Services

The wheels and indices are directly stored on AWS S3, and we use AWS CloudFront as a CDN in front of the S3 bucket.

Since S3 does not provide proper directory listing, to support PyPI-compatible simple repository API behavior, we deploy a CloudFront Function that:

- redirects any URL that does not end with `/` and does not look like a file (i.e., does not contain a dot `.` in the last path segment) to the same URL with a trailing `/`
- appends `/index.html` to any URL that ends with `/`

For example, the following requests would be handled as:

- `/nightly` -> `/nightly/index.html`
- `/nightly/cu130/` -> `/nightly/cu130/index.html`
- `/nightly/index.html` or `/nightly/vllm.whl` -> unchanged

!!! note "AWS S3 Filename Escaping"

    S3 will automatically escape filenames upon upload according to its [naming rule](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html). The direct impact on vllm is that `+` in filenames will be converted to `%2B`. We take special care in the index generation script to escape filenames properly when generating the HTML indices and JSON metadata, to ensure the URLs are correct and can be directly used.

## Usage of precompiled wheels in `setup.py` {#precompiled-wheels-usage}

When installing vLLM with `VLLM_USE_PRECOMPILED=1`, the `setup.py` script:

1. **Determines wheel location** via `precompiled_wheel_utils.determine_wheel_url()`:
    - Env var `VLLM_PRECOMPILED_WHEEL_LOCATION` (user-specified URL/path) always takes precedence and skips all other steps.
    - Determines the variant from `VLLM_MAIN_CUDA_VERSION` (can be overridden with env var `VLLM_PRECOMPILED_WHEEL_VARIANT`); the default variant will also be tried as a fallback.
    - Determines the _base commit_ (explained later) of this branch (can be overridden with env var `VLLM_PRECOMPILED_WHEEL_COMMIT`).
2. **Fetches metadata** from `https://wheels.vllm.ai/{commit}/vllm/metadata.json` (for the default variant) or `https://wheels.vllm.ai/{commit}/{variant}/vllm/metadata.json` (for a specific variant).
3. **Selects compatible wheel** based on:
    - Package name (`vllm`)
    - Platform tag (architecture match)
4. **Downloads and extracts** precompiled binaries from the wheel:
    - C++ extension modules (`.so` files)
    - Flash Attention Python modules
    - Triton kernel Python files
5. **Patches package_data** to include extracted files in the installation

!!! note "What is the base commit?"

    The base commit is determined by finding the merge-base
    between the current branch and upstream `main`, ensuring
    compatibility between source code and precompiled binaries.

_Note: it's users' responsibility to ensure there is no native code (e.g., C++ or CUDA) changes before using precompiled wheels._

## Implementation Files

Key files involved in the nightly wheel mechanism:

- **`.buildkite/release-pipeline.yaml`**: CI pipeline that builds wheels
- **`.buildkite/scripts/upload-wheels.sh`**: Script that uploads wheels and generates indices
- **`.buildkite/scripts/generate-nightly-index.py`**: Python script that generates PyPI-compatible indices
- **`setup.py`**: Contains `precompiled_wheel_utils` class for fetching and using precompiled wheels
