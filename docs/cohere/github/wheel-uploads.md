# Python Wheels Upload to Google Artifact Registry

The `.github/scripts/` directory contains scripts to upload Python wheels from vLLM Docker builds to Google Artifact Registry.

## Overview

During Docker builds, Python wheels are generated and stored at `/app/cohere/dist/` inside the final images. These wheels can be extracted and uploaded to GAR for distribution.

## Method

The workflow uses **crane** which only downloads the specific filesystem layers containing the wheels (~200-500MB), not the entire image (~5-8GB)!

## Automatic Upload (GitHub Actions)

The `.github/workflows/build-and-push.yaml` workflow automatically:

1. Builds Docker images for NVIDIA, AMD, and CPU
2. Extracts wheels from the built images
3. Uploads them to separate GAR Python repositories:
   - `us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-nvidia` (for NVIDIA builds)
   - `us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-amd` (for AMD builds)
   - `us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-cpu` (for CPU builds)

This happens automatically on:

- Tag pushes (`v*.*.*`)
- Manual workflow dispatch
- Workflow calls from other workflows

## Manual Upload

Uses `crane` to download only the layers containing wheels (~200-500MB instead of 5-8GB):

```bash
# Authenticate to GCP
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

# Extract wheels from already-built image
bash .github/scripts/extract-wheels-from-registry.sh \
  us-central1-docker.pkg.dev/cohere-artifacts/cohere/vllm-rocm:abc123 \
  ./wheels

# Upload to GAR
bash .github/scripts/upload-wheels-dir-to-gar.sh ./wheels amd
```

## Installing Wheels from GAR

Once uploaded, wheels can be installed using pip:

```bash
# For AMD wheels
pip install vllm \
  --extra-index-url https://us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-amd/simple/

# For NVIDIA wheels
pip install vllm \
  --extra-index-url https://us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-nvidia/simple/

# For CPU wheels
pip install vllm \
  --extra-index-url https://us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-cpu/simple/
```

Or add to `requirements.txt`:

```text
--extra-index-url https://us-central1-python.pkg.dev/cohere-artifacts/cohere-py-forks-amd/simple/
vllm==x.y.z
```

## Authentication for Installing

To install wheels from the private GAR repository, you need to authenticate:

```bash
# Option 1: Using gcloud (recommended for development)
gcloud auth login
gcloud auth application-default login

# Option 2: Using keyring (for CI/CD)
pip install keyring keyrings.google-artifactregistry-auth
gcloud auth application-default login

# Option 3: Using .pypirc (alternative)
# Create ~/.pypirc with credentials
```

## Repository Structure

- `cohere-py-forks-nvidia/` - Python wheels built for NVIDIA GPUs (CUDA)
- `cohere-py-forks-amd/` - Python wheels built for AMD GPUs (ROCm)
- `cohere-py-forks-cpu/` - Python wheels built for CPU platforms

Each repository contains wheels tagged with:

- Commit SHA
- Version number
- Platform identifiers (e.g., `manylinux_2_17_x86_64`)

## Permissions

The GitHub Actions workflow uses Workload Identity Federation with the service account:

- `dockerbuild@cohere-artifacts.iam.gserviceaccount.com`

This service account needs the following IAM roles:

- `roles/artifactregistry.writer` - To upload artifacts
- `roles/artifactregistry.repoAdmin` - To create repositories (if needed)

## Troubleshooting

### Wheel not found in image

If the script reports "No wheel files found", check that:

1. The Docker build completed successfully
2. The build includes the wheel copying step (see `docker/Dockerfile:362-365`, `docker/Dockerfile.rocm:103-106`, or the `COPY --from=vllm-build` in `docker/Dockerfile.cpu`)

### Authentication errors

If you get authentication errors:

1. Ensure you're logged in: `gcloud auth login`
2. Configure Docker: `gcloud auth configure-docker us-central1-docker.pkg.dev`
3. Check your IAM permissions in the GCP console

### Repository doesn't exist

The script will automatically create the repository if it doesn't exist. If this fails, manually create it:

```bash
gcloud artifacts repositories create cohere-py-forks-amd \
  --location=us-central1 \
  --repository-format=python \
  --project=cohere-artifacts \
  --description="Python wheels for vLLM AMD builds"
```
