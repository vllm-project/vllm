# Docker Build & Push

This repository includes a GitHub Actions workflow that automatically builds and pushes Docker images to GitHub Container Registry.

## Automatic Builds

The workflow (`.github/workflows/docker-build-push.yml`) triggers on:
- **Push to `main`**: Creates `latest` and `main-<sha>` tags
- **Version tags (`v*`)**: Creates versioned releases (e.g., `v1.0.0`, `v1.0`)
- **Pull requests**: Builds for validation (does not push)

## Setup

### 1. Enable GitHub Actions Permissions

Go to your repository settings:
1. Settings → Actions → General
2. Under "Workflow permissions":
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"
3. Click Save

### 2. Push Your Changes

```bash
git add .github/workflows/docker-build-push.yml
git commit -m "[CI] Add Docker build and push workflow"
git push origin main
```

The workflow will automatically start building. Check the Actions tab to monitor progress.

### 3. Wait for Build

First build takes approximately 30-45 minutes. Subsequent builds are faster due to layer caching.

## Using the Images

### Pull from GitHub Container Registry

```bash
# Latest version
docker pull ghcr.io/inference-sim/vllm:latest

# Specific commit
docker pull ghcr.io/inference-sim/vllm:main-<commit-sha>

# Tagged version
docker pull ghcr.io/inference-sim/vllm:v1.0.0
```

### Run Locally

```bash
# Single GPU
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/inference-sim/vllm:latest \
  --model meta-llama/Llama-3.1-8B-Instruct

# Multi-GPU (8x H100)
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/inference-sim/vllm:latest \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 8192
```

## Build Configuration

The images are built with:
- **CUDA**: 12.9.1 (H100 optimized)
- **Python**: 3.12
- **GPU Architectures**: 7.0, 7.5, 8.0, 8.9, 9.0, 10.0, 12.0
  - Architecture 9.0 = H100/H200 GPUs
- **Target**: `vllm-openai` (production server)

## Authentication (for Private Repositories)

If your repository packages are private:

```bash
# Create GitHub Personal Access Token with read:packages scope
# Then login:
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Pull image
docker pull ghcr.io/inference-sim/vllm:latest
```

## Manual Build (Local)

To build locally instead of using CI:

```bash
docker buildx build \
  --file docker/Dockerfile \
  --build-arg CUDA_VERSION=12.9.1 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg max_jobs=16 \
  --build-arg torch_cuda_arch_list="7.0 7.5 8.0 8.9 9.0 10.0 12.0" \
  --target vllm-openai \
  --tag vllm:local \
  .
```

This takes 30-60 minutes and requires ~50GB disk space.

## Troubleshooting

### Build Fails in GitHub Actions

1. Check Actions tab for detailed logs
2. Common issues:
   - Insufficient disk space (needs ~50GB)
   - Network timeout downloading dependencies
   - Missing workflow permissions

### Cannot Pull Image

1. Verify the image exists: Go to repository → Packages
2. If private, authenticate with `docker login ghcr.io`
3. Check package visibility in Settings

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

## Workflow Details

The GitHub Actions workflow:
1. Checks out the repository with full git history
2. Sets up Docker Buildx for advanced builds
3. Logs in to GitHub Container Registry
4. Extracts metadata for tags and labels
5. Builds the image with layer caching
6. Pushes to `ghcr.io/inference-sim/vllm`

Build arguments mirror vLLM's production configuration for H100 compatibility.
