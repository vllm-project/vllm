---
title: Update PyTorch version on vLLM OSS CI/CD
---

vLLM's current policy is to always use the latest PyTorch stable
release in CI/CD. It is standard practice to submit a PR to update the
PyTorch version as early as possible when a new [PyTorch stable
release](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence) becomes available.
This process is non-trivial due to the gap between PyTorch
releases. Using [#16859](https://github.com/vllm-project/vllm/pull/16859) as
an example, this document outlines common steps to achieve this update along with
a list of potential issues and how to address them.

## Test PyTorch release candidates (RCs)

Updating PyTorch in vLLM after the official release is not
ideal because any issues discovered at that point can only be resolved
by waiting for the next release or by implementing hacky workarounds in vLLM.
The better solution is to test vLLM with PyTorch release candidates (RC) to ensure
compatibility before each release.

PyTorch release candidates can be downloaded from PyTorch test index at https://download.pytorch.org/whl/test.
For example, torch2.7.0+cu12.8 RC can be installed using the following command:

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

When the final RC is ready for testing, it will be announced to the community
on the [PyTorch dev-discuss forum](https://dev-discuss.pytorch.org/c/release-announcements).
After this announcement, we can begin testing vLLM integration by drafting a pull request
following this 3-step process:

1. Update requirements files in https://github.com/vllm-project/vllm/tree/main/requirements
to point to the new releases for torch, torchvision, and torchaudio.
2. Use `--extra-index-url https://download.pytorch.org/whl/test/<PLATFORM>` to
get the final release candidates' wheels.  Some common platforms are `cpu`, `cu128`,
and `rocm6.2.4`.
3. As vLLM uses uv, make sure that `unsafe-best-match` strategy is set either
via `UV_INDEX_STRATEGY` env variable or via `--index-strategy unsafe-best-match`.

If failures are found in the pull request, raise them as issues on vLLM and
cc the PyTorch release team to initiate discussion on how to address them.

## Update CUDA version

The PyTorch release matrix includes both stable and experimental [CUDA versions](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix). Due to limitations, only the latest stable CUDA version (for example,
torch2.7.0+cu12.6) is uploaded to PyPI. However, vLLM may require a different CUDA version,
such as 12.8 for Blackwell support.
This complicates the process as we cannot use the out-of-the-box
`pip install torch torchvision torchaudio` command. The solution is to use
`--extra-index-url` in vLLM's Dockerfiles.

1. Use `--extra-index-url https://download.pytorch.org/whl/cu128` to install torch+cu128.
2. Other important indexes at the moment include:
    1. CPU ‒ https://download.pytorch.org/whl/cpu
    2. ROCm ‒ https://download.pytorch.org/whl/rocm6.2.4 and https://download.pytorch.org/whl/rocm6.3
    3. XPU ‒ https://download.pytorch.org/whl/xpu
3. Update .buildkite/release-pipeline.yaml and .buildkite/scripts/upload-wheels.sh to
match the CUDA version from step 1.  This makes sure that the release vLLM wheel is tested
on CI.

## Address long vLLM build time

When building vLLM with a new PyTorch/CUDA version, no cache will exist
in the vLLM sccache S3 bucket, causing the build job on CI to potentially take more than 5 hours
and timeout. Additionally, since vLLM's fastcheck pipeline runs in read-only mode,
it doesn't populate the cache, so re-running it to warm up the cache
is ineffective.

While ongoing efforts like [#17419](https://github.com/vllm-project/vllm/issues/17419)
address the long build time at its source, the current workaround is to set VLLM_CI_BRANCH
to a custom branch provided by @khluu (`VLLM_CI_BRANCH=khluu/use_postmerge_q`)
when manually triggering a build on Buildkite. This branch accomplishes two things:

1. Increase the timeout limit to 10 hours so that the build doesn't timeout.
2. Allow the compiled artifacts to be written to the vLLM sccache S3 bucket
to warm it up so that future builds are faster.

<p align="center" width="100%">
    <img width="60%" src="https://github.com/user-attachments/assets/a8ff0fcd-76e0-4e91-b72f-014e3fdb6b94">
</p>

## Update dependencies

Several vLLM dependencies, such as FlashInfer, also depend on PyTorch and need
to be updated accordingly. Rather than waiting for all of them to publish new
releases (which would take too much time), they can be built from
source to unblock the update process.

### FlashInfer
Here is how to build and install it from source with torch2.7.0+cu128 in vLLM [Dockerfile](https://github.com/vllm-project/vllm/blob/27bebcd89792d5c4b08af7a65095759526f2f9e1/docker/Dockerfile#L259-L271):

```
export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0 10.0+PTX'
export FLASHINFER_ENABLE_SM90=1
uv pip install --system --no-build-isolation "git+https://github.com/flashinfer-ai/flashinfer@v0.2.6.post1"
```

One caveat is that building FlashInfer from source adds approximately 30
minutes to the vLLM build time. Therefore, it's preferable to cache the wheel in a
public location for immediate installation, such as https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl. For future releases, contact the PyTorch release
team if you want to get the package published there.

### xFormers
Similar to FlashInfer, here is how to build and install xFormers from source:

```
export TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 8.9 9.0 10.0+PTX'
MAX_JOBS=16 uv pip install --system --no-build-isolation "git+https://github.com/facebookresearch/xformers@v0.0.30"
```

### Mamba

```
uv pip install --system --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4"
```

### causal-conv1d

```
uv pip install 'git+https://github.com/Dao-AILab/causal-conv1d@v1.5.0.post8'
```

## Update all the different vLLM platforms

Rather than attempting to update all vLLM platforms in a single pull request, it's more manageable
to handle some platforms separately. The separation of requirements and Dockerfiles
for different platforms in vLLM CI/CD allows us to selectively choose
which platforms to update. For instance, updating XPU requires the corresponding
release from https://github.com/intel/intel-extension-for-pytorch by Intel.
While https://github.com/vllm-project/vllm/pull/16859 updated vLLM to PyTorch
2.7.0 on CPU, CUDA, and ROCm, https://github.com/vllm-project/vllm/pull/17444
completed the update for XPU.
