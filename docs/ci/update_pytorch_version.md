---
title: Update PyTorch version on vLLM OSS CI/CD
---

The current vLLM's policy is to always use the latest PyTorch stable
release in CI/CD, so it’s a standard practice to submit a PR to update
PyTorch version as early as possible when there is a new [PyTorch stable
release](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence).
This is not a trivial process because of the complex integration between
[#16859](https://github.com/vllm-project/vllm/pull/16859)
an example, this documents common steps to achieve that together with
the list of potential issues and how to address them.

## Test PyTorch release candidates (RCs)

Updating the PyTorch version on vLLM after PyTorch has been released is not
ideal because any issues from PyTorch found at that point could only be resolved
by waiting for the next release or by providing hacky workarounds on vLLM.
The solution is to test vLLM with PyTorch release candidates to ensure that
they are working with each other before release.

PyTorch RCs are accessible via PyTorch test index at https://download.pytorch.org/whl/test.
For example, torch2.7.0+cu12.8 RC can be installed using the following command:

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

Once a final RC is ready for testing, its announcement to the community can be
found on [PyTorch dev-discuss forum](https://dev-discuss.pytorch.org/c/release-announcements).
This is the signal that we can start testing vLLM integration by
drafting a pull request following the 3-step process:

1. Update requirements files in https://github.com/vllm-project/vllm/tree/main/requirements
to point to the new releases for torch, torchvision, and torchaudio.
2. Use `--extra-index-url https://download.pytorch.org/whl/test/<PLATFORM>` to
get the final RC's wheels.  Some common platforms are `cpu`, `cu128`,
and `rocm6.2.4`.
3. As vLLM uses uv, make sure that `unsafe-best-match` strategy is set either
via `UV_INDEX_STRATEGY` env variable or via `--index-strategy unsafe-best-match`.

If failures were found in the pull request, the next step would be to raise them
as issues on vLLM and cc PyTorch release team to start the discussion.

## Update CUDA version

PyTorch release matrix has the concept of stable and experimental [CUDA versions](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix).  Due to its limitation, only the latest stable CUDA version, for example
torch2.7.0+cu12.6, is uploaded to PyPI.  On the other hand, vLLM might be looking
to use a different CUDA version, for example 12.8 for Blackwell support.
This makes the process more complicated because we couldn’t use out-of-the-box
`pip install torch torchvision torchaudio` command.  The solution is to use
`--extra-index-url` in Dockerfile.

1. Use `--extra-index-url https://download.pytorch.org/whl/cu128` to install torch+cu128.
2. Other important indexes at the moment include:
    1. CPU ‒ https://download.pytorch.org/whl/cpu
    2. ROCm ‒ https://download.pytorch.org/whl/rocm6.2.4 and https://download.pytorch.org/whl/rocm6.3
    3. XPU ‒ https://download.pytorch.org/whl/xpu
3. Update .buildkite/release-pipeline.yaml and .buildkite/scripts/upload-wheels.sh to
match the CUDA version from step 1.  This makes sure that the release vLLM wheel is tested
on CI.

## Address long vLLM build time

When building vLLM with a new PyTorch / CUDA version, there won’t be any cache
on vLLM sccache S3 bucket and the build job on CI could take more than 5 hours,
which will timeout.  In addition, running in read-only mode, vLLM PR’s fastcheck
pipeline doesn’t populate the cache, so rerunning to warm up the cache doesn’t
work.

While there are ongoing efforts like #17419 to fix the long build time at its
source, the current workaround is to point VLLM_CI_BRANCH to a custom branch
`VLLM_CI_BRANCH=khluu/use_postmerge_q` when triggering a build manually on
Buildkite.  This branch does 2 things:

1. Increase the timeout limit to 10 hours so that the build doesn’t timeout.
2. Allow the compiled artifacts to be written to the vLLM sccache S3 bucket
to warm it up so that future builds will then be faster.

<p align="center" width="100%">
    <img width="60%" src="https://github.com/user-attachments/assets/a8ff0fcd-76e0-4e91-b72f-014e3fdb6b94">
</p>

## Update dependencies

Some vLLM dependencies like FlashInfer also depend on PyTorch and will need
to be updated accordingly.  However, waiting for all of them to publish new
releases is going to take too much time.  Instead, they will be built from
source for the time being.

### FlashInfer
Here is how to build and install it from source with torch2.7.0+cu128 as done in vLLM [Dockerfile](https://github.com/vllm-project/vllm/blob/27bebcd89792d5c4b08af7a65095759526f2f9e1/docker/Dockerfile#L259-L271):

```
export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0 10.0+PTX'
export FLASHINFER_ENABLE_SM90=1
uv pip install --system --no-build-isolation "git+https://github.com/flashinfer-ai/flashinfer@v0.2.6.post1"
```

A small gotcha here is that building FlashInfer from source adds around 30
minutes to vLLM build time.  Thus, it’s better to cache the wheel somewhere
public so that it can be installed right away, for example https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl.  In future releases, please reach out to PyTorch release
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

Instead of trying to update all of them in one go, it’s ok to deal with some
platforms later.  The separation of requirements and Dockerfiles for different
platforms on vLLM gives us the ability to pick and choose the platforms to update.
For example, updating XPU requires a corresponding release of https://github.com/intel/intel-extension-for-pytorch
package and it’s better to ask Intel folks to handle that instead.  So, while https://github.com/vllm-project/vllm/pull/16859
updated vLLM to PyTorch 2.7.0 on CPU, CUDA, and ROCm, https://github.com/vllm-project/vllm/pull/17444 finished it for XPU.
